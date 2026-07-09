import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.storyboard_agent import StoryboardAgent
from app.agents.dialogue_agent import DialogueAgent
from app.agents.director_agent import DirectorAgent
from app.agents.comic_agent import ComicAgent
from app.agents.layout_agent import LayoutAgent
from app.db.models import Story, Comic, ComicPage, GenerationTask, PromptVersion, Scene

logger = logging.getLogger(__name__)


class ComicService:
    """漫画生成流水线编排：串行调用 5 个 Agent + GenerationTask 追踪。"""

    def __init__(
        self,
        storyboard_agent: StoryboardAgent,
        dialogue_agent: DialogueAgent,
        director_agent: DirectorAgent,
        comic_agent: ComicAgent,
        layout_agent: LayoutAgent,
    ):
        self.storyboard_agent = storyboard_agent
        self.dialogue_agent = dialogue_agent
        self.director_agent = director_agent
        self.comic_agent = comic_agent
        self.layout_agent = layout_agent

    async def generate_comic(self, story_id: int, style_key: str, session: AsyncSession) -> Optional[dict]:
        """完整的漫画生成流水线。

        Pipeline:
        1. 验证 Story 存在
        2. 创建 Comic 记录
        3. 加载风格配置
        4. 构建 story_graph
        5-9. 依次运行 5 个 Agent
        10. 异常处理
        """
        # 1. 验证 Story 存在
        stmt = select(Story).where(Story.id == story_id)
        result = await session.execute(stmt)
        story = result.scalar_one_or_none()
        if not story:
            return None

        # 2. 创建 Comic 记录
        comic = Comic(story_id=story_id, style=style_key, status="processing")
        session.add(comic)
        await session.flush()
        comic_id = comic.id

        try:
            # 3. 加载风格配置
            stmt = select(PromptVersion).where(
                PromptVersion.style_key == style_key,
                PromptVersion.is_active == 1
            )
            result = await session.execute(stmt)
            style_pv = result.scalar_one_or_none()
            if not style_pv:
                comic.status = "failed"
                await session.commit()
                return None

            style_config = {
                "style_key": style_pv.style_key,
                "style_name": style_pv.style_name,
                "prompt_template": style_pv.prompt_template,
                "mood": style_pv.mood,
                "dialogue_level": style_pv.dialogue_level,
                "cinematic": style_pv.cinematic,
            }

            # 4. 构建 story_graph（轻量版，类似 StoryService.get_story_graph）
            story_graph = await self._build_story_graph(story_id, session)

            # 5. Storyboard Agent
            task = await self._create_task(session, comic_id, "storyboard")
            task.input_snapshot = {"story_graph_scenes_count": len(story_graph.get("scenes", []))}
            storyboard_result = await self.storyboard_agent.run(story_graph=story_graph)
            await self._complete_task(session, task, "done", output=storyboard_result)

            storyboard_with_dialogue = storyboard_result.copy()
            storyboard_with_dialogue["story_title"] = story.title or ""

            # 6. Dialogue Agent
            task = await self._create_task(session, comic_id, "dialogue")
            task.input_snapshot = {"total_pages": storyboard_result.get("total_pages", 0)}
            dialogue_result = await self.dialogue_agent.run(
                storyboard=storyboard_result,
                story_title=story.title or ""
            )
            await self._complete_task(session, task, "done", output=dialogue_result)

            # 合并对白到 storyboard
            merged = self._merge_dialogue(storyboard_result, dialogue_result)
            if dialogue_result.get("dialogue_partial"):
                comic.dialogue_partial = 1

            # 7. Director Agent
            task = await self._create_task(session, comic_id, "director")
            task.input_snapshot = {"style_key": style_key, "style_name": style_config["style_name"]}
            director_result = await self.director_agent.run(
                storyboard_with_dialogue=merged,
                style_config=style_config
            )
            await self._complete_task(session, task, "done", output=director_result)

            # 8. Comic Agent
            task = await self._create_task(session, comic_id, "comic")
            task.input_snapshot = {"total_pages": len(director_result.get("pages", []))}
            comic_result = await self.comic_agent.run(
                revised_storyboard=director_result,
                comic_id=comic_id,
                style_key=style_key,
                style_name=style_config["style_name"]
            )
            await self._complete_task(session, task, "done", output={"image_partial": comic_result.get("image_partial")})

            # 持久化 ComicPage
            for p in comic_result.get("pages", []):
                page = ComicPage(
                    comic_id=comic_id,
                    page_num=p.get("page", 0),
                    shot_type=p.get("shot", "Medium"),
                    image_prompt=p.get("image_prompt", ""),
                    image_url=p.get("image_url", ""),
                    dialogue=p.get("dialogue", ""),
                    narration=p.get("narration", ""),
                )
                session.add(page)

            if comic_result.get("image_partial"):
                comic.image_partial = 1

            if comic_result.get("all_failed"):
                comic.status = "failed"
                await session.commit()
                return self._comic_to_dict(comic)

            await session.flush()

            # 9. Layout Agent
            task = await self._create_task(session, comic_id, "layout")
            task.input_snapshot = {"total_pages": len(comic_result.get("pages", []))}
            layout_result = await self.layout_agent.run(
                comic_id=comic_id,
                story_title=story.title or "",
                style_name=style_config["style_name"],
                pages=comic_result.get("pages", []),
                created_at_str=comic.created_at.strftime("%Y-%m-%d") if comic.created_at else ""
            )
            await self._complete_task(session, task, "done", output=layout_result)

            if layout_result.get("success"):
                comic.pdf_path = layout_result["pdf_path"]
                comic.total_pages = len(comic_result.get("pages", []))
                comic.status = "completed"
            else:
                comic.status = "failed"

            await session.commit()
            await session.refresh(comic)

            return self._comic_to_dict(comic)

        except Exception as e:
            logger.error(f"Comic generation failed for story {story_id}: {e}")
            comic.status = "failed"
            await session.commit()
            return self._comic_to_dict(comic)

    async def get_comic(self, comic_id: int, session: AsyncSession) -> Optional[dict]:
        """获取漫画详情。"""
        stmt = select(Comic).where(Comic.id == comic_id)
        result = await session.execute(stmt)
        comic = result.scalar_one_or_none()
        if not comic:
            return None
        return self._comic_to_dict(comic)

    async def get_comic_pages(self, comic_id: int, session: AsyncSession) -> Optional[dict]:
        """获取漫画所有页面。"""
        stmt = select(Comic).where(Comic.id == comic_id)
        result = await session.execute(stmt)
        comic = result.scalar_one_or_none()
        if not comic:
            return None

        stmt = select(ComicPage).where(ComicPage.comic_id == comic_id).order_by(ComicPage.page_num)
        result = await session.execute(stmt)
        pages = result.scalars().all()

        return {
            "story_id": comic.story_id,
            "comic_id": comic_id,
            "pages": [self._page_to_dict(p) for p in pages]
        }

    async def get_comic_tasks(self, comic_id: int, session: AsyncSession) -> Optional[dict]:
        """获取漫画所有生成任务。"""
        stmt = select(Comic).where(Comic.id == comic_id)
        result = await session.execute(stmt)
        comic = result.scalar_one_or_none()
        if not comic:
            return None

        stmt = select(GenerationTask).where(GenerationTask.comic_id == comic_id).order_by(GenerationTask.id)
        result = await session.execute(stmt)
        tasks = result.scalars().all()

        return {
            "comic_id": comic_id,
            "status": comic.status,
            "tasks": [self._task_to_dict(t) for t in tasks]
        }

    async def list_styles(self, session: AsyncSession) -> List[dict]:
        """列出所有活跃风格预设。"""
        stmt = select(PromptVersion).where(PromptVersion.is_active == 1)
        result = await session.execute(stmt)
        pvs = result.scalars().all()
        return [
            {
                "key": pv.style_key,
                "name": pv.style_name,
                "mood": pv.mood,
                "dialogue_level": pv.dialogue_level,
                "cinematic": bool(pv.cinematic),
            }
            for pv in pvs
        ]

    async def _build_story_graph(self, story_id: int, session: AsyncSession) -> dict:
        """构建轻量 story_graph 供 Agent 使用。"""
        stmt = select(Story).where(Story.id == story_id)
        result = await session.execute(stmt)
        story = result.scalar_one_or_none()

        stmt = select(Scene).where(Scene.story_id == story_id).order_by(Scene.seq_num)
        result = await session.execute(stmt)
        scenes = result.scalars().all()

        scenes_data = []
        for s in scenes:
            scenes_data.append({
                "seq": s.seq_num,
                "time": s.time_at or "",
                "location": s.location or "",
                "summary": s.summary or "",
                "narration": s.narration or "",
                "dialogue": s.dialogue or "",
                "characters": [],
                "actions": [],
                "emotions": [],
            })

        return {
            "scenes": scenes_data,
            "characters": {},
            "title": story.title if story else "",
        }

    async def _create_task(self, session: AsyncSession, comic_id: int, agent_name: str) -> GenerationTask:
        """创建 pending 状态的 GenerationTask。"""
        task = GenerationTask(
            comic_id=comic_id,
            agent_name=agent_name,
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(task)
        await session.flush()
        return task

    async def _complete_task(
        self, session: AsyncSession, task: GenerationTask,
        status: str, output: dict = None, error: str = ""
    ):
        """完成 GenerationTask 并更新状态。"""
        task.status = status
        task.ended_at = datetime.utcnow()
        if output:
            task.output_snapshot = output
        if error:
            task.error_message = error
        await session.flush()

    def _merge_dialogue(self, storyboard: dict, dialogue_result: dict) -> dict:
        """将对白合并到 storyboard 的 pages 中。"""
        pages = storyboard.get("pages", [])
        dialogue_pages = dialogue_result.get("pages", [])
        dialogue_map = {dp["page"]: dp.get("dialogue", "") for dp in dialogue_pages}

        for p in pages:
            p["dialogue"] = dialogue_map.get(p["page"], "")

        return storyboard

    def _comic_to_dict(self, comic: Comic) -> dict:
        return {
            "id": comic.id,
            "story_id": comic.story_id,
            "style": comic.style,
            "status": comic.status,
            "total_pages": comic.total_pages,
            "pdf_path": comic.pdf_path or "",
            "image_partial": bool(comic.image_partial),
            "dialogue_partial": bool(comic.dialogue_partial),
            "created_at": comic.created_at,
        }

    def _page_to_dict(self, page: ComicPage) -> dict:
        return {
            "id": page.id,
            "comic_id": page.comic_id,
            "page_num": page.page_num,
            "shot_type": page.shot_type or "Medium",
            "image_prompt": page.image_prompt or "",
            "image_url": page.image_url or "",
            "dialogue": page.dialogue or "",
            "narration": page.narration or "",
            "layout_json": page.layout_json,
        }

    def _task_to_dict(self, task: GenerationTask) -> dict:
        duration_ms = None
        if task.started_at and task.ended_at:
            duration_ms = int((task.ended_at - task.started_at).total_seconds() * 1000)
        return {
            "id": task.id,
            "comic_id": task.comic_id,
            "agent_name": task.agent_name,
            "status": task.status,
            "error_message": task.error_message or "",
            "retry_count": task.retry_count,
            "duration_ms": duration_ms,
        }
