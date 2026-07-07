import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.album_agent import AlbumAgent
from app.agents.vision_agent import VisionAgent
from app.agents.story_agent import StoryAgent
from app.db.models import (
    Album, Photo, PhotoAnalysis, Story, Scene,
    Character, SceneCharacter, SceneAction, SceneEmotion, ScenePhoto,
)

logger = logging.getLogger(__name__)


class StoryService:
    """Story 生成编排：串行调用 3 个 Agent。"""

    def __init__(
        self,
        album_agent: AlbumAgent,
        vision_agent: VisionAgent,
        story_agent: StoryAgent,
    ):
        self.album_agent = album_agent
        self.vision_agent = vision_agent
        self.story_agent = story_agent

    async def generate_story(self, album_id: int, session: AsyncSession) -> Optional[dict]:
        """完整的故事生成流水线。

        Args:
            album_id: 相册 ID
            session: 数据库会话

        Returns:
            dict or None: StoryResponse 序列化字典
        """
        # 1. 获取 Album 及其 Photos
        query = select(Album).where(Album.id == album_id).options(selectinload(Album.photos))
        result = await session.execute(query)
        album = result.scalar_one_or_none()

        if not album:
            return None

        photos = album.photos
        if not photos:
            return None

        # 2. 创建 Story 记录（status=processing）
        story = Story(album_id=album_id, title="", summary="", status="processing", scene_count=0)
        session.add(story)
        await session.commit()
        await session.refresh(story)

        try:
            # 3. Album Agent：构建时间线
            photos_data = [{"id": p.id, "path": p.path, "taken_at": p.taken_at or ""} for p in photos]
            timeline = await self.album_agent.run(photos=photos_data)

            # 4. Vision Agent：逐张分析（跳过失败的单张）
            analyses = []
            for p in photos:
                try:
                    analysis = await self.vision_agent.run(photo_id=p.id, image_path=p.path)
                    analyses.append(analysis)

                    # 持久化 PhotoAnalysis
                    pa = PhotoAnalysis(
                        photo_id=p.id,
                        objects=analysis.get("objects", []),
                        scene_desc=analysis.get("scene_desc", ""),
                        weather=analysis.get("weather", ""),
                        location=analysis.get("location", ""),
                        action=analysis.get("action", ""),
                        emotion=analysis.get("emotion", ""),
                        clothing=analysis.get("clothing", []),
                        people=analysis.get("people", []),
                    )
                    session.add(pa)
                except Exception as e:
                    logger.error(f"Vision analysis failed for photo {p.id}: {e}")

            if not analyses:
                story.status = "failed"
                await session.commit()
                return {
                    "id": story.id,
                    "album_id": album_id,
                    "title": "",
                    "summary": "",
                    "status": "failed",
                    "scene_count": 0,
                    "created_at": story.created_at,
                }

            await session.flush()

            # 5. Story Agent：生成故事图谱
            story_graph = await self.story_agent.run(timeline=timeline, analyses=analyses)

            # 6. 持久化 Scenes, Characters 等
            await self._persist_story_graph(session, story.id, story_graph)

            story.title = story_graph.get("story_title", "")
            story.summary = story_graph.get("story_summary", "")
            story.scene_count = len(story_graph.get("scenes", []))
            story.status = "completed"
            await session.commit()
            await session.refresh(story)

            return {
                "id": story.id,
                "album_id": story.album_id,
                "title": story.title,
                "summary": story.summary,
                "status": story.status,
                "scene_count": story.scene_count,
                "created_at": story.created_at,
            }

        except Exception as e:
            logger.error(f"Story generation failed for album {album_id}: {e}")
            story.status = "failed"
            await session.commit()
            return {
                "id": story.id,
                "album_id": album_id,
                "title": "",
                "summary": "",
                "status": "failed",
                "scene_count": 0,
                "created_at": story.created_at,
            }

    async def _persist_story_graph(self, session: AsyncSession, story_id: int, story_graph: dict):
        """持久化 Story Graph 中的 Scenes、Characters 等数据。"""
        # 角色映射：name -> Character id
        character_map: Dict[str, int] = {}

        for scene_data in story_graph.get("scenes", []):
            scene = Scene(
                story_id=story_id,
                seq_num=scene_data.get("seq", 0),
                time_at=scene_data.get("time", ""),
                location=scene_data.get("location", ""),
                summary=scene_data.get("summary", ""),
            )
            session.add(scene)
            await session.flush()

            # 角色
            for char_data in scene_data.get("characters", []):
                name = char_data.get("name", "")
                if name not in character_map:
                    char = Character(name=name, traits=[])
                    session.add(char)
                    await session.flush()
                    character_map[name] = char.id

                sc = SceneCharacter(
                    scene_id=scene.id,
                    character_id=character_map[name],
                    role=char_data.get("role", ""),
                )
                session.add(sc)

            # 动作
            for action_data in scene_data.get("actions", []):
                sa = SceneAction(
                    scene_id=scene.id,
                    verb=action_data.get("verb", ""),
                    object=action_data.get("object", ""),
                )
                session.add(sa)

            # 情绪
            for emotion_data in scene_data.get("emotions", []):
                se = SceneEmotion(
                    scene_id=scene.id,
                    type=emotion_data.get("type", ""),
                    intensity=emotion_data.get("intensity", 0.0),
                )
                session.add(se)

            # 来源照片
            for photo_id in scene_data.get("source_photos", []):
                sp = ScenePhoto(scene_id=scene.id, photo_id=photo_id)
                session.add(sp)

    async def get_story(self, story_id: int, session: AsyncSession) -> Optional[dict]:
        """获取故事详情。"""
        query = select(Story).where(Story.id == story_id)
        result = await session.execute(query)
        story = result.scalar_one_or_none()

        if not story:
            return None

        return {
            "id": story.id,
            "album_id": story.album_id,
            "title": story.title,
            "summary": story.summary,
            "status": story.status,
            "scene_count": story.scene_count,
            "created_at": story.created_at,
        }

    async def get_story_graph(self, story_id: int, session: AsyncSession) -> Optional[dict]:
        """获取完整故事图谱。"""
        query = select(Story).where(Story.id == story_id).options(
            selectinload(Story.scenes).selectinload(Scene.characters).selectinload(SceneCharacter.character),
            selectinload(Story.scenes).selectinload(Scene.actions),
            selectinload(Story.scenes).selectinload(Scene.emotions),
            selectinload(Story.scenes).selectinload(Scene.photos),
        )
        result = await session.execute(query)
        story = result.scalar_one_or_none()

        if not story:
            return None

        scenes = []
        all_characters: Dict[int, dict] = {}

        for scene in story.scenes:
            characters = []
            source_photos = []
            actions = []
            emotions = []

            for sc in scene.characters:
                c = sc.character
                if c.id not in all_characters:
                    all_characters[c.id] = {
                        "id": c.id,
                        "name": c.name,
                        "traits": c.traits or [],
                        "role": sc.role or "",
                        "avatar_url": c.avatar_url or "",
                    }
                characters.append({
                    "id": c.id,
                    "name": c.name,
                    "traits": c.traits or [],
                    "role": sc.role or "",
                    "avatar_url": c.avatar_url or "",
                })

            for a in scene.actions:
                actions.append({"verb": a.verb, "object": a.object or ""})

            for e in scene.emotions:
                emotions.append({"type": e.type, "intensity": e.intensity or 0.0})

            for sp in scene.photos:
                source_photos.append(sp.photo_id)

            scenes.append({
                "id": scene.id,
                "story_id": scene.story_id,
                "seq_num": scene.seq_num,
                "time_at": scene.time_at,
                "location": scene.location,
                "summary": scene.summary,
                "characters": characters,
                "actions": actions,
                "emotions": emotions,
                "source_photos": source_photos,
            })

        return {
            "story_id": story.id,
            "story_title": story.title,
            "story_summary": story.summary,
            "scenes": scenes,
            "global_characters": list(all_characters.values()),
        }

    async def get_scene_detail(self, scene_id: int, session: AsyncSession) -> Optional[dict]:
        """获取单个场景详情（含来源照片完整信息）。"""
        query = select(Scene).where(Scene.id == scene_id).options(
            selectinload(Scene.characters).selectinload(SceneCharacter.character),
            selectinload(Scene.actions),
            selectinload(Scene.emotions),
            selectinload(Scene.photos).selectinload(ScenePhoto.photo),
        )
        result = await session.execute(query)
        scene = result.scalar_one_or_none()

        if not scene:
            return None

        characters = []
        for sc in scene.characters:
            c = sc.character
            characters.append({
                "id": c.id,
                "name": c.name,
                "traits": c.traits or [],
                "role": sc.role or "",
                "avatar_url": c.avatar_url or "",
            })

        actions = [{"verb": a.verb, "object": a.object or ""} for a in scene.actions]
        emotions = [{"type": e.type, "intensity": e.intensity or 0.0} for e in scene.emotions]

        source_photos = []
        for sp in scene.photos:
            p = sp.photo
            source_photos.append({
                "id": p.id,
                "album_id": p.album_id,
                "path": p.path,
                "taken_at": p.taken_at,
            })

        return {
            "id": scene.id,
            "story_id": scene.story_id,
            "seq_num": scene.seq_num,
            "time_at": scene.time_at,
            "location": scene.location,
            "summary": scene.summary,
            "characters": characters,
            "actions": actions,
            "emotions": emotions,
            "source_photos": source_photos,
        }
