"""Comic Pipeline 集成测试 — 真实 API 调用验证 5 个 Agent + Service 编排。

需要有效的 .env 配置（OPENAI_API_KEY, QIANFAN_ACCESS_KEY 等）。
"""

import os
import pytest
from datetime import datetime

from app.config.config import get_settings
from app.llm.openai_provider import OpenAIProvider
from app.llm.qianfan_image_provider import QianfanImageProvider
from app.agents.storyboard_agent import StoryboardAgent
from app.agents.dialogue_agent import DialogueAgent
from app.agents.director_agent import DirectorAgent
from app.agents.comic_agent import ComicAgent
from app.agents.layout_agent import LayoutAgent
from app.services.comic_service import ComicService


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """清除 Settings 缓存，确保读取最新 .env"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def llm_provider():
    settings = get_settings()
    return OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=0.5,
        top_p=0.5,
    )


@pytest.fixture
def image_provider():
    settings = get_settings()
    return QianfanImageProvider(
        access_key=settings.QIANFAN_ACCESS_KEY,
        secret_key=settings.QIANFAN_SECRET_KEY,
    )


@pytest.fixture
def sample_story_graph():
    """简单的中文故事图谱，用于测试分镜生成。"""
    return {
        "title": "咖啡馆的午后",
        "scenes": [
            {
                "seq": 0,
                "time": "下午三点",
                "location": "街角咖啡馆",
                "summary": "小林走进咖啡馆，阳光透过玻璃窗洒在木桌上",
                "narration": "一个慵懒的午后，咖啡馆里弥漫着咖啡的香气。",
                "dialogue": "",
                "characters": ["小林", "咖啡师"],
                "actions": ["走进", "坐下"],
                "emotions": ["放松", "期待"],
            },
            {
                "seq": 1,
                "time": "下午三点半",
                "location": "咖啡馆窗边",
                "summary": "小林喝着拿铁，翻阅手中的书，窗外行人匆匆",
                "narration": "时间在咖啡的香气中慢慢流淌。",
                "dialogue": "小林：这本书真有意思。",
                "characters": ["小林"],
                "actions": ["喝咖啡", "看书"],
                "emotions": ["平静", "专注"],
            },
        ],
        "characters": {
            "小林": "喜欢独处的年轻人",
            "咖啡师": "热情的咖啡店员工",
        },
    }


# ────────────────────────────────
# Storyboard Agent 集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestStoryboardAgentIntegration:
    """Storyboard Agent — 真实 LLM 调用"""

    @pytest.mark.asyncio
    async def test_storyboard_generation(self, llm_provider, sample_story_graph):
        """验证分镜生成：返回 pages 含 shot/desc/source_scene_seq"""
        agent = StoryboardAgent(llm_provider)
        result = await agent.run(story_graph=sample_story_graph)

        assert "total_pages" in result
        assert "pages" in result
        assert result["total_pages"] > 0
        pages = result["pages"]

        valid_shots = {"Wide", "Medium", "Close-up", "Action", "Ending"}
        for p in pages:
            assert "page" in p
            assert "shot" in p
            assert "desc" in p
            assert "source_scene_seq" in p
            assert p["shot"] in valid_shots, f"无效镜头类型: {p['shot']}"
            assert len(p["desc"]) > 0, f"Page {p['page']} 缺少描述"

        print(f"\n[Storyboard] 生成 {result['total_pages']} 个分镜页")
        for p in pages:
            print(f"  Page {p['page']}: [{p['shot']}] {p['desc'][:60]}...")


# ────────────────────────────────
# Dialogue Agent 集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestDialogueAgentIntegration:
    """Dialogue Agent — 真实 LLM 调用"""

    @pytest.mark.asyncio
    async def test_dialogue_generation(self, llm_provider, sample_story_graph):
        """验证对白生成：返回 pages 含 dialogue 字段"""
        # 先用 Storyboard 生成分镜
        storyboard_agent = StoryboardAgent(llm_provider)
        storyboard = await storyboard_agent.run(story_graph=sample_story_graph)

        agent = DialogueAgent(llm_provider)
        result = await agent.run(
            storyboard=storyboard,
            story_title=sample_story_graph["title"]
        )

        assert "pages" in result
        pages = result["pages"]
        assert len(pages) == storyboard["total_pages"]

        dialogue_count = sum(1 for p in pages if p.get("dialogue", ""))
        print(f"\n[Dialogue] 为 {len(pages)} 个分镜页生成对白，{dialogue_count} 页有对白")
        for p in pages:
            d = p.get("dialogue", "")
            if d:
                print(f"  Page {p['page']}: {d[:80]}")


# ────────────────────────────────
# Director Agent 集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestDirectorAgentIntegration:
    """Director Agent — 真实 LLM 调用 + 风格模板"""

    @pytest.mark.asyncio
    async def test_director_miyazaki_style(self, llm_provider, sample_story_graph):
        """验证宫崎骏风格调整：director_applied=True，风格描述已修改"""
        # 先生成分镜 + 对白
        storyboard_agent = StoryboardAgent(llm_provider)
        storyboard = await storyboard_agent.run(story_graph=sample_story_graph)

        dialogue_agent = DialogueAgent(llm_provider)
        dialogue = await dialogue_agent.run(storyboard=storyboard, story_title=sample_story_graph["title"])

        # 合并对白
        dialogue_map = {dp["page"]: dp.get("dialogue", "") for dp in dialogue["pages"]}
        for p in storyboard["pages"]:
            p["dialogue"] = dialogue_map.get(p["page"], "")

        # 宫崎骏风格配置
        style_config = {
            "style_key": "miyazaki",
            "style_name": "宫崎骏风",
            "prompt_template": (
                "你是一位漫画导演，请将以下分镜脚本调整为「宫崎骏风格」。\n\n"
                "核心要求：\n"
                "- 画面色调：柔和、温暖、自然光\n"
                "- 氛围：温馨、治愈、带一点怀旧\n"
                "- 对白：尽量少，用画面和表情传达情绪\n"
                "- 细节：注意天空、风、花草等自然元素的描绘\n"
                "- 镜头：偏好中远景和全景，少用特写\n\n"
                "原始分镜：\n{storyboard_json}\n\n"
                "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
            ),
            "mood": "温馨",
            "dialogue_level": "少",
            "cinematic": 1,
        }

        agent = DirectorAgent(llm_provider)
        result = await agent.run(
            storyboard_with_dialogue=storyboard,
            style_config=style_config
        )

        assert "pages" in result
        assert result["director_applied"] is True
        assert len(result["pages"]) > 0

        print(f"\n[Director] 宫崎骏风格调整完成, director_applied={result['director_applied']}")
        if result.get("style_notes"):
            print(f"  风格备注: {result['style_notes'][:100]}...")


# ────────────────────────────────
# Comic Agent 集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestComicAgentIntegration:
    """Comic Agent — 真实千帆文生图 + 本地下载"""

    @pytest.mark.asyncio
    async def test_generate_and_download_image(self, llm_provider, image_provider, tmp_path):
        """验证生图+下载：单页测试"""
        storyboard = {
            "total_pages": 1,
            "pages": [
                {"page": 1, "shot": "Wide", "desc": "宁静的街角咖啡馆，阳光洒落", "dialogue": "", "source_scene_seq": 0}
            ]
        }

        agent = ComicAgent(image_provider, str(tmp_path))
        result = await agent.run(
            revised_storyboard=storyboard,
            comic_id=999,
            style_key="slice",
            style_name="日常风"
        )

        assert "pages" in result
        assert result["image_partial"] is False
        assert result["all_failed"] is False

        page = result["pages"][0]
        assert page["image_url"] != "", "图片生成失败"
        assert os.path.exists(page["image_url"]), f"图片文件不存在: {page['image_url']}"
        assert os.path.getsize(page["image_url"]) > 100, "图片文件太小"

        print(f"\n[Comic] 生图成功: {page['image_url']}")
        print(f"  文件大小: {os.path.getsize(page['image_url'])} bytes")


# ────────────────────────────────
# Layout Agent 集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestLayoutAgentIntegration:
    """Layout Agent — 真实 PDF 生成"""

    @pytest.mark.asyncio
    async def test_generate_pdf(self, llm_provider, image_provider, tmp_path):
        """完整流程：分镜→对白→导演→生图→排版→PDF"""
        # Step 1-3: 分镜 + 对白 + 导演
        storyboard_agent = StoryboardAgent(llm_provider)
        dialogue_agent = DialogueAgent(llm_provider)
        director_agent = DirectorAgent(llm_provider)
        comic_agent = ComicAgent(image_provider, str(tmp_path))

        story_graph = {
            "title": "夕阳下的约定",
            "scenes": [
                {
                    "seq": 0, "time": "傍晚", "location": "海边堤坝",
                    "summary": "两个人并肩坐在堤坝上看夕阳",
                    "narration": "橘红色的天空下，海浪轻轻拍打着堤坝。",
                    "dialogue": "A：明天你还会来吗？\nB：当然，每天都会。",
                    "characters": ["A", "B"], "actions": ["坐", "看"], "emotions": ["温暖", "期待"]
                }
            ],
            "characters": {"A": "短发女孩", "B": "戴眼镜的男孩"},
        }

        storyboard = await storyboard_agent.run(story_graph=story_graph)
        assert storyboard["total_pages"] > 0
        print(f"\n[Layout E2E] Storyboard: {storyboard['total_pages']} pages")

        dialogue = await dialogue_agent.run(storyboard=storyboard, story_title=story_graph["title"])
        dialogue_map = {dp["page"]: dp.get("dialogue", "") for dp in dialogue["pages"]}
        for p in storyboard["pages"]:
            p["dialogue"] = dialogue_map.get(p["page"], "")

        style_config = {
            "style_key": "slice",
            "style_name": "日常风",
            "prompt_template": (
                "你是一位漫画导演，请将以下分镜脚本调整为「日常温馨风格」。\n"
                "要求：自然光线、柔和色调、生活感强、对白轻松自然。\n\n"
                "原始分镜：\n{storyboard_json}\n\n"
                "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
            ),
            "mood": "轻松",
            "dialogue_level": "多",
            "cinematic": 0,
        }
        director_result = await director_agent.run(
            storyboard_with_dialogue=storyboard,
            style_config=style_config
        )
        print(f"[Layout E2E] Director: applied={director_result['director_applied']}")

        comic_result = await comic_agent.run(
            revised_storyboard=director_result,
            comic_id=998,
            style_key="slice",
            style_name="日常风"
        )
        assert not comic_result["all_failed"], "图片生成全部失败"
        print(f"[Layout E2E] Comic: image_partial={comic_result['image_partial']}")

        layout_agent = LayoutAgent(str(tmp_path))
        layout_result = await layout_agent.run(
            comic_id=998,
            story_title=story_graph["title"],
            style_name="日常风",
            pages=comic_result["pages"],
            created_at_str=datetime.utcnow().strftime("%Y-%m-%d")
        )

        assert layout_result["success"] is True
        assert layout_result["pdf_path"] != ""
        assert os.path.exists(layout_result["pdf_path"])
        assert os.path.getsize(layout_result["pdf_path"]) > 1000

        print(f"[Layout E2E] PDF 生成成功: {layout_result['pdf_path']}")
        print(f"  文件大小: {os.path.getsize(layout_result['pdf_path'])} bytes")


# ────────────────────────────────
# ComicService 端到端集成测试
# ────────────────────────────────


@pytest.mark.integration
class TestComicServiceE2E:
    """ComicService 端到端 — 完整 5 Agent 流水线（需要数据库）"""

    @pytest.mark.asyncio
    async def test_full_comic_pipeline(self, llm_provider, image_provider, tmp_path):
        """完整漫画流水线：Story → 5 Agents → Comic + Pages + PDF

        使用内存 SQLite 数据库，调用真实 API。
        """
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
        from app.db.models import Base, Story
        from app.db.seed_prompts import seed_prompt_versions

        # ── 创建内存数据库 ──
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with session_factory() as session:
            # ── Seed 风格预设 ──
            await seed_prompt_versions(session)
            await session.commit()
            print("\n[ComicService E2E] 风格预设已种子化")

            # ── 创建 Story ──
            story = Story(
                album_id=1,
                title="雨天的便利店",
                summary="雨天偶遇的温馨小故事",
                status="completed",
                scene_count=1,
            )
            session.add(story)
            await session.commit()
            await session.refresh(story)

            # ── 创建 Scene（供 _build_story_graph 使用）──
            from app.db.models import Scene
            scene = Scene(
                story_id=story.id,
                seq_num=0,
                time_at="下午五点",
                location="街角便利店",
                summary="下雨天，两个陌生人在便利店屋檐下躲雨",
                narration="突然下起的雨，让陌生的两个人有了交集。",
                dialogue="A：这雨不知道什么时候停。\nB：是啊，不过偶尔淋淋雨也不错。",
            )
            session.add(scene)
            await session.commit()

            print(f"[ComicService E2E] Story #{story.id} 已创建")

            # ── 构建 ComicService ──
            svc = ComicService(
                storyboard_agent=StoryboardAgent(llm_provider),
                dialogue_agent=DialogueAgent(llm_provider),
                director_agent=DirectorAgent(llm_provider),
                comic_agent=ComicAgent(image_provider, str(tmp_path)),
                layout_agent=LayoutAgent(str(tmp_path)),
            )

            # ── 运行流水线 ──
            print("[ComicService E2E] 开始漫画生成流水线...")
            result = await svc.generate_comic(
                story_id=story.id,
                style_key="miyazaki",
                session=session,
            )

            assert result is not None, "生成结果为空"
            print(f"[ComicService E2E] 完成! status={result['status']}")

            # ── 验证 Comic 记录 ──
            assert result["story_id"] == story.id
            assert result["style"] == "miyazaki"

            # ── 验证 ComicPage ──
            pages_result = await svc.get_comic_pages(result["id"], session)
            assert pages_result is not None
            assert len(pages_result["pages"]) > 0
            print(f"[ComicService E2E] ComicPage 数量: {len(pages_result['pages'])}")
            for p in pages_result["pages"]:
                print(f"  Page {p['page_num']}: [{p['shot_type']}] dialogue={bool(p['dialogue'])}, image={bool(p['image_url'])}")

            # ── 验证 GenerationTask ──
            tasks_result = await svc.get_comic_tasks(result["id"], session)
            assert tasks_result is not None
            task_names = [t["agent_name"] for t in tasks_result["tasks"]]
            print(f"[ComicService E2E] GenerationTasks: {task_names}")
            assert len(tasks_result["tasks"]) == 5
            for t in tasks_result["tasks"]:
                print(f"  {t['agent_name']}: {t['status']} (耗时 {t.get('duration_ms', 'N/A')}ms)")

            # ── 验证 PDF ──
            if result["status"] == "completed" and result["pdf_path"]:
                assert os.path.exists(result["pdf_path"])
                pdf_size = os.path.getsize(result["pdf_path"])
                assert pdf_size > 1000
                print(f"[ComicService E2E] PDF 生成成功: {result['pdf_path']} ({pdf_size} bytes)")

            # ── 验证风格列表 ──
            styles = await svc.list_styles(session)
            assert len(styles) == 4
            print(f"[ComicService E2E] 风格列表: {[s['key'] for s in styles]}")

        await engine.dispose()
        print("\n[ComicService E2E] 端到端测试 PASSED ✅")
