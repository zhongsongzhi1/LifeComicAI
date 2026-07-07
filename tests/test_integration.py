import os
import pytest

from app.config.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """清除 Settings 缓存，确保读取最新 .env"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.integration
class TestOpenAIProviderIntegration:
    """OpenAIProvider 集成测试 — 真实调用阿里云 DashScope"""

    @pytest.fixture
    def provider(self):
        from app.llm.openai_provider import OpenAIProvider
        settings = get_settings()
        return OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
        )

    def test_chat_simple(self, provider):
        """验证基本对话"""
        result = provider.chat([
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ])
        assert result
        assert len(result) > 0
        print(f"\n[DashScope chat] {result[:200]}")

    def test_chat_json(self, provider):
        """验证 JSON 输出"""
        result = provider.chat_json([
            {"role": "system", "content": "只输出 JSON，不要其他文字。"},
            {"role": "user", "content": '返回 {"name": "小明", "age": 25}'}
        ])
        assert isinstance(result, dict)
        assert "name" in result
        print(f"\n[DashScope chat_json] {result}")


@pytest.mark.integration
class TestQianfanProviderIntegration:
    """千帆 Provider 集成测试 — 真实调用百度千帆视觉模型"""

    @pytest.fixture
    def provider(self):
        from app.llm.qianfan_provider import QianfanProvider
        settings = get_settings()
        return QianfanProvider(
            access_key=settings.QIANFAN_ACCESS_KEY,
            secret_key=settings.QIANFAN_SECRET_KEY,
        )

    def test_analyze_image(self, provider):
        """验证图片分析"""
        import tempfile
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_path = f.name
        img.save(test_path)

        try:
            result = provider.analyze_image(test_path)
            assert isinstance(result, dict)
            assert "scene_desc" in result
            print(f"\n[Qianfan analyze] {result}")
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)


@pytest.mark.integration
class TestStoryAgentIntegration:
    """Story Agent 集成测试 — 端到端调用 Agent"""

    def test_story_agent_with_real_llm(self):
        """验证 Story Agent 生成故事图谱"""
        from app.llm.openai_provider import OpenAIProvider
        from app.agents.story_agent import StoryAgent

        settings = get_settings()
        llm = OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
        )

        agent = StoryAgent(llm_provider=llm)

        timeline = {
            "date": "2026-07-07",
            "entries": [
                {"seq": 1, "time": "08:30", "photos": [1], "label": "早餐"},
                {"seq": 2, "time": "10:00", "photos": [2], "label": "公园散步"},
            ],
            "photo_count": 2,
        }
        analyses = [
            {"photo_id": 1, "scene_desc": "餐厅里吃早餐，阳光透过窗户照进来", "location": "酒店餐厅", "emotion": "愉悦"},
            {"photo_id": 2, "scene_desc": "公园里散步，绿树成荫，有小湖", "location": "城市公园", "emotion": "放松"},
        ]

        import asyncio
        result = asyncio.run(agent.run(timeline=timeline, analyses=analyses))

        assert "story_title" in result
        assert "scenes" in result
        assert len(result["scenes"]) > 0
        print(f"\n[Story Agent] title: {result.get('story_title')}")
        print(f"[Story Agent] scenes: {len(result.get('scenes', []))}")


@pytest.mark.integration
class TestQianfanImageGeneration:
    """千帆文生图集成测试"""

    def test_generate_image(self):
        """验证千帆文生图 API"""
        from app.llm.qianfan_image_provider import QianfanImageProvider

        settings = get_settings()
        provider = QianfanImageProvider(
            access_key=settings.QIANFAN_ACCESS_KEY,
            secret_key=settings.QIANFAN_SECRET_KEY,
        )

        prompt = "一只可爱的橘猫坐在窗台上，阳光洒在它身上，温暖治愈的风格"
        result = provider.generate(prompt, size="1024x1024")

        assert result is not None, "文生图 API 调用失败"
        assert len(result.get("urls", [])) > 0, "未返回图片 URL"

        print(f"\n[千帆文生图] prompt: {prompt}")
        print(f"[千帆文生图] 生成图片 URL: {result['urls'][0][:80]}...")


@pytest.mark.integration
class TestComicGenerationE2E:
    """端到端漫画生成测试 — 真实照片 → 视觉分析 → 时间线 → 故事图谱"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_photos(self):
        """完整流水线：真实照片 → 创建相册 → 运行全部 Agent → 输出故事"""
        import os as _os

        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

        from app.db.models import Base, Album, Photo
        from app.config.config import get_settings
        from app.llm.openai_provider import OpenAIProvider
        from app.llm.qianfan_provider import QianfanProvider
        from app.llm.qianfan_image_provider import QianfanImageProvider
        from app.agents.album_agent import AlbumAgent
        from app.agents.vision_agent import VisionAgent
        from app.agents.story_agent import StoryAgent
        from app.services.story_service import StoryService

        settings = get_settings()

        # ── Step 1: 加载真实照片 ──
        image_dir = _os.path.join(_os.path.dirname(__file__), "..", "image")
        image_files = sorted([
            f for f in _os.listdir(image_dir) if f.endswith(".png")
        ])

        print("\n" + "=" * 60)
        print("E2E 漫画生成测试（真实照片 + 文生图）")
        print("=" * 60)
        print(f"测试图片: {len(image_files)} 张")
        for i, f in enumerate(image_files):
            print(f"  [{i+1}] {f}")

        assert len(image_files) >= 2, "至少需要 2 张测试图片"

        # ── Step 2: 创建 SQLite 内存数据库 ──
        engine = create_async_engine("sqlite+aiosqlite://", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            # 创建 Album
            album = Album(title="真实照片测试", date="2026-07-08")
            session.add(album)
            await session.flush()

            # 创建 Photos（按顺序分配时间）
            photo_ids = []
            times = ["08:30:00", "10:00:00", "14:00:00", "18:00:00"]
            for i, fname in enumerate(image_files):
                photo_path = _os.path.join(image_dir, fname)
                photo = Photo(
                    album_id=album.id,
                    path=photo_path,
                    taken_at=f"2026-07-08 {times[i] if i < len(times) else '12:00:00'}",
                )
                session.add(photo)
                await session.flush()
                photo_ids.append(photo.id)

            await session.commit()
            print(f"\nAlbum #{album.id} 创建成功，{len(photo_ids)} 张照片")

            # ── Step 3: 构建 Agent ──
            openai_llm = OpenAIProvider(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model_name=settings.OPENAI_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
            )
            qianfan = QianfanProvider(
                access_key=settings.QIANFAN_ACCESS_KEY,
                secret_key=settings.QIANFAN_SECRET_KEY,
            )
            qianfan.configure(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
            image_provider = QianfanImageProvider(
                access_key=settings.QIANFAN_ACCESS_KEY,
                secret_key=settings.QIANFAN_SECRET_KEY,
            )

            svc = StoryService(
                album_agent=AlbumAgent(llm_provider=openai_llm),
                vision_agent=VisionAgent(qianfan_provider=qianfan),
                story_agent=StoryAgent(llm_provider=openai_llm),
                image_provider=image_provider,
            )

            # ── Step 4: 运行完整流水线 ──
            print("\n正在分析照片并生成故事图谱...")
            result = await svc.generate_story(album_id=album.id, session=session)

            assert result is not None, "生成失败"
            assert result["status"] == "completed", f"状态异常: {result['status']}"

            print(f"\n{'─' * 40}")
            print(f"故事标题: {result['title']}")
            print(f"故事摘要: {result['summary']}")
            print(f"场景数量: {result['scene_count']}")

            # ── Step 5: 获取完整故事图谱 ──
            story_graph = await svc.get_story_graph(result["id"], session)
            assert story_graph is not None

            print(f"\n完整故事图谱:")
            print(f"  标题: {story_graph['story_title']}")
            print(f"  摘要: {story_graph['story_summary']}")
            print(f"  全局角色: {len(story_graph.get('global_characters', []))} 个")
            for c in story_graph.get("global_characters", []):
                print(f"    - {c['name']}: {c.get('traits', [])}")

            for scene in story_graph.get("scenes", []):
                print(f"\n  Scene #{scene['seq_num']} [{scene['time_at']}] @ {scene['location']}")
                print(f"    {scene['summary']}")
                narration = scene.get("narration", "")
                dialogue = scene.get("dialogue", "")
                if narration:
                    print(f"    旁白: {narration}")
                if dialogue:
                    print(f"    对话: {dialogue}")
                print(f"    角色: {[c['name'] for c in scene.get('characters', [])]}")
                print(f"    动作: {[(a['verb'], a['object']) for a in scene.get('actions', [])]}")
                print(f"    情绪: {[(e['type'], e['intensity']) for e in scene.get('emotions', [])]}")
                comic_url = scene.get("comic_image_url", "")
                if comic_url:
                    print(f"    漫画: {comic_url}")

            print(f"\n{'=' * 60}")
            print("E2E 漫画生成测试（真实照片）PASSED")
            print("=" * 60)

        await engine.dispose()
