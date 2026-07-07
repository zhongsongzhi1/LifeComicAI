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
