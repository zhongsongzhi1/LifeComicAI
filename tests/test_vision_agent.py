from unittest.mock import patch, MagicMock
import pytest


class TestVisionAgent:
    @pytest.mark.asyncio
    async def test_run_returns_analysis_dict(self):
        mock_provider = MagicMock()
        mock_provider.analyze_image.return_value = {
            "objects": ["人物A", "拉面", "桌子"],
            "scene_desc": "东京一家拉面店内，暖色灯光",
            "weather": "室内",
            "location": "拉面店",
            "action": "吃面",
            "emotion": "开心",
            "clothing": ["白T恤", "牛仔裤"],
            "people": [{"role": "主角", "traits": "长发/眼镜/白外套"}],
        }

        from app.agents.vision_agent import VisionAgent

        agent = VisionAgent(qianfan_provider=mock_provider)
        result = await agent.run(photo_id=1, image_path="/fake/path.jpg")

        assert result["photo_id"] == 1
        assert result["objects"] == ["人物A", "拉面", "桌子"]
        assert result["scene_desc"] == "东京一家拉面店内，暖色灯光"
        assert result["emotion"] == "开心"
        assert len(result["people"]) == 1

    @pytest.mark.asyncio
    async def test_run_handles_provider_error(self):
        mock_provider = MagicMock()
        mock_provider.analyze_image.side_effect = Exception("API timeout")

        from app.agents.vision_agent import VisionAgent

        agent = VisionAgent(qianfan_provider=mock_provider)
        result = await agent.run(photo_id=1, image_path="/fake/path.jpg")

        assert result["photo_id"] == 1
        assert result["objects"] == []
        assert result["emotion"] == ""
        assert result["people"] == []
