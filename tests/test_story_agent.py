from unittest.mock import MagicMock
import pytest


class TestStoryAgent:
    @pytest.mark.asyncio
    async def test_run_generates_story_graph(self):
        mock_provider = MagicMock()
        mock_provider.chat_json.return_value = {
            "story_title": "东京一日",
            "story_summary": "第一次来东京，迷路却遇见了美好",
            "scenes": [
                {
                    "seq": 1, "time": "08:30", "location": "酒店餐厅",
                    "summary": "在酒店吃早餐，期待今天的旅程",
                    "characters": [{"name": "我", "role": "主角"}],
                    "actions": [{"verb": "吃", "object": "早餐"}],
                    "emotions": [{"type": "期待", "intensity": 0.7}],
                    "source_photos": [1, 2],
                },
                {
                    "seq": 2, "time": "10:15", "location": "东京塔",
                    "summary": "登上东京塔，俯瞰整个城市",
                    "characters": [{"name": "我", "role": "主角"}],
                    "actions": [{"verb": "参观", "object": "东京塔"}],
                    "emotions": [{"type": "震撼", "intensity": 0.9}],
                    "source_photos": [3],
                },
            ],
            "global_characters": [
                {"name": "我", "traits": ["长发", "眼镜", "白外套"], "first_appearance_scene": 1},
            ],
        }

        from app.agents.story_agent import StoryAgent

        agent = StoryAgent(llm_provider=mock_provider)
        timeline = {
            "date": "2026-07-07",
            "entries": [
                {"seq": 1, "time": "08:30", "photos": [1, 2], "label": "早餐"},
                {"seq": 2, "time": "10:15", "photos": [3], "label": "东京塔"},
            ],
            "photo_count": 3,
        }
        analyses = [
            {"photo_id": 1, "scene_desc": "酒店早餐", "location": "酒店", "emotion": "期待"},
            {"photo_id": 2, "scene_desc": "酒店早餐", "location": "酒店", "emotion": "开心"},
            {"photo_id": 3, "scene_desc": "东京塔观光", "location": "东京塔", "emotion": "震撼"},
        ]

        result = await agent.run(timeline=timeline, analyses=analyses)

        assert result["story_title"] == "东京一日"
        assert len(result["scenes"]) == 2
        assert result["scenes"][0]["source_photos"] == [1, 2]
        assert len(result["global_characters"]) == 1
