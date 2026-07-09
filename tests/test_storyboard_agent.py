import pytest
from unittest.mock import MagicMock
from app.agents.storyboard_agent import StoryboardAgent


# Sample story graph fixture
@pytest.fixture
def sample_story_graph():
    return {
        "scenes": [
            {
                "seq": 0,
                "time": "早晨",
                "location": "公园",
                "summary": "小明在公园散步",
                "narration": "新的一天开始了",
                "characters": ["小明"],
                "actions": ["散步"],
                "emotions": ["平静"]
            },
            {
                "seq": 1,
                "time": "中午",
                "location": "餐厅",
                "summary": "小明和朋友吃饭",
                "narration": "午餐时间",
                "characters": ["小明", "朋友"],
                "actions": ["吃饭", "聊天"],
                "emotions": ["开心", "兴奋"]
            }
        ],
        "characters": {"小明": "男孩", "朋友": "女孩"},
        "title": "平凡的一天"
    }


@pytest.fixture
def sample_storyboard_output():
    return {
        "total_pages": 3,
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "公园全景，清晨阳光洒落", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "小明悠闲散步", "source_scene_seq": 0},
            {"page": 3, "shot": "Close-up", "desc": "小明和朋友开心聊天", "source_scene_seq": 1}
        ]
    }


@pytest.fixture
def mock_llm(sample_storyboard_output):
    """Mock LLM provider with chat_json returning sample storyboard."""
    llm = MagicMock()
    llm.chat_json.return_value = sample_storyboard_output
    return llm


@pytest.mark.asyncio
async def test_storyboard_agent_run(mock_llm, sample_story_graph):
    agent = StoryboardAgent(mock_llm)
    result = await agent.run(story_graph=sample_story_graph)
    assert result["total_pages"] == 3
    assert len(result["pages"]) == 3
    assert result["pages"][0]["shot"] in ["Wide", "Medium", "Close-up", "Action", "Ending"]
    assert "desc" in result["pages"][0]
    assert "source_scene_seq" in result["pages"][0]


@pytest.mark.asyncio
async def test_storyboard_agent_empty_scenes(mock_llm):
    agent = StoryboardAgent(mock_llm)
    result = await agent.run(story_graph={"scenes": []})
    assert result["total_pages"] == 0
    assert result["pages"] == []


@pytest.mark.asyncio
async def test_storyboard_agent_fallback(mock_llm):
    """When LLM returns empty result, should fallback to default storyboard."""
    mock_llm.chat_json.return_value = {}
    sample = {
        "scenes": [{"seq": 0, "summary": "test scene", "narration": "", "time": "", "location": "", "characters": [], "actions": [], "emotions": []}],
        "characters": {},
        "title": "test"
    }
    agent = StoryboardAgent(mock_llm)
    result = await agent.run(story_graph=sample)
    assert result["total_pages"] >= 1
    assert result["pages"][0]["shot"] == "Medium"


@pytest.mark.asyncio
async def test_storyboard_agent_llm_error(mock_llm):
    """When LLM raises exception, should fallback to default storyboard."""
    mock_llm.chat_json.side_effect = Exception("LLM error")
    sample = {
        "scenes": [{"seq": 0, "summary": "test scene", "narration": "", "time": "", "location": "", "characters": [], "actions": [], "emotions": []}],
        "characters": {},
        "title": "test"
    }
    agent = StoryboardAgent(mock_llm)
    result = await agent.run(story_graph=sample)
    assert result["total_pages"] >= 1
