import pytest
from unittest.mock import MagicMock, AsyncMock
from app.agents.dialogue_agent import DialogueAgent


@pytest.fixture
def sample_storyboard():
    return {
        "total_pages": 3,
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "公园全景", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "小明散步", "source_scene_seq": 0},
            {"page": 3, "shot": "Close-up", "desc": "小明和朋友聊天", "source_scene_seq": 1}
        ]
    }


@pytest.fixture
def sample_dialogue_output():
    return {
        "pages": [
            {"page": 1, "dialogue": ""},
            {"page": 2, "dialogue": "今天天气真好"},
            {"page": 3, "dialogue": "好久不见！最近怎么样？"}
        ]
    }


@pytest.fixture
def mock_llm(sample_dialogue_output):
    llm = MagicMock()
    llm.chat_json = AsyncMock(return_value=sample_dialogue_output)
    return llm


@pytest.mark.asyncio
async def test_dialogue_agent_run(mock_llm, sample_storyboard):
    agent = DialogueAgent(mock_llm)
    result = await agent.run(storyboard=sample_storyboard, story_title="test")
    assert "pages" in result
    assert len(result["pages"]) == 3
    assert "dialogue" in result["pages"][0]


@pytest.mark.asyncio
async def test_dialogue_agent_empty_storyboard(mock_llm):
    agent = DialogueAgent(mock_llm)
    result = await agent.run(storyboard={"total_pages": 0, "pages": []}, story_title="test")
    assert result["pages"] == []


@pytest.mark.asyncio
async def test_dialogue_agent_fallback(mock_llm):
    """When LLM returns empty, should return pages with empty dialogue."""
    mock_llm.chat_json = AsyncMock(return_value={})
    sample = {"total_pages": 1, "pages": [{"page": 1, "shot": "Medium", "desc": "test", "source_scene_seq": 0}]}
    agent = DialogueAgent(mock_llm)
    result = await agent.run(storyboard=sample, story_title="test")
    assert len(result["pages"]) == 1
    assert result["pages"][0]["dialogue"] == ""


@pytest.mark.asyncio
async def test_dialogue_agent_llm_error(mock_llm):
    """When LLM raises exception, should fallback."""
    mock_llm.chat_json = AsyncMock(side_effect=Exception("LLM error"))
    sample = {"total_pages": 1, "pages": [{"page": 1, "shot": "Medium", "desc": "test", "source_scene_seq": 0}]}
    agent = DialogueAgent(mock_llm)
    result = await agent.run(storyboard=sample, story_title="test")
    assert len(result["pages"]) == 1
