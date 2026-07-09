import pytest
from unittest.mock import MagicMock, AsyncMock
from app.agents.director_agent import DirectorAgent


@pytest.fixture
def sample_storyboard_with_dialogue():
    return {
        "total_pages": 2,
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "公园全景", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "小明散步", "dialogue": "今天天气真好", "source_scene_seq": 0}
        ]
    }


@pytest.fixture
def sample_style_config():
    return {
        "style_key": "miyazaki",
        "style_name": "宫崎骏风",
        "prompt_template": "调整为{storyboard_json}风格",
        "mood": "温馨",
        "dialogue_level": "少",
        "cinematic": 1
    }


@pytest.fixture
def sample_director_output():
    return {
        "style_notes": "应用宫崎骏风格，色调柔和",
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "柔和的公园全景", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "阳光下小明散步", "dialogue": "今天天气真好", "source_scene_seq": 0}
        ],
        "director_applied": True
    }


@pytest.fixture
def mock_llm(sample_director_output):
    llm = MagicMock()
    llm.chat_json = AsyncMock(return_value=sample_director_output)
    return llm


@pytest.mark.asyncio
async def test_director_agent_run(mock_llm, sample_storyboard_with_dialogue, sample_style_config):
    agent = DirectorAgent(mock_llm)
    result = await agent.run(storyboard_with_dialogue=sample_storyboard_with_dialogue, style_config=sample_style_config)
    assert "style_notes" in result
    assert result["director_applied"] is True
    assert len(result["pages"]) == 2
    assert result["pages"][0]["source_scene_seq"] == 0


@pytest.mark.asyncio
async def test_director_agent_fallback(mock_llm, sample_storyboard_with_dialogue, sample_style_config):
    """When LLM returns empty, should return original pages with director_applied=False."""
    mock_llm.chat_json = AsyncMock(return_value={})
    agent = DirectorAgent(mock_llm)
    result = await agent.run(storyboard_with_dialogue=sample_storyboard_with_dialogue, style_config=sample_style_config)
    assert result["director_applied"] is False
    assert len(result["pages"]) == 2
    assert result["pages"][0]["desc"] == "公园全景"  # 原始描述未变


@pytest.mark.asyncio
async def test_director_agent_llm_error(mock_llm, sample_storyboard_with_dialogue, sample_style_config):
    """When LLM raises exception, should fallback to original pages."""
    mock_llm.chat_json = AsyncMock(side_effect=Exception("LLM error"))
    agent = DirectorAgent(mock_llm)
    result = await agent.run(storyboard_with_dialogue=sample_storyboard_with_dialogue, style_config=sample_style_config)
    assert result["director_applied"] is False
    assert len(result["pages"]) == 2
