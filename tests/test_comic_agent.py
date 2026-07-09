import os
import pytest
from unittest.mock import MagicMock, patch
from app.agents.comic_agent import ComicAgent


@pytest.fixture
def sample_storyboard():
    return {
        "total_pages": 2,
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "公园全景，阳光明媚", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "小明微笑", "dialogue": "今天真开心", "source_scene_seq": 1}
        ]
    }


@pytest.fixture
def mock_image_provider():
    provider = MagicMock()
    provider.generate_with_retry = MagicMock(return_value={"urls": ["https://example.com/image.png"]})
    return provider


@pytest.mark.asyncio
async def test_build_prompt(mock_image_provider, tmp_path):
    """测试 _build_image_prompt 输出包含关键元素。"""
    agent = ComicAgent(mock_image_provider, str(tmp_path))
    prompt = agent._build_image_prompt("宫崎骏风", "公园全景", "你好")
    assert "宫崎骏风" in prompt
    assert "公园全景" in prompt
    assert "你好" in prompt
    assert "漫画" in prompt


@pytest.mark.asyncio
async def test_generates_and_downloads(mock_image_provider, tmp_path):
    """Mock 生图成功 + 下载成功，断言本地文件存在。"""
    with patch("app.agents.comic_agent.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        agent = ComicAgent(mock_image_provider, str(tmp_path))
        result = await agent.run(
            revised_storyboard={
                "total_pages": 1,
                "pages": [{"page": 1, "shot": "Wide", "desc": "test", "dialogue": "", "source_scene_seq": 0}]
            },
            comic_id=1,
            style_key="miyazaki",
            style_name="宫崎骏风"
        )

    assert result["image_partial"] is False
    assert result["all_failed"] is False
    assert len(result["pages"]) == 1
    assert "image_url" in result["pages"][0]
    assert os.path.exists(result["pages"][0]["image_url"])


@pytest.mark.asyncio
async def test_partial_failure(mock_image_provider, tmp_path):
    """Mock 第1页成功、第2页失败，断言 image_partial=True。"""
    call_count = [0]

    def mock_generate(prompt, size="1024x1024", max_retries=2):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"urls": ["https://example.com/page1.png"]}
        return None

    mock_image_provider.generate_with_retry = mock_generate

    with patch("app.agents.comic_agent.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        agent = ComicAgent(mock_image_provider, str(tmp_path))
        result = await agent.run(
            revised_storyboard={
                "total_pages": 2,
                "pages": [
                    {"page": 1, "shot": "Wide", "desc": "test1", "dialogue": "", "source_scene_seq": 0},
                    {"page": 2, "shot": "Medium", "desc": "test2", "dialogue": "hi", "source_scene_seq": 1}
                ]
            },
            comic_id=1,
            style_key="miyazaki",
            style_name="宫崎骏风"
        )

    assert result["image_partial"] is True
    assert result["all_failed"] is False
    assert result["pages"][1]["image_url"] == ""


@pytest.mark.asyncio
async def test_all_failed(mock_image_provider, tmp_path):
    """所有页生图失败，断言 all_failed=True。"""
    mock_image_provider.generate_with_retry = MagicMock(return_value=None)
    agent = ComicAgent(mock_image_provider, str(tmp_path))
    result = await agent.run(
        revised_storyboard={
            "total_pages": 1,
            "pages": [{"page": 1, "shot": "Wide", "desc": "test", "dialogue": "", "source_scene_seq": 0}]
        },
        comic_id=1,
        style_key="miyazaki",
        style_name="宫崎骏风"
    )
    assert result["all_failed"] is True
    assert result["image_partial"] is False
