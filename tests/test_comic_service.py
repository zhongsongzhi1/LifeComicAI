import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.comic_service import ComicService
from app.db.models import Story, Comic, ComicPage, GenerationTask, PromptVersion
from app.db.seed_prompts import seed_prompt_versions


@pytest.fixture
def mock_storyboard_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "total_pages": 2,
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "封面全景", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "小明微笑", "source_scene_seq": 1}
        ]
    })
    return agent


@pytest.fixture
def mock_dialogue_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "pages": [
            {"page": 1, "dialogue": ""},
            {"page": 2, "dialogue": "今天真开心"}
        ],
        "dialogue_partial": False
    })
    return agent


@pytest.fixture
def mock_director_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "style_notes": "宫崎骏风格调整完成",
        "pages": [
            {"page": 1, "shot": "Wide", "desc": "柔和色调的公园全景", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "desc": "温暖光线中微笑的小明", "dialogue": "今天真开心", "source_scene_seq": 1}
        ],
        "director_applied": True
    })
    return agent


@pytest.fixture
def mock_comic_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "pages": [
            {"page": 1, "shot": "Wide", "image_prompt": "test", "image_url": "/tmp/1/page_1.png", "desc": "test", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "image_prompt": "test", "image_url": "/tmp/1/page_2.png", "desc": "test", "dialogue": "hi", "source_scene_seq": 1}
        ],
        "image_partial": False,
        "all_failed": False
    })
    return agent


@pytest.fixture
def mock_layout_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={"pdf_path": "/tmp/1/comic.pdf", "success": True})
    return agent


@pytest.fixture
def comic_service(mock_storyboard_agent, mock_dialogue_agent, mock_director_agent, mock_comic_agent, mock_layout_agent):
    return ComicService(
        storyboard_agent=mock_storyboard_agent,
        dialogue_agent=mock_dialogue_agent,
        director_agent=mock_director_agent,
        comic_agent=mock_comic_agent,
        layout_agent=mock_layout_agent,
    )


@pytest_asyncio.fixture
async def setup_story(session: AsyncSession):
    """创建一个 Story 用于测试。"""
    story = Story(album_id=1, title="测试故事", summary="测试摘要", status="completed", scene_count=1)
    session.add(story)
    await session.commit()
    await session.refresh(story)
    return story


@pytest.mark.asyncio
async def test_generate_comic_success(comic_service, setup_story, session):
    """Mock 所有 agent 成功，验证 Comic + ComicPages + GenerationTasks 入库。"""
    # 先 seed prompt versions
    await seed_prompt_versions(session)
    
    result = await comic_service.generate_comic(setup_story.id, "miyazaki", session)
    
    assert result is not None
    assert result["status"] == "completed"
    assert result["pdf_path"] != ""
    
    # 验证 Comic 记录
    stmt = select(Comic).where(Comic.id == result["id"])
    res = await session.execute(stmt)
    comic = res.scalar_one()
    assert comic.story_id == setup_story.id
    assert comic.style == "miyazaki"
    
    # 验证 ComicPage 记录
    stmt = select(ComicPage).where(ComicPage.comic_id == comic.id)
    res = await session.execute(stmt)
    pages = res.scalars().all()
    assert len(pages) == 2
    
    # 验证 GenerationTask 记录
    stmt = select(GenerationTask).where(GenerationTask.comic_id == comic.id)
    res = await session.execute(stmt)
    tasks = res.scalars().all()
    assert len(tasks) == 5  # 5 agents


@pytest.mark.asyncio
async def test_generate_comic_invalid_style(comic_service, setup_story, session):
    """无效 style_key 应返回 None。"""
    await seed_prompt_versions(session)
    result = await comic_service.generate_comic(setup_story.id, "nonexistent", session)
    assert result is None


@pytest.mark.asyncio
async def test_generate_comic_story_not_found(comic_service, session):
    """不存在的 story_id 返回 None。"""
    result = await comic_service.generate_comic(99999, "miyazaki", session)
    assert result is None


@pytest.mark.asyncio
async def test_generate_comic_all_images_failed(comic_service, setup_story, session, mock_comic_agent):
    """Mock comic_agent 返回 all_failed=True，验证 comic.status="failed"。"""
    await seed_prompt_versions(session)
    mock_comic_agent.run = AsyncMock(return_value={
        "pages": [
            {"page": 1, "shot": "Wide", "image_prompt": "test", "image_url": "", "desc": "test", "dialogue": "", "source_scene_seq": 0},
            {"page": 2, "shot": "Medium", "image_prompt": "test", "image_url": "", "desc": "test", "dialogue": "hi", "source_scene_seq": 1}
        ],
        "image_partial": False,
        "all_failed": True
    })
    
    result = await comic_service.generate_comic(setup_story.id, "miyazaki", session)
    assert result is not None
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_get_comic_pages(comic_service, setup_story, session):
    """生成后调用 get_comic_pages 验证返回页面列表。"""
    await seed_prompt_versions(session)
    generate_result = await comic_service.generate_comic(setup_story.id, "miyazaki", session)
    
    pages_result = await comic_service.get_comic_pages(generate_result["id"], session)
    assert pages_result is not None
    assert pages_result["comic_id"] == generate_result["id"]
    assert len(pages_result["pages"]) == 2


@pytest.mark.asyncio
async def test_list_styles(comic_service, session):
    """seed 后调用 list_styles，验证返回 4 种风格。"""
    await seed_prompt_versions(session)
    styles = await comic_service.list_styles(session)
    assert len(styles) == 4
    style_keys = {s["key"] for s in styles}
    assert style_keys == {"miyazaki", "cinematic", "slice", "manga"}
