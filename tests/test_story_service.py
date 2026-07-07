import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.models import Base, Album, Photo, Story


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def mock_album_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "date": "2026-07-07",
        "entries": [
            {"seq": 1, "time": "08:30", "photos": [1], "label": "早餐"},
        ],
        "photo_count": 1,
    })
    return agent


@pytest.fixture
def mock_vision_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "photo_id": 1,
        "objects": [],
        "scene_desc": "测试场景",
        "weather": "",
        "location": "",
        "action": "",
        "emotion": "",
        "clothing": [],
        "people": [],
    })
    return agent


@pytest.fixture
def mock_story_agent():
    agent = MagicMock()
    agent.run = AsyncMock(return_value={
        "story_title": "测试故事",
        "story_summary": "测试摘要",
        "scenes": [
            {
                "seq": 1, "time": "08:30", "location": "测试地点",
                "summary": "测试场景摘要",
                "characters": [{"name": "我", "role": "主角"}],
                "actions": [{"verb": "测试", "object": "对象"}],
                "emotions": [{"type": "开心", "intensity": 0.8}],
                "source_photos": [1],
            },
        ],
        "global_characters": [
            {"name": "我", "traits": ["特征"], "first_appearance_scene": 1},
        ],
    })
    return agent


class TestStoryService:
    @pytest.mark.asyncio
    async def test_generate_story_success(self, db_session, mock_album_agent, mock_vision_agent, mock_story_agent):
        """验证完整的故事生成流程"""
        from app.services.story_service import StoryService

        # 准备测试数据
        album = Album(title="测试", date="2026-07-07")
        db_session.add(album)
        await db_session.flush()

        photo = Photo(album_id=album.id, path="/test.jpg", taken_at="2026-07-07 08:30:00")
        db_session.add(photo)
        await db_session.commit()

        svc = StoryService(
            album_agent=mock_album_agent,
            vision_agent=mock_vision_agent,
            story_agent=mock_story_agent,
        )

        result = await svc.generate_story(album_id=album.id, session=db_session)

        assert result["status"] == "completed"
        assert result["title"] == "测试故事"
        assert result["scene_count"] == 1

        # 验证 Agent 调用
        mock_album_agent.run.assert_called_once()
        mock_vision_agent.run.assert_called_once()
        mock_story_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_story_album_not_found(self, db_session, mock_album_agent, mock_vision_agent, mock_story_agent):
        """验证相册不存在时返回 None"""
        from app.services.story_service import StoryService

        svc = StoryService(
            album_agent=mock_album_agent,
            vision_agent=mock_vision_agent,
            story_agent=mock_story_agent,
        )

        result = await svc.generate_story(album_id=99999, session=db_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_story(self, db_session):
        """验证获取故事详情"""
        from app.services.story_service import StoryService

        album = Album(title="测试", date="2026-07-07")
        db_session.add(album)
        await db_session.flush()

        story = Story(album_id=album.id, title="测试故事", summary="摘要", status="completed", scene_count=2)
        db_session.add(story)
        await db_session.commit()

        svc = StoryService(
            album_agent=MagicMock(),
            vision_agent=MagicMock(),
            story_agent=MagicMock(),
        )

        result = await svc.get_story(story.id, db_session)
        assert result is not None
        assert result["title"] == "测试故事"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_story_not_found(self, db_session):
        """验证故事不存在返回 None"""
        from app.services.story_service import StoryService

        svc = StoryService(
            album_agent=MagicMock(),
            vision_agent=MagicMock(),
            story_agent=MagicMock(),
        )

        result = await svc.get_story(99999, db_session)
        assert result is None
