import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text


@pytest.mark.asyncio
async def test_create_and_query_album():
    """验证 Album 模型可创建和查询"""
    from app.db.models import Album, Photo
    from app.db.database import get_session

    # 使用 SQLite 内存数据库进行测试
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        album = Album(title="东京一日", date="2026-07-07")
        session.add(album)
        await session.commit()
        await session.refresh(album)

        assert album.id is not None
        assert album.title == "东京一日"
        assert album.date == "2026-07-07"

    await engine.dispose()


@pytest.mark.asyncio
async def test_photo_belongs_to_album():
    """验证 Photo 与 Album 的外键关系"""
    from app.db.models import Album, Photo, Base
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        album = Album(title="test", date="2026-07-07")
        session.add(album)
        await session.flush()

        photo = Photo(album_id=album.id, path="/uploads/test.jpg", taken_at="2026-07-07 10:00:00")
        session.add(photo)
        await session.commit()
        await session.refresh(photo)

        assert photo.album_id == album.id

    await engine.dispose()


@pytest.mark.asyncio
async def test_story_has_scenes():
    """验证 Story 与 Scene 的一对多关系"""
    from app.db.models import Story, Scene, Base
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        story = Story(album_id=1, title="东京一日", summary="测试故事")
        session.add(story)
        await session.flush()

        scene = Scene(story_id=story.id, seq_num=1, time_at="10:00", location="东京塔", summary="参观东京塔")
        session.add(scene)
        await session.commit()
        await session.refresh(scene)

        assert scene.story_id == story.id
        assert scene.seq_num == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_all_tables_exist():
    """验证所有 10 张表均可创建"""
    from app.db.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    expected_tables = {
        "albums", "photos", "photo_analysis",
        "stories", "scenes", "characters",
        "scene_characters", "scene_actions", "scene_emotions", "scene_photos",
        "comics", "comic_pages", "generation_tasks", "prompt_versions",
    }

    async with engine.connect() as conn:
        result = await conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        actual_tables = {row[0] for row in result.fetchall()}

    missing = expected_tables - actual_tables
    assert not missing, f"Missing tables: {missing}"

    await engine.dispose()
