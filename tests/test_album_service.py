import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.models import Base, Album, Photo


@pytest_asyncio.fixture
async def db_session():
    """创建 SQLite 内存测试数据库"""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class TestAlbumService:
    @pytest.mark.asyncio
    async def test_create_album(self, db_session):
        """验证创建相册"""
        from app.services.album_service import AlbumService

        svc = AlbumService()
        result = await svc.create_album(
            title="东京一日",
            date="2026-07-07",
            files=[],
            session=db_session,
        )

        assert result["id"] is not None
        assert result["title"] == "东京一日"
        assert result["photo_count"] == 0

    @pytest.mark.asyncio
    async def test_list_albums(self, db_session):
        """验证分页获取相册列表"""
        from app.services.album_service import AlbumService

        svc = AlbumService()
        await svc.create_album("A1", "2026-07-01", [], db_session)
        await svc.create_album("A2", "2026-07-02", [], db_session)

        result = await svc.list_albums(page=1, page_size=10, session=db_session)
        assert result["total"] == 2
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_get_album_with_photos(self, db_session):
        """验证获取相册详情（含照片列表）"""
        from app.services.album_service import AlbumService

        svc = AlbumService()
        album = await svc.create_album("测试", "2026-07-07", [], db_session)

        # 手动添加照片
        photo = Photo(album_id=album["id"], path="/uploads/test.jpg", taken_at="2026-07-07 10:00:00")
        db_session.add(photo)
        await db_session.commit()

        result = await svc.get_album(album["id"], db_session)
        assert result["id"] == album["id"]
        assert len(result["photos"]) == 1

    @pytest.mark.asyncio
    async def test_get_album_not_found(self, db_session):
        """验证相册不存在返回 None"""
        from app.services.album_service import AlbumService

        svc = AlbumService()
        result = await svc.get_album(99999, db_session)
        assert result is None
