import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.db.models import Base, Character


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class TestCharacterService:
    @pytest.mark.asyncio
    async def test_get_character(self, db_session):
        """验证获取角色信息"""
        from app.services.character_service import CharacterService

        char = Character(name="我", traits=["长发", "眼镜"])
        db_session.add(char)
        await db_session.commit()

        svc = CharacterService()
        result = await svc.get_character(char.id, db_session)

        assert result is not None
        assert result["name"] == "我"
        assert result["traits"] == ["长发", "眼镜"]

    @pytest.mark.asyncio
    async def test_get_character_not_found(self, db_session):
        """验证角色不存在返回 None"""
        from app.services.character_service import CharacterService

        svc = CharacterService()
        result = await svc.get_character(99999, db_session)
        assert result is None
