import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.schemas.common import APIResponse
from app.services.character_service import CharacterService

logger = logging.getLogger(__name__)

get_session_override = None


def _get_session():
    if get_session_override:
        return get_session_override()
    return get_session()


def create_characters_router(character_service: CharacterService = None) -> APIRouter:
    router = APIRouter(prefix="/api/v1/characters", tags=["characters"])

    async def get_svc() -> CharacterService:
        return character_service or CharacterService()

    @router.get("/{character_id}", response_model=APIResponse[dict])
    async def get_character(
        character_id: int,
        svc: CharacterService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """获取角色信息（跨故事聚合）。"""
        result = await svc.get_character(character_id, session)
        if not result:
            raise HTTPException(status_code=404, detail="Character not found")
        return APIResponse(data=result)

    return router


characters_router = create_characters_router()
