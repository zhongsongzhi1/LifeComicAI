import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.db.models import Album, Story
from app.schemas.common import APIResponse
from app.schemas.story import StoryGenerateRequest
from app.services.story_service import StoryService
from app.llm.qianfan_provider import QianfanProvider
from app.llm.openai_provider import OpenAIProvider
from app.agents.album_agent import AlbumAgent
from app.agents.vision_agent import VisionAgent
from app.agents.story_agent import StoryAgent
from app.config.config import get_settings
from app.services.cache_service import CacheService
from sqlalchemy import select

logger = logging.getLogger(__name__)

get_session_override = None

# 全局缓存服务实例（内存缓存，进程内共享）
_cache_service = None


def _get_cache_service() -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def _get_session():
    if get_session_override:
        return get_session_override()
    return get_session()


def _build_story_service() -> StoryService:
    settings = get_settings()
    qianfan = QianfanProvider(
        access_key=settings.QIANFAN_ACCESS_KEY,
        secret_key=settings.QIANFAN_SECRET_KEY,
    )
    openai_provider = OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
    )
    cache_svc = _get_cache_service()
    return StoryService(
        album_agent=AlbumAgent(llm_provider=openai_provider),
        vision_agent=VisionAgent(qianfan_provider=qianfan, cache_service=cache_svc),
        story_agent=StoryAgent(llm_provider=openai_provider),
    )


async def _generate_story_background(album_id: int, session_factory):
    """后台异步执行故事生成流水线。"""
    from app.db.database import async_session_factory as factory

    async with factory() as session:
        try:
            svc = _build_story_service()
            await svc.generate_story(album_id=album_id, session=session)
        except Exception as e:
            logger.error(f"Background story generation failed for album {album_id}: {e}")


def create_stories_router(story_service: StoryService = None) -> APIRouter:
    router = APIRouter(prefix="/api/v1/stories", tags=["stories"])

    async def get_svc() -> StoryService:
        return story_service or _build_story_service()

    @router.post("/generate", response_model=APIResponse[dict])
    async def generate_story(
        request: StoryGenerateRequest,
        background_tasks: BackgroundTasks,
        svc: StoryService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """触发 Vision Agent + Story Agent（异步，立即返回 story_id）。

        先创建 Story 记录（status=processing），然后通过 BackgroundTasks
        异步执行 Agent 流水线。
        """
        # 先验证相册存在
        album_query = select(Album).where(Album.id == request.album_id)
        album_result = await session.execute(album_query)
        album = album_result.scalar_one_or_none()

        if not album:
            raise HTTPException(status_code=404, detail="Album not found")

        # 创建 processing 状态的 Story
        story = Story(album_id=request.album_id, title="", summary="", status="processing", scene_count=0)
        session.add(story)
        await session.commit()
        await session.refresh(story)

        # 后台异步执行
        background_tasks.add_task(_generate_story_background, request.album_id, None)

        return APIResponse(data={
            "id": story.id,
            "album_id": story.album_id,
            "title": story.title,
            "summary": story.summary,
            "status": story.status,
            "scene_count": story.scene_count,
            "created_at": story.created_at,
        })

    @router.get("/{story_id}", response_model=APIResponse[dict])
    async def get_story(
        story_id: int,
        svc: StoryService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """获取故事详情（含状态）。"""
        result = await svc.get_story(story_id, session)
        if not result:
            raise HTTPException(status_code=404, detail="Story not found")
        return APIResponse(data=result)

    @router.get("/{story_id}/graph", response_model=APIResponse[dict])
    async def get_story_graph(
        story_id: int,
        svc: StoryService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """获取完整 Story Graph。"""
        result = await svc.get_story_graph(story_id, session)
        if not result:
            raise HTTPException(status_code=404, detail="Story not found")
        return APIResponse(data=result)

    @router.get("/{story_id}/scenes/{scene_id}", response_model=APIResponse[dict])
    async def get_scene_detail(
        story_id: int,
        scene_id: int,
        svc: StoryService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """获取单个场景详情（含来源照片）。"""
        result = await svc.get_scene_detail(scene_id, session)
        if not result:
            raise HTTPException(status_code=404, detail="Scene not found")
        if result["story_id"] != story_id:
            raise HTTPException(status_code=404, detail="Scene not found in this story")
        return APIResponse(data=result)

    return router


stories_router = create_stories_router()
