import logging
import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.db.models import Story, Comic, ComicPage, GenerationTask
from app.schemas.common import APIResponse
from app.schemas.comic import ComicGenerateRequest
from app.services.comic_service import ComicService
from app.llm.openai_provider import OpenAIProvider
from app.llm.qianfan_image_provider import QianfanImageProvider
from app.agents.storyboard_agent import StoryboardAgent
from app.agents.dialogue_agent import DialogueAgent
from app.agents.director_agent import DirectorAgent
from app.agents.comic_agent import ComicAgent as ComicGenAgent
from app.agents.layout_agent import LayoutAgent
from app.config.config import get_settings
from sqlalchemy import select

logger = logging.getLogger(__name__)

get_session_override = None


def _get_session():
    if get_session_override:
        return get_session_override()
    return get_session()


def _build_comic_service() -> ComicService:
    settings = get_settings()
    openai_provider = OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
    )
    image_provider = QianfanImageProvider(
        access_key=settings.QIANFAN_ACCESS_KEY,
        secret_key=settings.QIANFAN_SECRET_KEY,
    )
    return ComicService(
        storyboard_agent=StoryboardAgent(llm_provider=openai_provider),
        dialogue_agent=DialogueAgent(llm_provider=openai_provider),
        director_agent=DirectorAgent(llm_provider=openai_provider),
        comic_agent=ComicGenAgent(image_provider=image_provider, comics_dir=settings.COMICS_DIR),
        layout_agent=LayoutAgent(comics_dir=settings.COMICS_DIR),
    )


async def _generate_comic_background(comic_id: int, story_id: int, style_key: str, session_factory):
    """后台异步执行漫画生成流水线。"""
    from app.db.database import async_session_factory as factory

    async with factory() as session:
        try:
            svc = _build_comic_service()
            await svc.generate_comic(story_id=story_id, style_key=style_key, session=session)
        except Exception as e:
            logger.error(f"Background comic generation failed for story {story_id}: {e}")


def create_comics_router(comic_service: ComicService = None) -> APIRouter:
    router = APIRouter(prefix="/api/v1/comics", tags=["comics"])

    async def get_svc() -> ComicService:
        return comic_service or _build_comic_service()

    @router.post("/generate", response_model=APIResponse[dict])
    async def generate_comic(
        request: ComicGenerateRequest,
        background_tasks: BackgroundTasks,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """触发漫画生成流水线（异步后台执行）。

        先通过 Service 验证 Story 和风格存在，创建 Comic 记录，
        然后通过 BackgroundTasks 异步执行 Agent 流水线。
        """
        # 通过 Service 验证
        result = await svc.generate_comic(story_id=request.story_id, style_key=request.style_key, session=session)
        if not result:
            return APIResponse(code=400, message="Invalid style or story not found", data=None)

        # 创建 Comic 记录
        comic = Comic(story_id=request.story_id, style=request.style_key, status="processing")
        session.add(comic)
        await session.commit()
        await session.refresh(comic)

        # 后台异步执行
        background_tasks.add_task(
            _generate_comic_background,
            comic.id, request.story_id, request.style_key, None
        )

        return APIResponse(data={
            "comic_id": comic.id,
            "story_id": comic.story_id,
            "style": comic.style,
            "status": comic.status,
            "created_at": comic.created_at,
        })

    @router.get("/styles", response_model=APIResponse[list])
    async def list_styles(
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """列出所有可用的漫画风格预设。"""
        styles = await svc.list_styles(session)
        return APIResponse(data=styles)

    @router.get("/{comic_id}", response_model=APIResponse[dict])
    async def get_comic(
        comic_id: int,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """获取漫画详情（含状态）。"""
        result = await svc.get_comic(comic_id, session)
        if not result:
            return APIResponse(code=404, message="Comic not found", data=None)
        return APIResponse(data=result)

    @router.get("/{comic_id}/pages", response_model=APIResponse[dict])
    async def get_comic_pages(
        comic_id: int,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """获取漫画所有页面。"""
        result = await svc.get_comic_pages(comic_id, session)
        if not result:
            return APIResponse(code=404, message="Comic not found", data=None)
        return APIResponse(data=result)

    @router.get("/{comic_id}/pages/{page_num}", response_model=APIResponse[dict])
    async def get_single_page(
        comic_id: int,
        page_num: int,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """获取单个漫画页面。"""
        result = await svc.get_comic_pages(comic_id, session)
        if not result:
            return APIResponse(code=404, message="Comic not found", data=None)
        pages = result.get("pages", [])
        for p in pages:
            if p.get("page_num") == page_num:
                return APIResponse(data=p)
        return APIResponse(code=404, message="Page not found", data=None)

    @router.get("/{comic_id}/download")
    async def download_pdf(
        comic_id: int,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """下载漫画 PDF 文件。"""
        result = await svc.get_comic(comic_id, session)
        if not result:
            return APIResponse(code=404, message="Comic not found", data=None)

        pdf_path = result.get("pdf_path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            return APIResponse(code=404, message="PDF not available", data=None)

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"comic_{comic_id}.pdf"
        )

    @router.get("/{comic_id}/tasks", response_model=APIResponse[dict])
    async def get_comic_tasks(
        comic_id: int,
        svc: ComicService = Depends(get_svc),
        session: AsyncSession = Depends(get_session),
    ):
        """获取漫画生成任务列表（含各 Agent 耗时）。"""
        result = await svc.get_comic_tasks(comic_id, session)
        if not result:
            return APIResponse(code=404, message="Comic not found", data=None)
        return APIResponse(data=result)

    return router


comics_router = create_comics_router()
