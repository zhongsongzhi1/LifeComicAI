import logging
import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.strip_layout_agent import StripLayoutAgent
from app.agents.strip_script_agent import StripScriptAgent
from app.agents.vision_agent import VisionAgent
from app.config.config import get_settings
from app.db.database import get_session
from app.db.models import QuickStrip
from app.llm.modelscope_image_provider import ModelScopeImageProvider
from app.llm.qianfan_provider import QianfanProvider
from app.schemas.common import APIResponse
from app.services.cache_service import CacheService
from app.services.strip_service import StripService

logger = logging.getLogger(__name__)
get_session_override = None

# 全局缓存服务实例（内存缓存，进程内共享）
_cache_service = None


def _get_cache_service() -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


async def _get_session():
    if get_session_override:
        yield get_session_override()
    else:
        async for s in get_session():
            yield s


def _build_strip_service() -> StripService:
    settings = get_settings()
    qianfan = QianfanProvider(access_key=settings.QIANFAN_ACCESS_KEY, secret_key=settings.QIANFAN_SECRET_KEY)
    qianfan.configure(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    cache_svc = _get_cache_service()
    img_p = ModelScopeImageProvider(
        access_key=settings.MODELSCOPE_API_KEY,
        cache_service=cache_svc,
    )
    return StripService(
        vision_agent=VisionAgent(qianfan_provider=qianfan, cache_service=cache_svc),
        script_agent=StripScriptAgent(llm_provider=qianfan),
        image_provider=img_p,
        layout_agent=StripLayoutAgent(),
        comics_dir=settings.COMICS_DIR,
    )


strips_router = APIRouter(prefix="/api/v1/strip", tags=["strip"])


@strips_router.post("/generate", response_model=APIResponse[dict])
async def generate_strip(
    files: List[UploadFile] = File(...),
    svc: StripService = Depends(lambda: _build_strip_service()),
    session: AsyncSession = Depends(_get_session),
):
    MAX_FILE_SIZE = 50 * 1024 * 1024
    if not (2 <= len(files) <= 6):
        raise HTTPException(status_code=400, detail="请上传2-6张照片")
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
    upload_dir = os.path.join(get_settings().UPLOAD_DIR, "strip_photos")
    os.makedirs(upload_dir, exist_ok=True)
    photo_paths = []
    try:
        for f in files:
            if f.content_type not in allowed:
                raise HTTPException(status_code=400, detail=f"不支持的格式: {f.content_type}")
            ext = os.path.splitext(f.filename or "p.jpg")[1].lower() or ".jpg"
            if ext not in allowed_ext:
                ext = ".jpg"
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"单张图片大小不能超过50MB: {f.filename}")
            fp = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
            with open(fp, "wb") as out:
                out.write(content)
            photo_paths.append(fp)
        result = await svc.generate_strip(photo_paths=photo_paths, session=session)
        return APIResponse(code=0, message="success", data={
            "strip_id": result["strip_id"], "status": result["status"],
            "title": result["title"],
            "image_url": f"/api/v1/strip/{result['strip_id']}/image",
            "panels": result.get("panels", []),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strip gen error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"条漫生成失败: {str(e)[:200]}")


@strips_router.get("/{strip_id}", response_model=APIResponse[dict])
async def get_strip(strip_id: int, session: AsyncSession = Depends(_get_session)):
    r = await session.execute(select(QuickStrip).where(QuickStrip.id == strip_id))
    s = r.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="条漫不存在")
    return APIResponse(code=0, data={
        "strip_id": s.id, "title": s.title, "status": s.status,
        "image_url": f"/api/v1/strip/{s.id}/image" if s.status == "completed" else None,
        "error_msg": s.error_msg,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    })


@strips_router.get("/{strip_id}/image")
async def get_strip_image(strip_id: int, session: AsyncSession = Depends(_get_session)):
    r = await session.execute(select(QuickStrip).where(QuickStrip.id == strip_id))
    s = r.scalar_one_or_none()
    if not s or not s.image_path or not os.path.exists(s.image_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(s.image_path, media_type="image/png")
