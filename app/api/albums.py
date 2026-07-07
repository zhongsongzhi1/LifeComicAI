import logging
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.schemas.common import APIResponse
from app.services.album_service import AlbumService

logger = logging.getLogger(__name__)

# 用于测试覆盖的依赖
get_session_override = None


def _get_session():
    if get_session_override:
        return get_session_override()
    return get_session()


def create_albums_router(album_service: AlbumService = None) -> APIRouter:
    """创建 albums 路由器（支持依赖注入用于测试）。"""
    router = APIRouter(prefix="/api/v1/albums", tags=["albums"])

    async def get_svc() -> AlbumService:
        return album_service or AlbumService()

    @router.post("/upload", response_model=APIResponse[dict])
    async def upload_album(
        title: str = Form(default=""),
        date: str = Form(default=""),
        files: List[UploadFile] = File(default_factory=list),
        svc: AlbumService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """上传多张照片，创建 Album，触发 Album Agent。"""
        # 校验文件格式
        allowed_exts = {".jpg", ".jpeg", ".png", ".heic"}
        for f in files:
            if f.filename:
                ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
                if f".{ext}" not in allowed_exts:
                    raise HTTPException(status_code=400, detail=f"不支持的图片格式: {ext}")

        result = await svc.create_album(title=title, date=date, files=files, session=session)
        return APIResponse(data=result)

    @router.get("", response_model=APIResponse[dict])
    async def list_albums(
        page: int = 1,
        page_size: int = 20,
        svc: AlbumService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """分页获取 Album 列表。"""
        result = await svc.list_albums(page=page, page_size=page_size, session=session)
        return APIResponse(data=result)

    @router.get("/{album_id}", response_model=APIResponse[dict])
    async def get_album(
        album_id: int,
        svc: AlbumService = Depends(get_svc),
        session: AsyncSession = Depends(_get_session),
    ):
        """获取 Album 详情（含 Timeline）。"""
        result = await svc.get_album(album_id, session)
        if not result:
            raise HTTPException(status_code=404, detail="Album not found")
        return APIResponse(data=result)

    return router


albums_router = create_albums_router()
