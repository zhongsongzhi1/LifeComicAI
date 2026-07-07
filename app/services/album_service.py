import logging
import os
import math
from typing import List, Optional

from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.config import get_settings
from app.db.models import Album, Photo

logger = logging.getLogger(__name__)


class AlbumService:
    """Album 业务编排。"""

    async def create_album(
        self,
        title: str,
        date: str,
        files: List[UploadFile],
        session: AsyncSession,
    ) -> dict:
        """创建相册并存储上传的照片文件。

        Returns:
            dict: AlbumResponse 序列化字典
        """
        album = Album(title=title, date=date)
        session.add(album)
        await session.flush()

        settings = get_settings()
        album_dir = os.path.join(settings.UPLOAD_DIR, str(album.id))
        os.makedirs(album_dir, exist_ok=True)

        photo_count = 0
        for file in files:
            if not file.filename:
                continue

            # 安全文件名
            safe_name = os.path.basename(file.filename)
            file_path = os.path.join(album_dir, safe_name)

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            photo = Photo(
                album_id=album.id,
                path=file_path,
                taken_at="",
            )
            session.add(photo)
            photo_count += 1

        album.photo_count = photo_count
        await session.commit()
        await session.refresh(album)

        return {
            "id": album.id,
            "title": album.title,
            "date": album.date,
            "photo_count": album.photo_count,
            "created_at": album.created_at,
            "photos": [],
        }

    async def list_albums(
        self,
        page: int,
        page_size: int,
        session: AsyncSession,
    ) -> dict:
        """分页获取 Album 列表。

        Returns:
            dict: PaginatedResponse[AlbumResponse] 序列化字典
        """
        count_query = select(func.count(Album.id))
        total = (await session.execute(count_query)).scalar() or 0

        offset = (page - 1) * page_size
        query = select(Album).order_by(Album.created_at.desc()).offset(offset).limit(page_size)
        result = await session.execute(query)
        albums = result.scalars().all()

        items = []
        for a in albums:
            items.append({
                "id": a.id,
                "title": a.title,
                "date": a.date,
                "photo_count": a.photo_count,
                "created_at": a.created_at,
                "photos": [],
            })

        total_pages = math.ceil(total / page_size) if page_size > 0 else 0

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def get_album(self, album_id: int, session: AsyncSession) -> Optional[dict]:
        """获取 Album 详情（含照片列表）。

        Returns:
            dict or None: AlbumResponse 序列化字典
        """
        query = select(Album).where(Album.id == album_id)
        result = await session.execute(query)
        album = result.scalar_one_or_none()

        if not album:
            return None

        photos = []
        for p in album.photos:
            photos.append({
                "id": p.id,
                "album_id": p.album_id,
                "path": p.path,
                "taken_at": p.taken_at,
            })

        return {
            "id": album.id,
            "title": album.title,
            "date": album.date,
            "photo_count": album.photo_count,
            "created_at": album.created_at,
            "photos": photos,
        }
