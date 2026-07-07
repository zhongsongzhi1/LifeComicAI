from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AlbumCreate(BaseModel):
    """创建相册请求。"""
    title: str = Field(default="", description="相册标题")
    date: str = Field(default="", description="相册日期，格式 YYYY-MM-DD")


class PhotoResponse(BaseModel):
    """照片响应。"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    album_id: int
    path: str
    taken_at: Optional[str] = ""


class AlbumResponse(BaseModel):
    """相册响应。"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    date: str
    photo_count: int = 0
    created_at: Optional[datetime] = None
    photos: List[PhotoResponse] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """时间线条目。"""
    seq: int
    time: str
    photos: List[int] = Field(default_factory=list, description="照片 ID 列表")
    label: str = ""


class TimelineResponse(BaseModel):
    """时间线响应。"""
    date: str
    entries: List[TimelineEntry] = Field(default_factory=list)
    photo_count: int = 0
