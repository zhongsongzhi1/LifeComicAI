from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ComicGenerateRequest(BaseModel):
    story_id: int = Field(..., description="故事 ID")
    style_key: str = Field(default="slice", description="风格预设 key")


class ComicResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    story_id: int
    style: str = ""
    status: str = "processing"
    total_pages: int = 0
    pdf_path: Optional[str] = ""
    image_partial: bool = False
    dialogue_partial: bool = False
    created_at: Optional[datetime] = None


class ComicPageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    comic_id: int
    page_num: int
    shot_type: Optional[str] = "Medium"
    image_prompt: Optional[str] = ""
    image_url: Optional[str] = ""
    dialogue: Optional[str] = ""
    narration: Optional[str] = ""
    layout_json: Optional[Dict[str, Any]] = None


class GenerationTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    comic_id: int
    agent_name: str
    status: str
    error_message: Optional[str] = ""
    retry_count: int = 0
    duration_ms: Optional[int] = None


class StylePresetResponse(BaseModel):
    key: str
    name: str
    mood: str
    dialogue_level: str
    cinematic: bool


class ComicPagesResponse(BaseModel):
    story_id: int
    comic_id: int
    pages: List[ComicPageResponse]


class ComicTasksResponse(BaseModel):
    comic_id: int
    status: str
    tasks: List[GenerationTaskResponse]
