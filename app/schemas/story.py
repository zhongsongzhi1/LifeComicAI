from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StoryGenerateRequest(BaseModel):
    """故事生成请求。"""
    album_id: int = Field(..., description="相册 ID")


class ActionResponse(BaseModel):
    """动作响应。"""
    verb: str = ""
    object: str = ""


class EmotionResponse(BaseModel):
    """情绪响应。"""
    type: str = ""
    intensity: float = 0.0


class CharacterResponse(BaseModel):
    """角色响应。"""
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str = ""
    traits: List[str] = Field(default_factory=list)
    role: str = ""
    avatar_url: Optional[str] = ""


class SceneResponse(BaseModel):
    """场景响应。"""
    model_config = ConfigDict(from_attributes=True)
    id: int
    story_id: int
    seq_num: int
    time_at: Optional[str] = ""
    location: Optional[str] = ""
    summary: Optional[str] = ""
    characters: List[CharacterResponse] = Field(default_factory=list)
    actions: List[ActionResponse] = Field(default_factory=list)
    emotions: List[EmotionResponse] = Field(default_factory=list)
    source_photos: List[int] = Field(default_factory=list)


class StoryResponse(BaseModel):
    """故事响应。"""
    model_config = ConfigDict(from_attributes=True)
    id: int
    album_id: int
    title: str = ""
    summary: Optional[str] = ""
    status: str = "processing"
    scene_count: int = 0
    created_at: Optional[datetime] = None


class StoryGraphResponse(BaseModel):
    """完整故事图谱响应。"""
    story_id: int
    story_title: str = ""
    story_summary: Optional[str] = ""
    scenes: List[SceneResponse] = Field(default_factory=list)
    global_characters: List[CharacterResponse] = Field(default_factory=list)


class SceneDetailResponse(SceneResponse):
    """场景详情响应（含来源照片完整信息）。"""
    source_photos: List[Dict[str, Any]] = Field(default_factory=list)
