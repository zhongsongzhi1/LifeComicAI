import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def mock_story_service():
    svc = MagicMock()
    svc.generate_story = AsyncMock(return_value={
        "id": 1, "album_id": 1, "title": "测试故事",
        "summary": "摘要", "status": "processing", "scene_count": 0, "created_at": None,
    })
    svc.get_story = AsyncMock(return_value={
        "id": 1, "album_id": 1, "title": "测试故事",
        "summary": "摘要", "status": "completed", "scene_count": 3, "created_at": None,
    })
    svc.get_story_graph = AsyncMock(return_value={
        "story_id": 1, "story_title": "测试故事", "story_summary": "摘要",
        "scenes": [], "global_characters": [],
    })
    svc.get_scene_detail = AsyncMock(return_value={
        "id": 1, "story_id": 1, "seq_num": 1,
        "time_at": "08:30", "location": "测试", "summary": "测试场景",
        "characters": [], "actions": [], "emotions": [], "source_photos": [],
    })
    return svc


@pytest.fixture
def client(mock_story_service):
    from app.api.stories import create_stories_router

    app = FastAPI()
    router = create_stories_router(mock_story_service)
    app.include_router(router)

    # 覆盖数据库依赖
    from app.api import stories as stories_module
    stories_module.get_session_override = lambda: AsyncMock()

    with TestClient(app) as c:
        yield c


class TestStoryAPI:
    def test_generate_story(self, client):
        """验证触发故事生成"""
        response = client.post("/api/v1/stories/generate", json={"album_id": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "processing"

    def test_get_story(self, client):
        """验证获取故事详情"""
        response = client.get("/api/v1/stories/1")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "completed"

    def test_get_story_graph(self, client):
        """验证获取故事图谱"""
        response = client.get("/api/v1/stories/1/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_get_scene_detail(self, client):
        """验证获取场景详情"""
        response = client.get("/api/v1/stories/1/scenes/1")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["seq_num"] == 1
