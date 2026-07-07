import io
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def mock_album_service():
    svc = MagicMock()
    svc.create_album = AsyncMock(return_value={
        "id": 1, "title": "东京一日", "date": "2026-07-07",
        "photo_count": 0, "created_at": None, "photos": [],
    })
    svc.list_albums = AsyncMock(return_value={
        "items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 0,
    })
    svc.get_album = AsyncMock(return_value={
        "id": 1, "title": "东京一日", "date": "2026-07-07",
        "photo_count": 2, "created_at": None, "photos": [],
    })
    return svc


@pytest.fixture
def client(mock_album_service):
    from app.api.albums import create_albums_router

    app = FastAPI()
    router = create_albums_router(mock_album_service)
    app.include_router(router)

    # 覆盖数据库依赖
    from app.api import albums as albums_module
    albums_module.get_session_override = lambda: MagicMock()

    with TestClient(app) as c:
        yield c


class TestAlbumAPI:
    def test_upload_album(self, client):
        """验证上传相册接口"""
        files = [
            ("files", ("test1.jpg", io.BytesIO(b"fake image data"), "image/jpeg")),
        ]
        response = client.post(
            "/api/v1/albums/upload",
            data={"title": "东京一日", "date": "2026-07-07"},
            files=files,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["title"] == "东京一日"

    def test_list_albums(self, client):
        """验证获取相册列表"""
        response = client.get("/api/v1/albums?page=1&page_size=20")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_get_album(self, client):
        """验证获取相册详情"""
        response = client.get("/api/v1/albums/1")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["id"] == 1
