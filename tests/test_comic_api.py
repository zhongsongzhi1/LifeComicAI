import os
import pytest
from unittest.mock import MagicMock, AsyncMock
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from app.db.database import get_session
from app.services.comic_service import ComicService


@pytest.fixture
def mock_comic_service():
    svc = MagicMock(spec=ComicService)
    svc.generate_comic = AsyncMock(return_value={
        "id": 1, "story_id": 1, "style": "miyazaki", "status": "completed",
        "total_pages": 2, "pdf_path": "/tmp/1/comic.pdf",
        "image_partial": False, "dialogue_partial": False, "created_at": None
    })
    svc.get_comic = AsyncMock(return_value={
        "id": 1, "story_id": 1, "style": "miyazaki", "status": "completed",
        "total_pages": 2, "pdf_path": "/tmp/1/comic.pdf",
        "image_partial": False, "dialogue_partial": False, "created_at": None
    })
    svc.get_comic_pages = AsyncMock(return_value={
        "story_id": 1, "comic_id": 1,
        "pages": [
            {"id": 1, "comic_id": 1, "page_num": 1, "shot_type": "Wide", "image_prompt": "", "image_url": "", "dialogue": "", "narration": "", "layout_json": None}
        ]
    })
    svc.get_comic_tasks = AsyncMock(return_value={
        "comic_id": 1, "status": "completed",
        "tasks": [{"id": 1, "comic_id": 1, "agent_name": "storyboard", "status": "done", "error_message": "", "retry_count": 0, "duration_ms": 1000}]
    })
    svc.list_styles = AsyncMock(return_value=[
        {"key": "miyazaki", "name": "宫崎骏风", "mood": "温馨", "dialogue_level": "少", "cinematic": True},
    ])
    return svc


@pytest.fixture
def client(mock_comic_service, session):
    """创建测试 HTTP 客户端，注入 mock ComicService 和 test session。"""
    from app.api.comics import create_comics_router

    test_app = FastAPI()
    router = create_comics_router(comic_service=mock_comic_service)
    test_app.include_router(router)

    # 覆盖 get_session
    async def override_get_session():
        yield session

    test_app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=test_app)
    client = AsyncClient(transport=transport, base_url="http://test")
    yield client


@pytest.mark.asyncio
async def test_generate_comic(client, mock_comic_service):
    """POST /api/v1/comics/generate 返回 200 和 comic_id。"""
    response = await client.post("/api/v1/comics/generate", json={"story_id": 1, "style_key": "miyazaki"})
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["comic_id"] == 1
    assert data["data"]["status"] == "processing"


@pytest.mark.asyncio
async def test_get_comic(client, mock_comic_service):
    """GET /api/v1/comics/1 返回 200。"""
    response = await client.get("/api/v1/comics/1")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["data"]["id"] == 1


@pytest.mark.asyncio
async def test_get_comic_not_found(client, mock_comic_service):
    """GET /api/v1/comics/99999 返回 404。"""
    mock_comic_service.get_comic = AsyncMock(return_value=None)
    response = await client.get("/api/v1/comics/99999")
    assert response.status_code == 200  # APIResponse 包装，不是 HTTP 404
    data = response.json()
    assert data["code"] == 404


@pytest.mark.asyncio
async def test_get_comic_pages(client, mock_comic_service):
    """GET /api/v1/comics/1/pages 返回 200。"""
    response = await client.get("/api/v1/comics/1/pages")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert len(data["data"]["pages"]) == 1


@pytest.mark.asyncio
async def test_get_single_page(client, mock_comic_service):
    """GET /api/v1/comics/1/pages/1 返回 200。"""
    response = await client.get("/api/v1/comics/1/pages/1")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0


@pytest.mark.asyncio
async def test_list_styles(client, mock_comic_service):
    """GET /api/v1/comics/styles 返回 200。"""
    response = await client.get("/api/v1/comics/styles")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert len(data["data"]) == 1


@pytest.mark.asyncio
async def test_get_tasks(client, mock_comic_service):
    """GET /api/v1/comics/1/tasks 返回 200。"""
    response = await client.get("/api/v1/comics/1/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert len(data["data"]["tasks"]) == 1


@pytest.mark.asyncio
async def test_download_pdf(tmp_path, client, mock_comic_service):
    """GET /api/v1/comics/1/download 返回 FileResponse。"""
    # 创建真实 PDF 文件
    pdf_dir = tmp_path / "1"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "comic.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

    mock_comic_service.get_comic = AsyncMock(return_value={
        "id": 1, "story_id": 1, "style": "miyazaki", "status": "completed",
        "total_pages": 2, "pdf_path": str(pdf_path),
        "image_partial": False, "dialogue_partial": False, "created_at": None
    })

    response = await client.get("/api/v1/comics/1/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


@pytest.mark.asyncio
async def test_download_not_ready(client, mock_comic_service):
    """无 pdf_path 时 GET /download 返回 404。"""
    mock_comic_service.get_comic = AsyncMock(return_value={
        "id": 1, "story_id": 1, "style": "miyazaki", "status": "processing",
        "total_pages": 0, "pdf_path": "",
        "image_partial": False, "dialogue_partial": False, "created_at": None
    })
    response = await client.get("/api/v1/comics/1/download")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 404


@pytest.mark.asyncio
async def test_invalid_style(client, mock_comic_service):
    """POST 无效 style_key 返回 400。"""
    mock_comic_service.generate_comic = AsyncMock(return_value=None)
    response = await client.post("/api/v1/comics/generate", json={"story_id": 1, "style_key": "nonexistent"})
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 400
