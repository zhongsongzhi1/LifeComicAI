import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def mock_character_service():
    svc = MagicMock()
    svc.get_character = AsyncMock(return_value={
        "id": 1, "name": "我", "traits": ["长发"], "role": "",
        "avatar_url": "", "appeared_in_stories": [1, 2],
    })
    return svc


@pytest.fixture
def client(mock_character_service):
    from app.api.characters import create_characters_router

    app = FastAPI()
    router = create_characters_router(mock_character_service)
    app.include_router(router)

    from app.api import characters as characters_module
    characters_module.get_session_override = lambda: MagicMock()

    with TestClient(app) as c:
        yield c


class TestCharacterAPI:
    def test_get_character(self, client):
        """验证获取角色信息"""
        response = client.get("/api/v1/characters/1")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["name"] == "我"
        assert data["data"]["appeared_in_stories"] == [1, 2]

    def test_get_character_not_found(self, client, mock_character_service):
        """验证角色不存在返回 404"""
        mock_character_service.get_character.return_value = None
        response = client.get("/api/v1/characters/99999")
        assert response.status_code == 404
