import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """创建测试客户端，Mock 数据库依赖。"""
    app.dependency_overrides = {}

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}


class TestEndToEnd:
    def test_health_check(self, client):
        """端到端：健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "ok"

    def test_app_title(self):
        """验证应用元信息"""
        assert app.title == "LifeOS Story Graph Engine"
        assert app.version == "0.1.0"
