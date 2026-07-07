# tests/test_common_schema.py
from app.schemas.common import APIResponse, PaginationParams, PaginatedResponse


class TestAPIResponse:
    def test_success_response(self):
        resp = APIResponse(data={"id": 1, "name": "test"})
        assert resp.code == 0
        assert resp.message == "success"
        assert resp.data == {"id": 1, "name": "test"}

    def test_error_response(self):
        resp = APIResponse(code=404, message="Not found")
        assert resp.code == 404
        assert resp.message == "Not found"
        assert resp.data is None

    def test_response_json_serialization(self):
        resp = APIResponse(data="hello")
        d = resp.model_dump()
        assert d == {"code": 0, "message": "success", "data": "hello"}


class TestPaginationParams:
    def test_default_values(self):
        p = PaginationParams()
        assert p.page == 1
        assert p.page_size == 20

    def test_custom_values(self):
        p = PaginationParams(page=3, page_size=10)
        assert p.page == 3
        assert p.page_size == 10


class TestPaginatedResponse:
    def test_paginated_response(self):
        resp = PaginatedResponse(
            items=[1, 2, 3],
            total=100,
            page=1,
            page_size=20,
        )
        d = resp.model_dump()
        assert d["items"] == [1, 2, 3]
        assert d["total"] == 100
        assert d["page"] == 1
        assert d["page_size"] == 20
        assert d["total_pages"] == 5
