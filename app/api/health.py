from fastapi import APIRouter

from app.schemas.common import APIResponse

health_router = APIRouter(tags=["health"])


@health_router.get("/api/v1/health", response_model=APIResponse[dict])
async def health_check():
    return APIResponse(data={"status": "ok"})
