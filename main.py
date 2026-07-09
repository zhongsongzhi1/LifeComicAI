import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.api.albums import albums_router
from app.api.stories import stories_router
from app.api.characters import characters_router
from app.api.comics import comics_router
from app.api.health import health_router
from app.config.config import get_settings
from app.db.database import init_db

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="LifeOS Story Graph Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(albums_router)
app.include_router(stories_router)
app.include_router(characters_router)
app.include_router(comics_router)
app.include_router(health_router)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库。"""
    try:
        await init_db()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


if __name__ == "__main__":
    settings = get_settings()
    logger.info(f"Starting LifeOS Story Graph Engine on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
