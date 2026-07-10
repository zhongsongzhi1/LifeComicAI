import logging
import os
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
from app.api.strips import strips_router
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
app.include_router(strips_router)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库和种子数据。"""
    try:
        await init_db()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        try:
            from app.db.database import engine
            from sqlalchemy import text
            async with engine.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS quick_strips (
                        id SERIAL PRIMARY KEY,
                        title VARCHAR(255) NOT NULL DEFAULT '',
                        status VARCHAR(20) NOT NULL DEFAULT 'processing',
                        image_path VARCHAR(500) NOT NULL DEFAULT '',
                        error_msg TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now()
                    )
                """))
            logger.info("quick_strips table created (fallback)")
        except Exception as e2:
            logger.error(f"quick_strips fallback creation failed: {e2}")

    try:
        settings = get_settings()
        os.makedirs(settings.COMICS_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.COMICS_DIR, "strips"), exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Directory creation failed: {e}")

    # 种子风格预设数据
    try:
        from app.db.database import async_session_factory
        from app.db.seed_prompts import seed_prompt_versions

        async with async_session_factory() as session:
            await seed_prompt_versions(session)
            await session.commit()
        logger.info("Style presets seeded successfully")
    except Exception as e:
        logger.error(f"Style presets seeding failed: {e}")


if __name__ == "__main__":
    settings = get_settings()
    logger.info(f"Starting LifeOS Story Graph Engine on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
