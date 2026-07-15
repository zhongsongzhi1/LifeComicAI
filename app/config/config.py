import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    """应用配置，从环境变量加载。"""

    def __init__(self):
        # 百度千帆
        self.QIANFAN_ACCESS_KEY: str = os.getenv("QIANFAN_ACCESS_KEY", "")
        self.QIANFAN_SECRET_KEY: str = os.getenv("QIANFAN_SECRET_KEY", "")

        # ModelScope 文生图
        self.MODELSCOPE_API_KEY: str = os.getenv("MODELSCOPE_API_KEY", "")
        self.MODELSCOPE_IMAGE_MODEL: str = os.getenv("MODELSCOPE_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")

        # OpenAI 兼容接口
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))

        # 数据库
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", "")

        # Redis
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Ollama Embedding
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-large-zh-v1.5")

        # 图片上传路径
        self.UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

        # 漫画目录
        self.COMICS_DIR: str = os.getenv("COMICS_DIR", "./comics")

        # 服务配置
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))

        # CORS 配置
        cors_origins = os.getenv("CORS_ORIGINS", "")
        self.CORS_ORIGINS: list = [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else []

        self._validate()


    def _validate(self):
        required_keys = ["QIANFAN_ACCESS_KEY", "QIANFAN_SECRET_KEY", "MODELSCOPE_API_KEY", "DATABASE_URL"]
        missing = []
        for key in required_keys:
            value = getattr(self, key, "")
            if not value:
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def __repr__(self):
        return f"Settings(QIANFAN_ACCESS_KEY={'***' if self.QIANFAN_ACCESS_KEY else ''}, MODELSCOPE_API_KEY={'***' if self.MODELSCOPE_API_KEY else ''}, OPENAI_API_KEY={'***' if self.OPENAI_API_KEY else ''})"


@lru_cache()
def get_settings() -> Settings:
    """返回 Settings 单例。"""
    return Settings()
