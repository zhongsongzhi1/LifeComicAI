import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置，从环境变量加载。"""

    def __init__(self):
        # 百度千帆
        self.QIANFAN_ACCESS_KEY: str = os.getenv("QIANFAN_ACCESS_KEY", "")
        self.QIANFAN_SECRET_KEY: str = os.getenv("QIANFAN_SECRET_KEY", "")

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

        # 服务配置
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))


@lru_cache()
def get_settings() -> Settings:
    """返回 Settings 单例。"""
    return Settings()
