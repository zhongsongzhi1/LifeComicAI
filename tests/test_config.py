import os
import pytest


class TestSettings:
    def test_settings_loads_from_env(self, monkeypatch):
        """验证 Settings 从环境变量正确加载配置"""
        monkeypatch.setenv("QIANFAN_ACCESS_KEY", "test_ak")
        monkeypatch.setenv("QIANFAN_SECRET_KEY", "test_sk")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.api.com/v1")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/testdb")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("UPLOAD_DIR", "./test_uploads")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        from app.config.config import Settings

        s = Settings()

        assert s.QIANFAN_ACCESS_KEY == "test_ak"
        assert s.QIANFAN_SECRET_KEY == "test_sk"
        assert s.OPENAI_API_KEY == "test_openai_key"
        assert s.OPENAI_BASE_URL == "https://test.api.com/v1"
        assert s.OPENAI_MODEL_NAME == "gpt-4o"
        assert s.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        assert s.REDIS_URL == "redis://localhost:6379/0"
        assert s.UPLOAD_DIR == "./test_uploads"
        assert s.OLLAMA_BASE_URL == "http://localhost:11434"
        assert s.EMBEDDING_MODEL == "bge-large-zh-v1.5"

    def test_settings_defaults(self, monkeypatch):
        """验证默认值"""
        monkeypatch.setenv("QIANFAN_ACCESS_KEY", "ak")
        monkeypatch.setenv("QIANFAN_SECRET_KEY", "sk")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.com/v1")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        from app.config.config import Settings

        s = Settings()

        assert s.OPENAI_MODEL_NAME == "gpt-4o"
        assert s.UPLOAD_DIR == "./uploads"
        assert s.EMBEDDING_MODEL == "bge-large-zh-v1.5"
        assert s.HOST == "0.0.0.0"
        assert s.PORT == 8000

    def test_get_settings_returns_singleton(self, monkeypatch):
        """验证 get_settings 返回单例"""
        monkeypatch.setenv("QIANFAN_ACCESS_KEY", "ak")
        monkeypatch.setenv("QIANFAN_SECRET_KEY", "sk")
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.com/v1")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        from app.config.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
