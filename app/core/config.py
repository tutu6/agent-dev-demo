from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI 私厨助手"
    app_version: str = "0.1.0"

    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    qwen_chat_model: str = "qwen-plus"
    qwen_vision_model: str = "qwen-vl-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_chat_temperature: float = 0.2
    qwen_vision_temperature: float = 0.1

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    tavily_max_results: int = 8

    sqlite_checkpoint_path: str = "./checkpoints/chef_graph.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
