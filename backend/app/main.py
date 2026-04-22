from __future__ import annotations

from fastapi import FastAPI

from app.agents.private_chef_agent import PrivateChefAgent
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.graph.workflow import ChefGraphFactory
from app.services.llm_service import LLMService
from app.services.tavily_service import TavilyService

setup_logging()
settings = get_settings()

llm_service = LLMService(settings)
tavily_service = TavilyService(settings)
graph = ChefGraphFactory(
    llm_service=llm_service,
    tavily_service=tavily_service,
    sqlite_path=settings.sqlite_checkpoint_path,
).create()
agent = PrivateChefAgent(graph)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
