from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.private_chef_agent import PrivateChefAgent
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.graph.workflow import ChefGraphFactory
from app.services.llm_service import LLMService
from app.services.tavily_service import TavilyService

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    llm_service = LLMService(settings)
    tavily_service = TavilyService(settings)

    db_path = Path(settings.sqlite_checkpoint_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)

    graph = ChefGraphFactory(
        llm_service=llm_service,
        tavily_service=tavily_service,
    ).create(checkpointer=saver)

    app.state.agent = PrivateChefAgent(graph)
    app.state.sqlite_conn = conn
    try:
        yield
    finally:
        conn.close()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI 私厨助手服务运行中", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
