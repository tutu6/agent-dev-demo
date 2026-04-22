from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RecipeItem(BaseModel):
    rank: int
    name: str
    reason: str
    score: float
    source_url: str


class AnalyzeResponse(BaseModel):
    thread_id: str
    ingredients: list[dict[str, Any]]
    recipes: list[RecipeItem]
    table_markdown: str


class FollowupResponse(BaseModel):
    thread_id: str
    answer: str
    selected_index: int | None


class WeeklyPlanResponse(BaseModel):
    thread_id: str
    weekly_plan_markdown: str


class HistoryResponse(BaseModel):
    thread_id: str
    state: dict[str, Any]
