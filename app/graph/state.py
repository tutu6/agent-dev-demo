from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


class ChefState(TypedDict, total=False):
    # Messages: conversation context managed by LangGraph checkpoint
    messages: list[BaseMessage]

    # Structured State - 使用 dict 存储，避免 LangGraph checkpoint 序列化问题
    thread_id: str
    input_image_url: str
    intent: Literal["analyze", "followup", "weekly_plan"]
    question: str
    history_text: str

    ingredients: list[dict[str, Any]]  # 存储 Ingredient.model_dump()
    recipes: list[dict[str, Any]]  # 存储 RankedRecipe.model_dump() 或 RecipeCandidate.model_dump()
    selected_index: int | None
    step: str
    table_markdown: str
    answer: str
    weekly_plan: list[dict[str, Any]]  # 存储 WeeklyPlanDay.model_dump()
    weekly_plan_markdown: str
