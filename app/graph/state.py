from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage

from app.domain.models import Ingredient, RankedRecipe, WeeklyPlanDay


class ChefState(TypedDict, total=False):
    # Messages: conversation context managed by LangGraph checkpoint
    messages: list[BaseMessage]

    # Structured State
    thread_id: str
    input_image_url: str
    intent: Literal["analyze", "followup", "weekly_plan"]
    question: str
    history_text: str

    ingredients: list[Ingredient]
    recipes: list[RankedRecipe]
    selected_index: int | None
    step: str
    table_markdown: str
    answer: str
    weekly_plan: list[WeeklyPlanDay]
    weekly_plan_markdown: str
