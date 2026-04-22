from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


class ChefState(TypedDict, total=False):
    # Messages: conversation context managed by LangGraph checkpoint
    messages: list[BaseMessage]

    # Structured State
    thread_id: str
    input_image_url: str
    intent: Literal["analyze", "followup", "weekly_plan"]
    question: str
    history_text: str

    ingredients: list[dict[str, Any]]
    recipes: list[dict[str, Any]]
    selected_index: int | None
    step: str
    table_markdown: str
    answer: str
    weekly_plan_markdown: str
