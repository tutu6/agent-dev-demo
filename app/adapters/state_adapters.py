from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.models import Ingredient, RankedRecipe, RecipeCandidate, WeeklyPlanDay
from app.schemas.responses import RecipeItem


class RankStateIn(BaseModel):
    ingredients: list[Ingredient] = Field(default_factory=list)
    candidates: list[RecipeCandidate] = Field(default_factory=list)


class FollowupStateIn(BaseModel):
    ingredients: list[Ingredient] = Field(default_factory=list)
    recipes: list[RankedRecipe] = Field(default_factory=list)
    selected_index: int | None = None
    step: str | None = None


class AnalyzeStateOut(BaseModel):
    ingredients: list[Ingredient] = Field(default_factory=list)
    recipes: list[RecipeItem] = Field(default_factory=list)
    table_markdown: str = ""


class WeeklyPlanStateOut(BaseModel):
    weekly_plan: list[WeeklyPlanDay] = Field(default_factory=list)
    weekly_plan_markdown: str = ""


def to_rank_input(state: dict[str, Any]) -> RankStateIn:
    return RankStateIn(
        ingredients=[Ingredient.model_validate(item) for item in state.get("ingredients", [])],
        candidates=[RecipeCandidate.model_validate(item) for item in state.get("recipes", [])],
    )


def to_followup_input(state: dict[str, Any]) -> FollowupStateIn:
    return FollowupStateIn(
        ingredients=[Ingredient.model_validate(item) for item in state.get("ingredients", [])],
        recipes=[RankedRecipe.model_validate(item) for item in state.get("recipes", [])],
        selected_index=state.get("selected_index"),
        step=state.get("step"),
    )


def to_analyze_response_output(state: dict[str, Any]) -> AnalyzeStateOut:
    recipes = [
        RecipeItem.model_validate(item.model_dump() if hasattr(item, "model_dump") else item)
        for item in state.get("recipes", [])
    ]
    return AnalyzeStateOut(
        ingredients=[Ingredient.model_validate(item) for item in state.get("ingredients", [])],
        recipes=recipes,
        table_markdown=state.get("table_markdown", ""),
    )


def to_weekly_plan_response_output(state: dict[str, Any]) -> WeeklyPlanStateOut:
    return WeeklyPlanStateOut(
        weekly_plan=[WeeklyPlanDay.model_validate(item) for item in state.get("weekly_plan", [])],
        weekly_plan_markdown=state.get("weekly_plan_markdown", ""),
    )


def serialize_model(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
    return value
