from __future__ import annotations

from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    name: str
    freshness: int = Field(ge=0, le=100)
    remaining_ratio: float = Field(ge=0, le=1)
    note: str


class RecipeCandidate(BaseModel):
    title: str
    content: str
    url: str


class RankedRecipe(BaseModel):
    rank: int = Field(ge=1)
    name: str
    reason: str
    score: float = Field(ge=0, le=100)
    source_url: str


class RankRecipesResult(BaseModel):
    top3: list[RankedRecipe] = Field(default_factory=list)
    table_markdown: str = ""


class WeeklyPlanDay(BaseModel):
    day: str
    breakfast: str
    lunch: str
    dinner: str
    reason: str


class WeeklyPlanResult(BaseModel):
    weekly_plan: list[WeeklyPlanDay] = Field(default_factory=list)
    weekly_plan_markdown: str = ""
