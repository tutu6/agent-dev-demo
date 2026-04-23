from __future__ import annotations

from tavily import TavilyClient

from app.core.config import Settings
from app.domain.models import Ingredient, RecipeCandidate


class TavilyService:
    """Search recipe candidates via Tavily API."""

    def __init__(self, settings: Settings) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._max_results = settings.tavily_max_results

    def search_recipes(self, ingredients: list[Ingredient]) -> list[RecipeCandidate]:
        ingredient_names = ",".join(item.name for item in ingredients)
        query = f"家常菜 菜谱 做法 食材: {ingredient_names}"
        result = self._client.search(
            query=query,
            search_depth="basic",
            max_results=self._max_results,
            include_answer=False,
        )
        candidates: list[RecipeCandidate] = []
        for item in result.get("results", []):
            candidates.append(
                RecipeCandidate(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    url=item.get("url", ""),
                )
            )
        return candidates
