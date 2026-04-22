from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from app.core.config import Settings


class TavilyService:
    """Search recipe candidates via Tavily API."""

    def __init__(self, settings: Settings) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    def search_recipes(self, ingredients: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ingredient_names = ",".join(item["name"] for item in ingredients)
        query = f"家常菜 菜谱 做法 食材: {ingredient_names}"
        result = self._client.search(
            query=query,
            search_depth="basic",
            max_results=8,
            include_answer=False,
        )
        candidates: list[dict[str, Any]] = []
        for item in result.get("results", []):
            candidates.append(
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                }
            )
        return candidates
