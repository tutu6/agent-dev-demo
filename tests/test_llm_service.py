from __future__ import annotations

from app.core.config import Settings
from app.domain.models import Ingredient, RecipeCandidate
from app.services.llm_service import LLMService


def test_prepare_candidates_for_rerank_dedup_sort_and_truncate():
    service = LLMService(
        Settings(
            rerank_candidate_limit=2,
            rerank_content_max_chars=20,
        )
    )
    ingredients = [
        Ingredient(name="鸡蛋", freshness=90, remaining_ratio=0.6, note="ok"),
        Ingredient(name="番茄", freshness=80, remaining_ratio=0.4, note="ok"),
    ]
    candidates = [
        RecipeCandidate(title="无关网页", content="今天新闻", url="https://x.com"),
        RecipeCandidate(title="番茄炒蛋做法", content="番茄和鸡蛋都要有", url="https://a.com"),
        RecipeCandidate(title="番茄炒蛋重复", content="重复链接", url="https://a.com"),
        RecipeCandidate(title="鸡蛋汤", content="鸡蛋和水即可" * 20, url="https://b.com"),
    ]

    compacted = service._prepare_candidates_for_rerank(ingredients, candidates)

    assert len(compacted) == 2
    assert compacted[0]["url"] == "https://a.com"
    assert compacted[1]["url"] == "https://b.com"
    assert len(compacted[1]["content"]) == 20
