from __future__ import annotations

import pytest

from app.domain.models import Ingredient, RecipeCandidate
from app.services.ranking_service import RankingWeights, RuleBasedRecipeRanker


def test_ranking_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        RankingWeights(nutrition=0.5, match=0.5, simplicity=0.2)


def test_rule_based_ranker_returns_sorted_top_k():
    ranker = RuleBasedRecipeRanker()
    ingredients = [
        Ingredient(name="番茄", freshness=90, remaining_ratio=0.6, note="新鲜"),
        Ingredient(name="鸡蛋", freshness=88, remaining_ratio=0.7, note="可用"),
    ]
    candidates = [
        RecipeCandidate(title="番茄炒蛋", content="简单家常，步骤1. 步骤2.", url="https://a.com"),
        RecipeCandidate(title="油炸鸡蛋卷", content="油炸做法，步骤很多", url="https://b.com"),
        RecipeCandidate(title="番茄蛋花汤", content="煮汤，快手，三步完成", url="https://c.com"),
    ]

    top2 = ranker.rank(ingredients, candidates, top_k=2)

    assert len(top2) == 2
    assert top2[0].score >= top2[1].score
    assert top2[0].rank == 1
    assert top2[1].rank == 2
    assert top2[0].source_url in {"https://a.com", "https://c.com"}
