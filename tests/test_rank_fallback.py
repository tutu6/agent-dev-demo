from __future__ import annotations

from app.domain.models import Ingredient, RecipeCandidate
from app.services.llm_service import LLMService


def test_extract_recipe_name_prefers_semantic_dish_name():
    title = "家常菜做法大全｜番茄炒蛋怎么做_下厨房"
    content = "番茄炒蛋的做法步骤：先炒鸡蛋再下番茄。"
    assert LLMService._extract_recipe_name(title, content) == "番茄炒蛋"


def test_heuristic_rank_recipes_uses_recipe_name_not_page_title():
    ingredients = [
        Ingredient(name="番茄", freshness=90, remaining_ratio=0.7, note="新鲜"),
        Ingredient(name="鸡蛋", freshness=85, remaining_ratio=0.8, note="可用"),
    ]
    candidates = [
        RecipeCandidate(
            title="家常菜谱｜番茄炒蛋的做法_某美食站",
            content="番茄炒蛋，酸甜开胃，步骤简单。",
            url="https://a.com",
        ),
        RecipeCandidate(
            title="十大快手菜排行榜",
            content="鸡蛋饼做法：鸡蛋打散后煎至两面金黄。",
            url="https://b.com",
        ),
    ]

    result = LLMService._heuristic_rank_recipes(ingredients, candidates)
    assert result.top3[0].name == "番茄炒蛋"
    assert result.top3[0].name != candidates[0].title


def test_extract_recipe_name_avoids_marketing_title_fragment():
    title = "一周鲜榨果蔬汁!低卡营养不重样"
    content = "鲜榨果蔬汁做法：苹果、芹菜和黄瓜一起榨汁即可。"
    assert LLMService._extract_recipe_name(title, content) == "鲜榨果蔬汁"
