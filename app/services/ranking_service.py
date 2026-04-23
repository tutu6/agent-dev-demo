from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from app.domain.models import Ingredient, RankedRecipe, RecipeCandidate


class RankingWeights(BaseModel):
    nutrition: float = Field(default=0.3, ge=0)
    match: float = Field(default=0.4, ge=0)
    simplicity: float = Field(default=0.3, ge=0)

    @model_validator(mode="after")
    def validate_total_weight(self) -> "RankingWeights":
        total = self.nutrition + self.match + self.simplicity
        if abs(total - 1.0) > 1e-6:
            raise ValueError("ranking weights must sum to 1.0")
        return self


class RuleBasedRecipeRanker:
    _HEALTHY_KEYWORDS = ("蒸", "煮", "炖", "烤", "清炒", "沙拉", "少油", "高蛋白", "低脂", "蔬菜")
    _HEAVY_KEYWORDS = ("炸", "油炸", "红烧", "奶油", "黄油", "糖", "肥肉")
    _SIMPLE_KEYWORDS = ("快手", "简单", "家常", "10分钟", "15分钟", "一锅", "三步")
    _COMPLEX_KEYWORDS = ("腌制", "发酵", "裹粉", "勾芡", "复炸", "慢炖", "高压锅")

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self._weights = weights or RankingWeights()

    def rank(self, ingredients: list[Ingredient], candidates: list[RecipeCandidate], top_k: int = 3) -> list[RankedRecipe]:
        ranked: list[RankedRecipe] = []
        ingredient_names = [item.name for item in ingredients]
        for candidate in candidates:
            full_text = f"{candidate.title} {candidate.content}".lower()
            nutrition_score = self._calc_nutrition_score(full_text)
            match_score = self._calc_match_score(full_text, ingredient_names)
            simplicity_score = self._calc_simplicity_score(full_text)
            final_score = (
                nutrition_score * self._weights.nutrition
                + match_score * self._weights.match
                + simplicity_score * self._weights.simplicity
            )
            ranked.append(
                RankedRecipe(
                    rank=1,
                    name=self._extract_recipe_name(candidate.title),
                    reason=(
                        f"食材匹配度 {match_score:.1f}、健康度 {nutrition_score:.1f}、简易度 {simplicity_score:.1f}，"
                        f"综合得分 {final_score:.1f}"
                    ),
                    score=round(final_score, 1),
                    source_url=candidate.url,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        top_recipes = ranked[:top_k]
        for idx, item in enumerate(top_recipes, start=1):
            item.rank = idx
        return top_recipes

    @staticmethod
    def build_table_markdown(recipes: list[RankedRecipe]) -> str:
        header = "| 排名 | 菜名 | 评分 |\n|---|---|---|\n"
        rows = "".join(f"| {item.rank} | {item.name} | {item.score:.1f} |\n" for item in recipes)
        return header + rows

    def _calc_match_score(self, text: str, ingredient_names: list[str]) -> float:
        if not ingredient_names:
            return 0.0
        hit = sum(1 for name in ingredient_names if name.lower() in text)
        return min(100.0, hit / len(ingredient_names) * 100)

    def _calc_nutrition_score(self, text: str) -> float:
        healthy_hit = sum(1 for item in self._HEALTHY_KEYWORDS if item in text)
        heavy_hit = sum(1 for item in self._HEAVY_KEYWORDS if item in text)
        return max(0.0, min(100.0, 70 + healthy_hit * 8 - heavy_hit * 10))

    def _calc_simplicity_score(self, text: str) -> float:
        simple_hit = sum(1 for item in self._SIMPLE_KEYWORDS if item in text)
        complex_hit = sum(1 for item in self._COMPLEX_KEYWORDS if item in text)
        step_count = self._estimate_step_count(text)
        return max(0.0, min(100.0, 75 + simple_hit * 8 - complex_hit * 8 - max(0, step_count - 5) * 4))

    @staticmethod
    def _estimate_step_count(text: str) -> int:
        markers = re.findall(r"(步骤\s*\d+|第\s*\d+\s*步|\d+\.)", text)
        if markers:
            return len(markers)
        return text.count("。") + text.count("\n")

    @staticmethod
    def _extract_recipe_name(title: str) -> str:
        return title.split("_")[0].split("-")[0].strip() or "推荐菜谱"
