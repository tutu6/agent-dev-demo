from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.adapters import to_followup_input, to_rank_input
from app.domain.models import Ingredient, RankRecipesResult, RankedRecipe, RecipeCandidate
from app.graph.state import ChefState
from app.services.llm_service import LLMService
from app.services.ranking_service import RuleBasedRecipeRanker
from app.services.tavily_service import TavilyService

logger = logging.getLogger(__name__)


class ChefGraphFactory:
    def __init__(
        self,
        llm_service: LLMService,
        tavily_service: TavilyService,
        recipe_ranker: RuleBasedRecipeRanker | None = None,
        enable_llm_rerank: bool = False,
        llm_rerank_pool_size: int = 5,
        final_top_k: int = 3,
    ) -> None:
        self.llm_service = llm_service
        self.tavily_service = tavily_service
        self.recipe_ranker = recipe_ranker or RuleBasedRecipeRanker()
        self.enable_llm_rerank = enable_llm_rerank
        self.llm_rerank_pool_size = llm_rerank_pool_size
        self.final_top_k = final_top_k

    def create(self, checkpointer: Any):
        graph = StateGraph(ChefState)
        graph.add_node("recognize", self._recognize_node)
        graph.add_node("search", self._search_node)
        graph.add_node("rank", self._rank_node)
        graph.add_node("followup", self._followup_node)
        graph.add_node("weekly", self._weekly_node)

        graph.add_conditional_edges(START, self._route_intent)
        graph.add_edge("recognize", "search")
        graph.add_edge("search", "rank")
        graph.add_edge("rank", END)
        graph.add_edge("followup", END)
        graph.add_edge("weekly", END)

        return graph.compile(checkpointer=checkpointer)

    def _route_intent(self, state: ChefState) -> str:
        intent = state.get("intent", "analyze")
        if intent == "analyze":
            return "recognize"
        if intent == "followup":
            return "followup"
        return "weekly"

    def _recognize_node(self, state: ChefState) -> ChefState:
        image_url = state["input_image_url"]
        raw_ingredients = self.llm_service.recognize_ingredients(image_url)
        ingredients = [Ingredient.model_validate(item) for item in raw_ingredients]
        logger.info("recognized %s ingredients", len(ingredients))
        return {
            "ingredients": [item.model_dump() for item in ingredients],  # 转为 dict
            "step": "ingredients_recognized",
            "messages": [HumanMessage(content=f"识别食材: {[item.model_dump() for item in ingredients]}")],
        }

    def _search_node(self, state: ChefState) -> ChefState:
        ingredients = [Ingredient.model_validate(item) for item in state.get("ingredients", [])]
        raw_candidates = self.tavily_service.search_recipes(ingredients)
        candidates = [RecipeCandidate.model_validate(item) for item in raw_candidates]
        logger.info("found %s recipe candidates", len(candidates))
        return {"recipes": [item.model_dump() for item in candidates], "step": "recipes_searched"}  # 转为 dict

    def _rank_node(self, state: ChefState) -> ChefState:
        rank_input = to_rank_input(state)
        pre_ranked = self.recipe_ranker.rank(
            rank_input.ingredients,
            rank_input.candidates,
            top_k=max(self.llm_rerank_pool_size, self.final_top_k),
        )
        ranked = self._maybe_llm_rerank(
            ingredients=rank_input.ingredients,
            candidates=rank_input.candidates,
            pre_ranked=pre_ranked,
        )[: self.final_top_k]
        recipes = [item.model_dump() for item in ranked]  # 转为 dict
        return {
            "recipes": recipes,
            "table_markdown": self.recipe_ranker.build_table_markdown(ranked),
            "selected_index": 0 if recipes else None,
            "step": "recipes_ranked",
            "messages": [AIMessage(content="已生成 Top3 菜谱")],
        }

    def _maybe_llm_rerank(
        self,
        ingredients: list[Ingredient],
        candidates: list[RecipeCandidate],
        pre_ranked: list[RankedRecipe],
    ) -> list[RankedRecipe]:
        if not self.enable_llm_rerank:
            return pre_ranked
        if len(pre_ranked) <= 1:
            return pre_ranked

        candidate_by_url = {item.url: item for item in candidates}
        rerank_candidates: list[RecipeCandidate] = []
        for item in pre_ranked[: self.llm_rerank_pool_size]:
            candidate = candidate_by_url.get(item.source_url)
            if candidate:
                rerank_candidates.append(candidate)

        if len(rerank_candidates) <= 1:
            return pre_ranked

        try:
            llm_ranked = RankRecipesResult.model_validate(
                self.llm_service.rank_recipes(
                    ingredients=ingredients,
                    candidates=rerank_candidates,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("llm rerank failed, fallback to rule-based result")
            return pre_ranked

        if not llm_ranked.top3:
            return pre_ranked
        return llm_ranked.top3

    def _followup_node(self, state: ChefState) -> ChefState:
        followup_input = to_followup_input(state)
        context = {
            "ingredients": [item.model_dump() for item in followup_input.ingredients],
            "recipes": [item.model_dump() for item in followup_input.recipes],
            "selected_index": followup_input.selected_index,
            "step": followup_input.step,
        }
        question = state.get("question", "")
        answer = self.llm_service.followup(context=context, question=question)
        return {
            "answer": answer,
            "step": "followup_answered",
            "messages": [HumanMessage(content=question), AIMessage(content=answer)],
        }

    def _weekly_node(self, state: ChefState) -> ChefState:
        weekly_plan = self.llm_service.weekly_plan(state.get("history_text", ""))
        plan_data = weekly_plan.weekly_plan if hasattr(weekly_plan, "weekly_plan") else weekly_plan.get("weekly_plan", [])
        return {
            "weekly_plan": [item.model_dump() if hasattr(item, "model_dump") else item for item in plan_data],  # 转为 dict
            "weekly_plan_markdown": weekly_plan.weekly_plan_markdown if hasattr(weekly_plan, "weekly_plan_markdown") else weekly_plan.get("weekly_plan_markdown", ""),
            "step": "weekly_plan_generated",
            "messages": [AIMessage(content="已生成周计划")],
        }
