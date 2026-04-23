from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.domain.models import Ingredient, RankRecipesResult, RankedRecipe
from app.graph.state import ChefState
from app.services.llm_service import LLMService
from app.services.tavily_service import TavilyService

logger = logging.getLogger(__name__)


class ChefGraphFactory:
    def __init__(self, llm_service: LLMService, tavily_service: TavilyService) -> None:
        self.llm_service = llm_service
        self.tavily_service = tavily_service

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
        ingredients = [Ingredient.model_validate(item) for item in self.llm_service.recognize_ingredients(image_url)]
        logger.info("recognized %s ingredients", len(ingredients))
        return {
            "ingredients": ingredients,
            "step": "ingredients_recognized",
            "messages": [HumanMessage(content=f"识别食材: {[item.model_dump() for item in ingredients]}")],
        }

    def _search_node(self, state: ChefState) -> ChefState:
        ingredients = [Ingredient.model_validate(item) for item in state.get("ingredients", [])]
        candidates = self.tavily_service.search_recipes(ingredients)
        logger.info("found %s recipe candidates", len(candidates))
        return {"recipes": candidates, "step": "recipes_searched"}

    def _rank_node(self, state: ChefState) -> ChefState:
        ingredients = [Ingredient.model_validate(item) for item in state.get("ingredients", [])]
        candidates = state.get("recipes", [])
        ranked_result = RankRecipesResult.model_validate(self.llm_service.rank_recipes(ingredients=ingredients, candidates=candidates))
        recipes = [RankedRecipe.model_validate(item) for item in ranked_result.top3]
        return {
            "recipes": recipes,
            "table_markdown": ranked_result.table_markdown,
            "selected_index": 0 if recipes else None,
            "step": "recipes_ranked",
            "messages": [AIMessage(content="已生成 Top3 菜谱")],
        }

    def _followup_node(self, state: ChefState) -> ChefState:
        context = {
            "ingredients": [Ingredient.model_validate(item).model_dump() for item in state.get("ingredients", [])],
            "recipes": [RankedRecipe.model_validate(item).model_dump() for item in state.get("recipes", [])],
            "selected_index": state.get("selected_index"),
            "step": state.get("step"),
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
        return {
            "weekly_plan": weekly_plan.weekly_plan if hasattr(weekly_plan, "weekly_plan") else weekly_plan.get("weekly_plan", []),
            "weekly_plan_markdown": weekly_plan.weekly_plan_markdown if hasattr(weekly_plan, "weekly_plan_markdown") else weekly_plan.get("weekly_plan_markdown", ""),
            "step": "weekly_plan_generated",
            "messages": [AIMessage(content="已生成周计划")],
        }
