from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.state import ChefState
from app.services.llm_service import LLMService
from app.services.tavily_service import TavilyService

logger = logging.getLogger(__name__)


class ChefGraphFactory:
    def __init__(self, llm_service: LLMService, tavily_service: TavilyService, sqlite_path: str) -> None:
        self.llm_service = llm_service
        self.tavily_service = tavily_service
        self.sqlite_path = sqlite_path

    def create(self):
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

        db_path = Path(self.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # langgraph-checkpoint-sqlite>=2.x changed `from_conn_string` to return
        # a context manager. `graph.compile` requires a concrete
        # `BaseCheckpointSaver` instance, so we create the saver directly from
        # a long-lived sqlite connection.
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        return graph.compile(checkpointer=saver)

    def _route_intent(self, state: ChefState) -> str:
        intent = state.get("intent", "analyze")
        if intent == "analyze":
            return "recognize"
        if intent == "followup":
            return "followup"
        return "weekly"

    def _recognize_node(self, state: ChefState) -> ChefState:
        image_url = state["input_image_url"]
        ingredients = self.llm_service.recognize_ingredients(image_url)
        logger.info("recognized %s ingredients", len(ingredients))
        return {
            "ingredients": ingredients,
            "step": "ingredients_recognized",
            "messages": [HumanMessage(content=f"识别食材: {ingredients}")],
        }

    def _search_node(self, state: ChefState) -> ChefState:
        candidates = self.tavily_service.search_recipes(state.get("ingredients", []))
        logger.info("found %s recipe candidates", len(candidates))
        return {"recipes": candidates, "step": "recipes_searched"}

    def _rank_node(self, state: ChefState) -> ChefState:
        ranked = self.llm_service.rank_recipes(
            ingredients=state.get("ingredients", []),
            candidates=state.get("recipes", []),
        )
        recipes = ranked.get("top3", [])
        return {
            "recipes": recipes,
            "table_markdown": ranked.get("table_markdown", ""),
            "selected_index": 0 if recipes else None,
            "step": "recipes_ranked",
            "messages": [AIMessage(content="已生成 Top3 菜谱")],
        }

    def _followup_node(self, state: ChefState) -> ChefState:
        context = {
            "ingredients": state.get("ingredients", []),
            "recipes": state.get("recipes", []),
            "selected_index": state.get("selected_index"),
            "step": state.get("step"),
        }
        answer = self.llm_service.followup(context=context, question=state.get("question", ""))
        return {
            "answer": answer,
            "step": "followup_answered",
            "messages": [HumanMessage(content=state.get("question", "")), AIMessage(content=answer)],
        }

    def _weekly_node(self, state: ChefState) -> ChefState:
        weekly_plan = self.llm_service.weekly_plan(state.get("history_text", ""))
        return {
            "weekly_plan_markdown": weekly_plan,
            "step": "weekly_plan_generated",
            "messages": [AIMessage(content="已生成周计划")],
        }
