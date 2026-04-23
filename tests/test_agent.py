from __future__ import annotations

import base64
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.private_chef_agent import PrivateChefAgent
from app.graph.workflow import ChefGraphFactory


class DummyLLM:
    def recognize_ingredients(self, image_url: str):
        return [
            {"name": "番茄", "freshness": 90, "remaining_ratio": 0.6, "note": "新鲜"},
            {"name": "鸡蛋", "freshness": 85, "remaining_ratio": 0.7, "note": "可用"},
        ]

    def rank_recipes(self, ingredients, candidates):
        return {
            "top3": [
                {"rank": 1, "name": "番茄炒蛋", "reason": "高匹配", "score": 92, "source_url": "https://a.com"},
                {"rank": 2, "name": "西红柿蛋汤", "reason": "快手", "score": 87, "source_url": "https://b.com"},
                {"rank": 3, "name": "蛋饼", "reason": "简单", "score": 80, "source_url": "https://c.com"},
            ],
            "table_markdown": "|排名|菜名|\n|---|---|\n|1|番茄炒蛋|",
        }

    def followup(self, context, question: str):
        return f"回答: {question}"

    def weekly_plan(self, history_text: str):
        return {
            "weekly_plan": [
                {"day": "周一", "breakfast": "燕麦", "lunch": "鸡胸肉沙拉", "dinner": "蔬菜面", "reason": "控油"},
                {"day": "周二", "breakfast": "鸡蛋", "lunch": "糙米饭", "dinner": "鱼汤", "reason": "高蛋白"},
            ],
            "weekly_plan_markdown": "| 周一 | 周二 |\n|---|---|\n| 燕麦 | 鸡蛋 |",
        }


class DummyTavily:
    def search_recipes(self, ingredients):
        return [{"title": "番茄炒蛋", "content": "步骤", "url": "https://a.com"}]


def build_agent(tmp_path: Path) -> PrivateChefAgent:
    conn = sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False)
    graph = ChefGraphFactory(DummyLLM(), DummyTavily()).create(checkpointer=SqliteSaver(conn))
    return PrivateChefAgent(graph)


def test_graph_analyze_followup_and_weekly_plan(tmp_path: Path):
    agent = build_agent(tmp_path)

    image_b64 = base64.b64encode(b"fakejpg").decode("utf-8")
    analyzed = agent.analyze_by_upload("t-1", image_b64)
    assert analyzed["step"] == "recipes_ranked"
    assert len(analyzed["ingredients"]) == 2
    assert len(analyzed["recipes"]) == 3

    followup = agent.followup("t-1", "第一个菜怎么做")
    assert followup["step"] == "followup_answered"
    assert "第一个菜怎么做" in followup["answer"]

    weekly = agent.weekly_plan("t-1", "工作日常吃外卖")
    assert weekly["step"] == "weekly_plan_generated"
    assert len(weekly["weekly_plan"]) >= 1
    assert "周一" in weekly["weekly_plan_markdown"]

    history = agent.get_history("t-1")
    assert history["thread_id"] == "t-1"


def test_invalid_base64(tmp_path: Path):
    agent = build_agent(tmp_path)

    try:
        agent.analyze_by_upload("t-2", "not-valid-base64")
        assert False, "should raise"
    except ValueError as exc:
        assert "invalid base64" in str(exc)
