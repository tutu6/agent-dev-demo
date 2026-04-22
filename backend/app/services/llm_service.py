from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """Wraps Qwen chat and vision models using LangChain."""

    def __init__(self, settings: Settings) -> None:
        self._chat = ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.2,
        )
        self._vision = ChatOpenAI(
            model=settings.qwen_vision_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.1,
        )

    def recognize_ingredients(self, image_url: str) -> list[dict[str, Any]]:
        prompt = (
            "识别图片中的食材。输出 JSON 数组，每个元素必须包含: "
            "name, freshness(0-100), remaining_ratio(0-1), note。不要输出额外文本。"
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        response = self._vision.invoke([message])
        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse ingredients JSON: {e}")
            logger.error(f"Response content: {response.content}")
            return []

    def rank_recipes(self, ingredients: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        system = SystemMessage(content="你是严谨的私厨助手。只输出 JSON。")
        user = HumanMessage(
            content=(
                "根据食材列表和候选菜谱，按可执行性、食材匹配度、健康度打分。\n"
                "重要要求：\n"
                "1. name 必须是具体的菜谱名称（如'青椒炒肉丝'），不能是网页标题或榜单名称\n"
                "2. 从 candidates 的 title 和 content 中提取真正的菜名\n"
                "3. source_url 必须从 candidates 中的 url 字段获取，保持原样不要修改\n"
                "4. reason 要说明为什么推荐这个菜谱\n"
                "5. score 是 0-100 的分数\n"
                "输出 JSON 格式：{\"top3\": [{\"rank\": 1, \"name\": \"具体菜名\", \"reason\": \"推荐理由\", \"score\": 85, \"source_url\": \"网址\"}], \"table_markdown\": \"markdown表格\"}\n"
                f"ingredients={json.dumps(ingredients, ensure_ascii=False)}\n"
                f"candidates={json.dumps(candidates, ensure_ascii=False)}"
            )
        )
        result = self._chat.invoke([system, user])
        try:
            return json.loads(result.content)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse recipes JSON: {e}")
            logger.error(f"Response content: {result.content}")
            return {"top3": [], "table_markdown": str(result.content)}

    def followup(self, context: dict[str, Any], question: str) -> str:
        response = self._chat.invoke(
            [
                SystemMessage(content="你是私厨助手，回答要分步骤、可执行。"),
                HumanMessage(content=f"context={json.dumps(context, ensure_ascii=False)}\nquestion={question}"),
            ]
        )
        return str(response.content)

    def weekly_plan(self, history_text: str) -> dict[str, Any]:
        response = self._chat.invoke(
            [
                SystemMessage(content="你是营养师，请只输出 JSON，不要输出额外文本。"),
                HumanMessage(
                    content=(
                        "根据用户历史饮食，生成一周饮食计划（早/午/晚餐）。"
                        "输出 JSON，格式为："
                        '{"weekly_plan":[{"day":"周一","breakfast":"...","lunch":"...","dinner":"...","reason":"..."}],'
                        '"weekly_plan_markdown":"markdown表格"}。'
                        "必须包含周一到周日共 7 天。\n"
                        f"history={history_text}"
                    )
                ),
            ]
        )
        raw = str(response.content)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"weekly_plan": [], "weekly_plan_markdown": raw}
        return {
            "weekly_plan": data.get("weekly_plan", []),
            "weekly_plan_markdown": data.get("weekly_plan_markdown", ""),
        }
