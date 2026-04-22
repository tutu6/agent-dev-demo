from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings


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
        return json.loads(response.content)

    def rank_recipes(self, ingredients: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        system = SystemMessage(content="你是严谨的私厨助手。只输出 JSON。")
        user = HumanMessage(
            content=(
                "根据食材列表和候选菜谱，按可执行性、食材匹配度、健康度打分。"
                "输出 JSON: {top3:[{rank,name,reason,score,source_url}], table_markdown:""...""}。\n"
                f"ingredients={json.dumps(ingredients, ensure_ascii=False)}\n"
                f"candidates={json.dumps(candidates, ensure_ascii=False)}"
            )
        )
        result = self._chat.invoke([system, user])
        return json.loads(result.content)

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
