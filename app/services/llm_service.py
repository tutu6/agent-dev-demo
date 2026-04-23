from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.domain.errors import ParseError, UpstreamServiceError
from app.domain.models import Ingredient, RankRecipesResult, RecipeCandidate, WeeklyPlanResult

logger = logging.getLogger(__name__)


class LLMService:
    """Wraps Qwen chat and vision models using LangChain."""

    def __init__(self, settings: Settings) -> None:
        self._rerank_candidate_limit = max(1, settings.rerank_candidate_limit)
        self._rerank_content_max_chars = max(1, settings.rerank_content_max_chars)
        self._chat = ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            temperature=settings.qwen_chat_temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self._vision = ChatOpenAI(
            model=settings.qwen_vision_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            temperature=settings.qwen_vision_temperature,
        )

    @staticmethod
    def _extract_json_text(raw_content: str) -> str:
        if "```json" in raw_content:
            return raw_content.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
        if "```" in raw_content:
            return raw_content.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
        return raw_content

    def _parse_json(self, raw_content: str, operation: str) -> Any:
        json_str = self._extract_json_text(raw_content)
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.exception("failed to parse %s response: %s", operation, raw_content[:500])
            raise ParseError(f"failed to parse model output for {operation}") from exc

    def _prepare_candidates_for_rerank(
        self, ingredients: list[Ingredient], candidates: list[RecipeCandidate]
    ) -> list[dict[str, str]]:
        ingredient_names = [item.name.strip().lower() for item in ingredients if item.name.strip()]

        deduped: list[RecipeCandidate] = []
        seen_urls: set[str] = set()
        for candidate in candidates:
            url = candidate.url.strip()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            deduped.append(candidate)

        scored: list[tuple[int, RecipeCandidate]] = []
        for candidate in deduped:
            haystack = f"{candidate.title}\n{candidate.content}".lower()
            overlap = sum(1 for name in ingredient_names if name and name in haystack)
            scored.append((overlap, candidate))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in scored[: self._rerank_candidate_limit]]

        compacted: list[dict[str, str]] = []
        for item in selected:
            compacted.append(
                {
                    "title": item.title.strip()[:120],
                    "content": item.content.strip()[: self._rerank_content_max_chars],
                    "url": item.url.strip(),
                }
            )
        return compacted

    def recognize_ingredients(self, image_url: str) -> list[Ingredient]:
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
        try:
            response = self._vision.invoke([message])
        except Exception as exc:  # noqa: BLE001
            raise UpstreamServiceError("vision model call failed") from exc

        raw_content = str(response.content).strip()
        parsed = self._parse_json(raw_content, "recognize_ingredients")
        return [Ingredient.model_validate(item) for item in parsed]

    def rank_recipes(self, ingredients: list[Ingredient], candidates: list[RecipeCandidate]) -> RankRecipesResult:
        compact_candidates = self._prepare_candidates_for_rerank(ingredients, candidates)
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
                f"ingredients={json.dumps([item.model_dump() for item in ingredients], ensure_ascii=False)}\n"
                f"candidates={json.dumps(compact_candidates, ensure_ascii=False)}"
            )
        )
        started = time.perf_counter()
        try:
            result = self._chat.invoke([system, user])
        except Exception as exc:  # noqa: BLE001
            raise UpstreamServiceError("chat model call failed in rank_recipes") from exc
        elapsed = time.perf_counter() - started
        logger.info(
            "rank_recipes latency=%.2fs candidates_raw=%s candidates_sent=%s",
            elapsed,
            len(candidates),
            len(compact_candidates),
        )

        raw_content = str(result.content).strip()
        parsed = self._parse_json(raw_content, "rank_recipes")
        return RankRecipesResult.model_validate(parsed)

    def followup(self, context: dict[str, Any], question: str) -> str:
        try:
            response = self._chat.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是私厨助手，回答要分步骤、可执行。\n"
                            "请生成严格遵循 CommonMark 规范的 Markdown 内容：\n"
                            "- 只使用标准 Markdown 语法（不支持表情符号、HTML标签）\n"
                            "- 标题使用 #、##、###\n"
                            "- 列表使用 - 或 1. 并保持正确的缩进\n"
                            "- 加粗使用 **text**，斜体使用 *text*\n"
                            "- 不使用 😀、🍳 等表情符号\n"
                            "- 引用块使用 > 后加空格\n"
                            "- 代码或食材名称使用 `反引号` 标注"
                        )
                    ),
                    HumanMessage(content=f"context={json.dumps(context, ensure_ascii=False)}\nquestion={question}"),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamServiceError("chat model call failed in followup") from exc
        return str(response.content)

    def weekly_plan(self, history_text: str) -> WeeklyPlanResult:
        try:
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
        except Exception as exc:  # noqa: BLE001
            raise UpstreamServiceError("chat model call failed in weekly_plan") from exc

        raw_content = str(response.content).strip()
        parsed = self._parse_json(raw_content, "weekly_plan")
        return WeeklyPlanResult.model_validate(parsed)
