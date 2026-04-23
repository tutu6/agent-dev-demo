from __future__ import annotations

import base64
from typing import Any

from app.adapters import serialize_model
from app.graph.state import ChefState


class PrivateChefAgent:
    def __init__(self, compiled_graph: Any) -> None:
        self.graph = compiled_graph

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _base64_to_data_url(image_base64: str) -> str:
        try:
            base64.b64decode(image_base64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("invalid base64 image") from exc
        return f"data:image/jpeg;base64,{image_base64}"

    def analyze_by_upload(self, thread_id: str, image_base64: str) -> ChefState:
        image_url = self._base64_to_data_url(image_base64)
        return self.analyze_by_url(thread_id=thread_id, image_url=image_url)

    def analyze_by_url(self, thread_id: str, image_url: str) -> ChefState:
        state: ChefState = {
            "thread_id": thread_id,
            "intent": "analyze",
            "input_image_url": image_url,
            "step": "start",
        }
        return self.graph.invoke(state, config=self._config(thread_id))

    def followup(self, thread_id: str, question: str) -> ChefState:
        state: ChefState = {
            "thread_id": thread_id,
            "intent": "followup",
            "question": question,
        }
        return self.graph.invoke(state, config=self._config(thread_id))

    def weekly_plan(self, thread_id: str, history_text: str) -> ChefState:
        state: ChefState = {
            "thread_id": thread_id,
            "intent": "weekly_plan",
            "history_text": history_text,
        }
        return self.graph.invoke(state, config=self._config(thread_id))

    def get_history(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot:
            return {}

        serialized: dict[str, Any] = {}
        for key, value in dict(snapshot.values).items():
            serialized[key] = serialize_model(value)
        return serialized
