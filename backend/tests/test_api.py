from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import get_agent
from app.main import app


class FakeAgent:
    def analyze_by_upload(self, thread_id: str, image_base64: str):
        return {
            "ingredients": [{"name": "鸡蛋", "freshness": 90, "remaining_ratio": 0.5, "note": "ok"}],
            "recipes": [{"rank": 1, "name": "蛋炒饭", "reason": "快", "score": 9.0, "source_url": "https://x.com"}],
            "table_markdown": "|rank|name|",
        }

    def analyze_by_url(self, thread_id: str, image_url: str):
        return self.analyze_by_upload(thread_id, "")

    def followup(self, thread_id: str, question: str):
        return {"answer": "步骤1...", "selected_index": 0}

    def weekly_plan(self, thread_id: str, history_text: str):
        return {"weekly_plan_markdown": "|周一|周二|"}

    def get_history(self, thread_id: str):
        return {"thread_id": thread_id, "step": "recipes_ranked"}


def test_all_required_apis():
    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    client = TestClient(app)

    upload_resp = client.post("/upload", json={"thread_id": "t1", "image_base64": "ZmFrZQ=="})
    assert upload_resp.status_code == 200

    url_resp = client.post("/url", json={"thread_id": "t1", "image_url": "https://example.com/a.jpg"})
    assert url_resp.status_code == 200

    followup_resp = client.post("/followup", json={"thread_id": "t1", "question": "怎么做"})
    assert followup_resp.status_code == 200

    weekly_resp = client.post("/weekly_plan", json={"thread_id": "t1", "history_text": "过去经常吃面"})
    assert weekly_resp.status_code == 200

    history_resp = client.get("/history/t1")
    assert history_resp.status_code == 200

    app.dependency_overrides.clear()
