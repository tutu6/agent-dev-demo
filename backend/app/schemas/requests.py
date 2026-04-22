from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class UrlRequest(BaseModel):
    thread_id: str = Field(..., description="会话 ID")
    image_url: HttpUrl


class FollowupRequest(BaseModel):
    thread_id: str
    question: str


class WeeklyPlanRequest(BaseModel):
    thread_id: str
    history_text: str = Field(..., description="用户历史饮食文本")
