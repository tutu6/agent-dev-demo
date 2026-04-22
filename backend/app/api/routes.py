from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.agents.private_chef_agent import PrivateChefAgent
from app.schemas.requests import FollowupRequest, UploadRequest, UrlRequest, WeeklyPlanRequest
from app.schemas.responses import AnalyzeResponse, FollowupResponse, HistoryResponse, WeeklyPlanResponse

router = APIRouter()


def get_agent() -> PrivateChefAgent:
    from app.main import agent

    return agent


@router.post("/upload", response_model=AnalyzeResponse)
def upload_image(payload: UploadRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> AnalyzeResponse:
    try:
        state = chef_agent.analyze_by_upload(payload.thread_id, payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalyzeResponse(
        thread_id=payload.thread_id,
        ingredients=state.get("ingredients", []),
        recipes=state.get("recipes", []),
        table_markdown=state.get("table_markdown", ""),
    )


@router.post("/url", response_model=AnalyzeResponse)
def analyze_url(payload: UrlRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> AnalyzeResponse:
    state = chef_agent.analyze_by_url(payload.thread_id, str(payload.image_url))
    return AnalyzeResponse(
        thread_id=payload.thread_id,
        ingredients=state.get("ingredients", []),
        recipes=state.get("recipes", []),
        table_markdown=state.get("table_markdown", ""),
    )


@router.post("/followup", response_model=FollowupResponse)
def followup(payload: FollowupRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> FollowupResponse:
    state = chef_agent.followup(payload.thread_id, payload.question)
    return FollowupResponse(
        thread_id=payload.thread_id,
        answer=state.get("answer", ""),
        selected_index=state.get("selected_index"),
    )


@router.post("/weekly_plan", response_model=WeeklyPlanResponse)
def weekly_plan(payload: WeeklyPlanRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> WeeklyPlanResponse:
    state = chef_agent.weekly_plan(payload.thread_id, payload.history_text)
    return WeeklyPlanResponse(thread_id=payload.thread_id, weekly_plan_markdown=state.get("weekly_plan_markdown", ""))


@router.get("/history/{thread_id}", response_model=HistoryResponse)
def history(thread_id: str, chef_agent: PrivateChefAgent = Depends(get_agent)) -> HistoryResponse:
    return HistoryResponse(thread_id=thread_id, state=chef_agent.get_history(thread_id))
