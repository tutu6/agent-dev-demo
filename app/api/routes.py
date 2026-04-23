from __future__ import annotations

import base64
import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.agents.private_chef_agent import PrivateChefAgent
from app.adapters import to_analyze_response_output, to_weekly_plan_response_output
from app.domain.errors import ParseError, UpstreamServiceError
from app.schemas.requests import FollowupRequest, UrlRequest, WeeklyPlanRequest
from app.schemas.responses import AnalyzeResponse, FollowupResponse, HistoryResponse, WeeklyPlanResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_agent(request: Request) -> PrivateChefAgent:
    return request.app.state.agent


def _invoke_with_error_mapping(func: Callable[[], dict]) -> dict:
    try:
        return func()
    except ParseError as exc:
        raise HTTPException(status_code=502, detail="模型输出解析失败") from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=502, detail="上游模型服务调用失败") from exc


@router.post("/upload", response_model=AnalyzeResponse)
async def upload_image(
    thread_id: str = Form(...),
    image_file: UploadFile = File(...),
    chef_agent: PrivateChefAgent = Depends(get_agent),
) -> AnalyzeResponse:
    logger.info(f"[API] POST /upload | thread_id={thread_id} | filename={image_file.filename} | content_type={image_file.content_type}")
    content = await image_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty image file")
    image_base64 = base64.b64encode(content).decode("utf-8")
    try:
        state = _invoke_with_error_mapping(lambda: chef_agent.analyze_by_upload(thread_id, image_base64))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output = to_analyze_response_output(state)
    return AnalyzeResponse(
        thread_id=thread_id,
        ingredients=[item.model_dump() for item in output.ingredients],
        recipes=output.recipes,
        table_markdown=output.table_markdown,
    )


@router.post("/url", response_model=AnalyzeResponse)
def analyze_url(payload: UrlRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> AnalyzeResponse:
    logger.info(f"[API] POST /url | thread_id={payload.thread_id} | image_url={payload.image_url}")
    state = _invoke_with_error_mapping(lambda: chef_agent.analyze_by_url(payload.thread_id, str(payload.image_url)))
    output = to_analyze_response_output(state)
    return AnalyzeResponse(
        thread_id=payload.thread_id,
        ingredients=[item.model_dump() for item in output.ingredients],
        recipes=output.recipes,
        table_markdown=output.table_markdown,
    )


@router.post("/followup", response_model=FollowupResponse)
def followup(payload: FollowupRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> FollowupResponse:
    logger.info(f"[API] POST /followup | thread_id={payload.thread_id} | question={payload.question[:50]}...")
    state = _invoke_with_error_mapping(lambda: chef_agent.followup(payload.thread_id, payload.question))
    return FollowupResponse(
        thread_id=payload.thread_id,
        answer=state.get("answer", ""),
        selected_index=state.get("selected_index"),
    )


@router.post("/weekly_plan", response_model=WeeklyPlanResponse)
def weekly_plan(payload: WeeklyPlanRequest, chef_agent: PrivateChefAgent = Depends(get_agent)) -> WeeklyPlanResponse:
    logger.info(f"[API] POST /weekly_plan | thread_id={payload.thread_id} | history_text_length={len(payload.history_text)}")
    state = _invoke_with_error_mapping(lambda: chef_agent.weekly_plan(payload.thread_id, payload.history_text))
    output = to_weekly_plan_response_output(state)
    return WeeklyPlanResponse(
        thread_id=payload.thread_id,
        weekly_plan=[item.model_dump() for item in output.weekly_plan],
        weekly_plan_markdown=output.weekly_plan_markdown,
    )


@router.get("/history/{thread_id}", response_model=HistoryResponse)
def history(thread_id: str, chef_agent: PrivateChefAgent = Depends(get_agent)) -> HistoryResponse:
    logger.info(f"[API] GET /history/{thread_id}")
    return HistoryResponse(thread_id=thread_id, state=chef_agent.get_history(thread_id))
