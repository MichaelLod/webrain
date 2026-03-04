from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_token, get_current_user_id
from app.models.compute import ComputeResult
from app.models.training import TrainingState
from app.services.compute_service import manager

router = APIRouter()


class StatsResponse(BaseModel):
    tasks_completed: int
    total_compute_time_ms: float
    tokens_earned: int


class TrainingStatusResponse(BaseModel):
    current_step: int
    current_loss: float
    total_flops: float
    model_version: int
    connected_workers: int
    is_training: bool


@router.get("/stats", response_model=StatsResponse)
async def get_compute_stats(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(
            func.count(ComputeResult.id),
            func.coalesce(func.sum(ComputeResult.compute_time_ms), 0),
            func.coalesce(func.sum(ComputeResult.tokens_awarded), 0),
        ).where(ComputeResult.user_id == user_id)
    )
    row = result.one()
    return StatsResponse(
        tasks_completed=row[0],
        total_compute_time_ms=row[1],
        tokens_earned=row[2],
    )


@router.get("/training-status", response_model=TrainingStatusResponse)
async def get_training_status(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrainingState).limit(1))
    state = result.scalar_one_or_none()
    return TrainingStatusResponse(
        current_step=state.current_step if state else 0,
        current_loss=state.current_loss if state else 0.0,
        total_flops=state.total_flops if state else 0.0,
        model_version=state.model_version if state else 1,
        connected_workers=manager.worker_count,
        is_training=manager.is_training,
    )


@router.websocket("/ws")
async def compute_websocket(websocket: WebSocket, token: str):
    try:
        user_id = decode_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(websocket, user_id, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
