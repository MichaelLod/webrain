from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_token, get_current_user_id
from app.models.compute import ComputeResult
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


class ModelInfoResponse(BaseModel):
    name: str
    total_parameters: int
    text_parameters: int
    vision_parameters: int
    architecture: str
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    tokenizer: str
    training_steps: int
    current_loss: float
    training_data_chars: int
    training_data_sources: int
    checkpoint_size_bytes: int


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
async def get_training_status():
    from app.ml.trainer import trainer
    return TrainingStatusResponse(
        current_step=trainer.step,
        current_loss=trainer.current_loss,
        total_flops=trainer.total_flops,
        model_version=1,
        connected_workers=manager.worker_count,
        is_training=manager.is_training,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(db: AsyncSession = Depends(get_db)):
    from app.ml.trainer import trainer
    from app.models.data_submission import DataSubmission, SubmissionStatus

    text_params = sum(p.numel() for p in trainer.model.parameters())
    vision_params = sum(p.numel() for p in trainer.vision_encoder.parameters())
    cfg = trainer.cfg

    # Training data stats
    chars_q = await db.execute(
        select(
            func.coalesce(func.sum(func.length(DataSubmission.extracted_text)), 0),
            func.count(DataSubmission.id),
        ).where(DataSubmission.status == SubmissionStatus.READY)
        .where(DataSubmission.extracted_text.isnot(None))
    )
    row = chars_q.one()
    data_chars = row[0] or 0
    data_sources = row[1] or 0

    # Checkpoint size
    checkpoint_bytes = 0
    import os
    local_ckpt = os.path.join(os.path.dirname(__file__), "../../ml/checkpoints/latest.pt")
    if os.path.exists(local_ckpt):
        checkpoint_bytes = os.path.getsize(local_ckpt)

    return ModelInfoResponse(
        name="TinyGPT",
        total_parameters=text_params + vision_params,
        text_parameters=text_params,
        vision_parameters=vision_params,
        architecture=f"{cfg.n_layers}-layer Transformer + 2-layer Vision Encoder",
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        tokenizer="character-level (256 tokens)",
        training_steps=trainer.step,
        current_loss=trainer.current_loss,
        training_data_chars=data_chars,
        training_data_sources=data_sources,
        checkpoint_size_bytes=checkpoint_bytes,
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
        pass
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
    finally:
        manager.disconnect(websocket)
