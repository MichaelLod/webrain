from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import HF_REPO_ID, HF_TOKEN
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
    collective_intelligence: float
    active_experts: int
    pipeline_stages: int
    pipeline_active: bool


class ModelInfoResponse(BaseModel):
    name: str
    total_parameters: int
    text_parameters: int
    vision_parameters: int
    architecture: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    tokenizer: str
    norm_type: str
    ff_type: str
    pos_encoding: str
    training_steps: int
    current_loss: float
    training_data_chars: int
    training_data_sources: int
    checkpoint_size_bytes: int
    huggingface_url: str | None = None


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
    from app.ml.trainer import trainer, CONFIG_VERSION
    swarm = manager.swarm
    return TrainingStatusResponse(
        current_step=trainer.step,
        current_loss=trainer.current_loss,
        total_flops=trainer.total_flops,
        model_version=CONFIG_VERSION,
        connected_workers=manager.worker_count,
        is_training=manager.is_training,
        collective_intelligence=swarm.collective_intelligence if swarm else 0.25,
        active_experts=swarm.active_experts if swarm else 1,
        pipeline_stages=manager.pipeline_stages,
        pipeline_active=manager.pipeline_active,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(db: AsyncSession = Depends(get_db)):
    from app.ml.trainer import trainer
    from app.models.data_submission import DataSubmission, SubmissionStatus

    text_params = sum(p.numel() for p in trainer.model.parameters())
    vision_params = sum(p.numel() for p in trainer.vision_encoder.parameters())
    cfg = trainer.cfg

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

    checkpoint_bytes = 0
    import os
    local_ckpt = os.path.join(os.path.dirname(__file__), "../../ml/checkpoints/latest.pt")
    if os.path.exists(local_ckpt):
        checkpoint_bytes = os.path.getsize(local_ckpt)

    hf_url = f"https://huggingface.co/{HF_REPO_ID}" if HF_REPO_ID and HF_TOKEN else None

    tokenizer_desc = f"BPE ({cfg.vocab_size} tokens)"

    return ModelInfoResponse(
        name="WeBrainGPT",
        total_parameters=text_params + vision_params,
        text_parameters=text_params,
        vision_parameters=vision_params,
        architecture=f"{cfg.n_layers}-layer Transformer (GQA + SwiGLU + RoPE) + 2-layer Vision Encoder",
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        tokenizer=tokenizer_desc,
        norm_type="RMSNorm",
        ff_type="SwiGLU",
        pos_encoding=f"RoPE (theta={cfg.rope_theta})",
        training_steps=trainer.step,
        current_loss=trainer.current_loss,
        training_data_chars=data_chars,
        training_data_sources=data_sources,
        checkpoint_size_bytes=checkpoint_bytes,
        huggingface_url=hf_url,
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
