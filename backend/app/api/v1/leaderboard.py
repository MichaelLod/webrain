from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.compute import ComputeResult
from app.models.user import User

router = APIRouter()


class LeaderboardEntry(BaseModel):
    rank: int
    display_name: str
    tiles_computed: int
    tokens_earned: int


class LeaderboardResponse(BaseModel):
    top_contributors: list[LeaderboardEntry]
    total_contributors: int
    total_tiles: int
    total_compute_time_ms: float


@router.get("", response_model=LeaderboardResponse)
async def get_leaderboard(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    # Top contributors by tiles computed
    top_q = (
        select(
            User.display_name,
            func.count(ComputeResult.id).label("tiles"),
            func.coalesce(func.sum(ComputeResult.tokens_awarded), 0).label("tokens"),
        )
        .join(ComputeResult, ComputeResult.user_id == User.id)
        .group_by(User.id, User.display_name)
        .order_by(func.count(ComputeResult.id).desc())
        .limit(min(limit, 100))
    )
    rows = (await db.execute(top_q)).all()

    entries = [
        LeaderboardEntry(
            rank=i + 1,
            display_name=row.display_name,
            tiles_computed=row.tiles,
            tokens_earned=row.tokens,
        )
        for i, row in enumerate(rows)
    ]

    # Totals
    totals_q = select(
        func.count(func.distinct(ComputeResult.user_id)),
        func.count(ComputeResult.id),
        func.coalesce(func.sum(ComputeResult.compute_time_ms), 0),
    )
    totals = (await db.execute(totals_q)).one()

    return LeaderboardResponse(
        top_contributors=entries,
        total_contributors=totals[0],
        total_tiles=totals[1],
        total_compute_time_ms=totals[2],
    )
