from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.token import TokenTransaction
from app.models.user import User

router = APIRouter()


class BalanceResponse(BaseModel):
    balance: int


class TxResponse(BaseModel):
    id: int
    amount: int
    tx_type: str
    reference_id: str | None
    balance_after: int
    created_at: str


@router.get("/balance", response_model=BalanceResponse)
async def get_balance(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User.token_balance).where(User.id == user_id))
    balance = result.scalar_one()
    return BalanceResponse(balance=balance)


@router.get("/history", response_model=list[TxResponse])
async def get_history(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    result = await db.execute(
        select(TokenTransaction)
        .where(TokenTransaction.user_id == user_id)
        .order_by(TokenTransaction.created_at.desc())
        .limit(limit)
    )
    txs = result.scalars().all()
    return [
        TxResponse(
            id=tx.id,
            amount=tx.amount,
            tx_type=tx.tx_type.value,
            reference_id=tx.reference_id,
            balance_after=tx.balance_after,
            created_at=tx.created_at.isoformat(),
        )
        for tx in txs
    ]
