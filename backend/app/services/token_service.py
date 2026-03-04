from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.token import TokenTransaction, TxType
from app.models.user import User


async def credit_tokens(
    db: AsyncSession,
    user_id: int,
    amount: int,
    tx_type: TxType,
    reference_id: str | None = None,
) -> int:
    result = await db.execute(
        select(User).where(User.id == user_id).with_for_update()
    )
    user = result.scalar_one()
    user.token_balance += amount
    new_balance = user.token_balance

    tx = TokenTransaction(
        user_id=user_id,
        amount=amount,
        tx_type=tx_type,
        reference_id=reference_id,
        balance_after=new_balance,
    )
    db.add(tx)
    await db.flush()
    return new_balance


async def debit_tokens(
    db: AsyncSession,
    user_id: int,
    amount: int,
    tx_type: TxType,
    reference_id: str | None = None,
) -> int:
    result = await db.execute(
        select(User).where(User.id == user_id).with_for_update()
    )
    user = result.scalar_one()
    if user.token_balance < amount:
        raise HTTPException(status_code=402, detail="Insufficient tokens")

    user.token_balance -= amount
    new_balance = user.token_balance

    tx = TokenTransaction(
        user_id=user_id,
        amount=-amount,
        tx_type=tx_type,
        reference_id=reference_id,
        balance_after=new_balance,
    )
    db.add(tx)
    await db.flush()
    return new_balance
