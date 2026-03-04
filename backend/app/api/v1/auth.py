from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from app.core.config import SIGNUP_BONUS_TOKENS
from app.models.user import User
from app.models.token import TokenTransaction, TxType

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: str
    token_balance: int
    compute_trust_score: float


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        token_balance=SIGNUP_BONUS_TOKENS,
    )
    db.add(user)
    await db.flush()

    tx = TokenTransaction(
        user_id=user.id,
        amount=SIGNUP_BONUS_TOKENS,
        tx_type=TxType.SIGNUP_BONUS,
        balance_after=SIGNUP_BONUS_TOKENS,
    )
    db.add(tx)
    await db.commit()
    await db.refresh(user)

    return AuthResponse(
        token=create_access_token(user.id),
        user={
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "token_balance": user.token_balance,
        },
    )


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return AuthResponse(
        token=create_access_token(user.id),
        user={
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "token_balance": user.token_balance,
        },
    )


@router.get("/me", response_model=UserResponse)
async def me(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        token_balance=user.token_balance,
        compute_trust_score=user.compute_trust_score,
    )
