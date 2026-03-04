from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.core.database import Base, engine
from app.api.v1 import auth, tokens, compute, chat, leaderboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="WeBrain - AI from the people, for the people",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(tokens.router, prefix="/api/v1/tokens", tags=["tokens"])
app.include_router(compute.router, prefix="/api/v1/compute", tags=["compute"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(leaderboard.router, prefix="/api/v1/leaderboard", tags=["leaderboard"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
