import os

_raw_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://webrain:webrain_dev@localhost:5433/webrain",
)
# Railway provides postgresql:// — SQLAlchemy async needs postgresql+asyncpg://
DATABASE_URL = _raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1) if _raw_db_url.startswith("postgresql://") else _raw_db_url
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
SIGNUP_BONUS_TOKENS = 100
TILE_SIZE = 64
