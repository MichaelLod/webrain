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

# S3-compatible bucket for model checkpoints
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://t3.storageapi.dev")
S3_REGION = os.getenv("S3_REGION", "auto")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Hugging Face Hub
HF_REPO_ID = os.getenv("HF_REPO_ID", "webrain/thepeoplesai")
HF_TOKEN = os.getenv("HF_TOKEN", "")

CHAT_IMAGE_COST = 20
