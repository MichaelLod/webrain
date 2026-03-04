import io
import logging

import numpy as np
import torch
from PIL import Image

from app.core.config import S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT, S3_REGION

log = logging.getLogger(__name__)

IMAGE_SIZE = 224
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

S3_IMAGE_PREFIX = "vision/"


def _get_s3_client():
    if not S3_BUCKET or not S3_ACCESS_KEY_ID:
        return None
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Resize and normalize image bytes to tensor [3, 224, 224]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return torch.from_numpy(arr)


def save_image_tensor_to_s3(tensor: torch.Tensor, submission_id: int) -> str | None:
    """Store preprocessed tensor in S3. Returns the S3 key or None on failure."""
    s3 = _get_s3_client()
    if not s3:
        return None
    key = f"{S3_IMAGE_PREFIX}{submission_id}.pt"
    buf = io.BytesIO()
    torch.save(tensor, buf)
    buf.seek(0)
    try:
        s3.upload_fileobj(buf, S3_BUCKET, key)
        log.info("Saved image tensor to S3: %s", key)
        return key
    except Exception as e:
        log.error("Failed to save image tensor to S3: %s", e)
        return None


def load_image_tensor_from_s3(submission_id: int) -> torch.Tensor | None:
    """Load preprocessed image tensor from S3."""
    s3 = _get_s3_client()
    if not s3:
        return None
    key = f"{S3_IMAGE_PREFIX}{submission_id}.pt"
    buf = io.BytesIO()
    try:
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as e:
        log.debug("Failed to load image tensor from S3 (%s): %s", key, e)
        return None


def delete_image_from_s3(submission_id: int):
    """Delete image tensor from S3 after training."""
    s3 = _get_s3_client()
    if not s3:
        return
    key = f"{S3_IMAGE_PREFIX}{submission_id}.pt"
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        log.info("Deleted image tensor from S3: %s", key)
    except Exception as e:
        log.warning("Failed to delete image tensor from S3: %s", e)
