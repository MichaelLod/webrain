"""HTTP endpoints for weight shard downloads.

More efficient than WebSocket for large transfers — supports Range headers
and fp16 transfer.
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response

log = logging.getLogger(__name__)

router = APIRouter()

SHARDS_DIR = os.path.join(os.path.dirname(__file__), "../../ml/checkpoints/shards")


@router.get("/shards/manifest")
async def get_manifest():
    manifest_path = os.path.join(SHARDS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="No sharded checkpoint available")
    with open(manifest_path) as f:
        return json.load(f)


@router.get("/shards/{shard_name}")
async def get_shard(shard_name: str, request: Request):
    shard_path = os.path.join(SHARDS_DIR, shard_name)
    if not os.path.exists(shard_path):
        raise HTTPException(status_code=404, detail=f"Shard {shard_name} not found")

    file_size = os.path.getsize(shard_path)

    # Handle Range header for partial downloads
    range_header = request.headers.get("range")
    if range_header:
        try:
            range_str = range_header.replace("bytes=", "")
            start_str, end_str = range_str.split("-")
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(shard_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            return Response(
                content=data,
                status_code=206,
                media_type="application/octet-stream",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(length),
                    "Accept-Ranges": "bytes",
                },
            )
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid range")

    with open(shard_path, "rb") as f:
        data = f.read()

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        },
    )
