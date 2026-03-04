import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import CHAT_IMAGE_COST
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.chat import ChatMessage
from app.models.token import TxType
from app.services.token_service import debit_tokens
from app.ml.inference import generate_text

router = APIRouter()

CHAT_COST = 10


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@router.post("/send")
async def send_message(
    body: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conversation_id = body.conversation_id or str(uuid.uuid4())

    await debit_tokens(db, user_id, CHAT_COST, TxType.CHAT_SPEND, conversation_id)

    user_msg = ChatMessage(
        user_id=user_id,
        conversation_id=conversation_id,
        role="user",
        content=body.message,
        token_cost=CHAT_COST,
    )
    db.add(user_msg)
    await db.commit()

    async def stream():
        full_response = ""
        for token in generate_text(body.message):
            full_response += token
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

        async with db.begin():
            assistant_msg = ChatMessage(
                user_id=user_id,
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                token_cost=0,
            )
            db.add(assistant_msg)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/send-with-image")
async def send_message_with_image(
    message: str = Form(...),
    conversation_id: str | None = Form(None),
    image: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    from app.ml.vision import preprocess_image

    conv_id = conversation_id or str(uuid.uuid4())

    await debit_tokens(db, user_id, CHAT_IMAGE_COST, TxType.CHAT_SPEND, conv_id)

    image_bytes = await image.read()
    image_tensor = preprocess_image(image_bytes)

    user_msg = ChatMessage(
        user_id=user_id,
        conversation_id=conv_id,
        role="user",
        content=f"[image] {message}",
        token_cost=CHAT_IMAGE_COST,
    )
    db.add(user_msg)
    await db.commit()

    async def stream():
        full_response = ""
        for token in generate_text(message, image=image_tensor):
            full_response += token
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

        async with db.begin():
            assistant_msg = ChatMessage(
                user_id=user_id,
                conversation_id=conv_id,
                role="assistant",
                content=full_response,
                token_cost=0,
            )
            db.add(assistant_msg)

    return StreamingResponse(stream(), media_type="text/event-stream")
