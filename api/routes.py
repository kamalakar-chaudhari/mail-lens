# api/chat.py

from fastapi import APIRouter, Request

from config.app_context import pf_analyzer_agent

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    query = body.get("message")

    session_id = request.headers.get("session_id")
    reply = pf_analyzer_agent.invoke(session_id, query)
    return {"reply": reply}
