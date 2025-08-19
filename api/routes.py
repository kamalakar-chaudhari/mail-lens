# api/chat.py

from fastapi import APIRouter, Request

from config.app_context import workflow_agent

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    query = body.get("message")

    reply = await workflow_agent.run(input=query)
    return {"reply": reply}
