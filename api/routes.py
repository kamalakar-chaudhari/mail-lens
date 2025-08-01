# api/chat.py
from io import BytesIO

from fastapi import APIRouter, File, Form, Request, UploadFile

from config.app_context import cas_etl_workflow, pf_analyzer_agent

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    query = body.get("message")

    session_id = request.headers.get("session_id")
    reply = pf_analyzer_agent.invoke(session_id, query)
    return {"reply": reply}


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), password: str = Form(...)):
    file_bytes = await file.read()
    file_stream = BytesIO(file_bytes)

    session_id = request.headers.get("session_id")
    pf_summary = cas_etl_workflow.invoke(session_id, file_stream, password)
    return {"reply": pf_summary}
