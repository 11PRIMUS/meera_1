from .config import get_settings
from .nebuis_c import nebius_client
from .supabase_c import supabase_client
from .schemas import ChatMessage, ChatRequest, ChatResponse, DiaryResponse
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from datetime import date, datetime, timezone
from typing import Any, Dict, List
import httpx


settings=get_settings()
supabase=supabase_client()
nebuis = nebius_client()

app=FastAPI(title="meera 3", version="0.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health()-> Dict[str, str]:
    return {"status":"ok"}

@app.options("/api/chat")
def chat_pre()->Response:
    return Response(status_code=204)

#chat history
@app.get("/api/chat/history/{user_id}", response_model=List[ChatMessage])
def get_history(user_id:str, limit:int=25)->List[ChatMessage]:
    try:
        rows=supabase.select(settings.messages_table, filters={"user_id":user_id}, order="created_at", descending=True, limit=limit)
    except httpx.HTTPException as exc:
        raise HTTPException(status_code =502, detail=f"failed to load_history:{exc}") from exc
    return  list (reversed([ChatMessage(**item) for item in rows]))

