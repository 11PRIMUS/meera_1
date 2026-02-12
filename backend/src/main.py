from .config import get_settings
from .nebuis_c import nebius_client
from .supabase_c import supabase_client
from .schemas import ChatMessage, DiaryResponse, DiaryEntry, ChatRequest, ChatResponse
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from datetime import date, datetime, timezone
from typing import Any, Dict, List
import httpx
from openai import OpenAIError


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

@app.get("/api/diary/{user_id}", response_model=DiaryResponse)
def list_diary(user_id: str, limit:int =10)-> DiaryResponse:
    try:
        rows=supabase.select(settings.diary_table, filters={"user_id":user_id}, order="created_at", descending=True, limit=limit)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"failed to load meera diary: {exec}") from exc
    
    entries=[DiaryEntry(**item) for item in rows]
    return DiaryResponse(entries=entries)

#meera diary msg store
def store_msg(payload:Dict[str, Any]) ->ChatMessage:
    try:
        inserted = supabase.insert(settings.messages_table, payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to store msg in meera diary :{exc}") from exc
    return ChatMessage(**inserted)

def diary_entry(payload:Dict[str, Any])-> DiaryEntry:
    try:
        inserted=supabase.insert(settings.diary_table, payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to store diary entry :{exc}") from exc
    return DiaryEntry(**inserted)

def find_entry(user_id:str, target_date:date)->DiaryEntry | None:
    try:
        rows=supabase.select(settings.diary_table, filters={"user_id":user_id}, order="created_at", descending=True, limit=25)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"failed to get diary entry: {exc}") from exc
    
    for row in rows:
        entry=DiaryEntry(**row)
        if entry.created_at and entry.created_at.date()==target_date:
            return entry
    return None

def fetch_msgdate(user_id:str, target_date:date, limit:int=500)->List[ChatMessage]:
    try:
        rows=supabase.select(
            settings.messages_table, filters={"user_id":user_id}, order="created_id", descending=True, limit=limit)
    except httpx.HTTPError as exc:  
        raise HTTPException(status_code=502, detail=f"failed to load history: {exc}") from exc

    day_messages: List[ChatMessage] = []
    for item in rows:
        message = ChatMessage(**item)
        if message.created_at and message.created_at.date() == target_date:
            day_messages.append(message)
    return list(reversed(day_messages))  

def format_his(history: List[ChatMessage]) ->List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content":settings.assistant_system_prompt}
    ]
    messages.extend({"role":item.role,"content": item.content} for item in history)
    return messages

#build meera reply
def meera_reply(history: List[ChatMessage]) -> str:
    messages = format_his(history)
    try:
        return nebius_client.generate_reply(messages)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502, detail=f"Assistant failed to generate a reply: {exc}"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

