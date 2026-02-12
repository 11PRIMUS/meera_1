from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="authenticated Supabase user id")
    message: str = Field(..., min_length=1, max_length=2000)

class ChatMessage(BaseModel):
    id: Optional[str] = None
    user_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: Optional[datetime] = None


class DiaryEntry(BaseModel):
    id: Optional[str] = None
    user_id: str
    title: str
    content: str
    created_at: Optional[datetime] = None

class DiaryResponse(BaseModel):
    entries: list[DiaryEntry]

class ChatResponse(BaseModel):
    reply: str
    diary_entry: DiaryEntry | None = None
    history: list[ChatMessage]
