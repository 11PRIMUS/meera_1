import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5500",
]
DEFAULT_NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1/"
DEFAULT_NEBIUS_MODEL ="openai/gpt-oss-120b"
DEFAULT_ASSISTANT_PROMPT = (
    "You are Meera, an empathetic journaling companion. "
    "Respond with warmth, validation, and gentle prompts that help users reflect."
)


class Settings(BaseModel):
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str | None = None
    diary_table: str = "diary_entries"
    messages_table: str = "messages"
    allowed_origins: List[str] = DEFAULT_ALLOWED_ORIGINS
    nebius_api_key: str
    nebius_base_url: str = DEFAULT_NEBIUS_BASE_URL
    nebius_model: str = DEFAULT_NEBIUS_MODEL
    assistant_system_prompt: str = DEFAULT_ASSISTANT_PROMPT


def web_origins() -> List[str]:
    env_value=os.getenv("ALLOWED_ORIGINS")
    if not env_value:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in env_value.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment."
        )

    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    if not nebius_api_key:
        raise RuntimeError("NEBIUS_API_KEY must be set in the environment.")

    allowed_origins = web_origins()

    return Settings(
        supabase_url=supabase_url,
        supabase_service_role_key=supabase_key,
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
        allowed_origins=allowed_origins,
        nebius_api_key=nebius_api_key,
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL),
        nebius_model=os.getenv("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL),
        assistant_system_prompt=os.getenv(
            "ASSISTANT_SYSTEM_PROMPT", DEFAULT_ASSISTANT_PROMPT
        ),
    )
