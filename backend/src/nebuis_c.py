from functools import lru_cache
from typing import List

from openai import OpenAI

from .config import Settings, get_settings


class NebiusClient:
    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
        )
        self._model = settings.nebius_model

    def meera_reply(self, messages: List[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        choice = response.choices[0]
        content = choice.message.content if choice.message else None
        if not content:
            raise RuntimeError("Nebius response did not include message content")
        return content.strip()


@lru_cache
def nebius_client() -> NebiusClient:
    settings = get_settings()
    return NebiusClient(settings)
