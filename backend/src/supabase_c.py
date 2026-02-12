import httpx
from .config import Settings, get_settings
from functools import lru_cache
from typing import Any, Dict, Optional

class SupabaseClient:
    def __init__(self, settings: Settings) -> None:
        self._settings=settings
        self._client =httpx.Client(base_url=f"{settings.supabase_url}/rest/v1", headers={
                "apikey": settings.supabase_service_role_key,
                "Authorization": f"Bearer {settings.supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )

    def select(self, table: str, filters: Optional[Dict[str, Any]] = None, order: Optional[str] = None, descending: bool = False, limit: Optional[int] = None,) -> list[Dict[str, Any]]:
        params: Dict[str, Any]={"select": "*"}
        for key, value in (filters or {}).items():
            if isinstance(value, tuple) and len(value) == 2:
                operator, raw_value = value
                params.setdefault(key, [])
                existing = params[key]
                if isinstance(existing, list):
                    existing.append((operator, raw_value))
                else:
                    params[key] = [(operator, raw_value)]
            else:
                params[key] = f"eq.{value}"
        if order:
            direction = "desc" if descending else "asc"
            params["order"] = f"{order}.{direction}"
        if limit is not None:
            params["limit"] = limit

        flat_params: list[tuple[str, Any]] = []
        for key, value in params.items():
            if isinstance(value, list):
                for operator, raw_value in value:
                    flat_params.append((key, f"{operator}.{raw_value}"))
            else:
                flat_params.append((key, value))

        response = self._client.get(f"/{table}", params=flat_params)
        response.raise_for_status()
        return response.json()

    def insert(self, table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.post(
            f"/{table}",
            json=[payload],
            headers={"Prefer": "return=representation"},
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}

    def update(
        self,
        table: str,
        filters: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, value in filters.items():
            params[key] = f"eq.{value}"

        response = self._client.patch(
            f"/{table}",
            params=params,
            json=payload,
            headers={"Prefer": "return=representation"},
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if data else {}


@lru_cache
def get_client() -> SupabaseClient:
    settings = get_settings()
    return SupabaseClient(settings)
