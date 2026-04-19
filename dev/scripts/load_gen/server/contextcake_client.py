import os
from typing import Any, Dict, Optional

import requests
from fastapi import HTTPException


def normalize_contextcake_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


CONTEXTCAKE_BASE_URL = normalize_contextcake_base_url(
    os.getenv("CONTEXTCAKE_BASE_URL", "http://localhost:50350")
)


def get_contextcake_base_url() -> str:
    return CONTEXTCAKE_BASE_URL


def build_contextcake_openai_base_url(base_url: str) -> str:
    return f"{normalize_contextcake_base_url(base_url)}/v1"


def set_contextcake_base_url(base_url: str) -> str:
    normalized = normalize_contextcake_base_url(base_url)
    os.environ["CONTEXTCAKE_BASE_URL"] = normalized
    global CONTEXTCAKE_BASE_URL
    CONTEXTCAKE_BASE_URL = normalized
    return normalized


def parse_http_response_body(response: requests.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


class ContextCakeHttpClient:
    def __init__(
        self,
        base_url: str,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = normalize_contextcake_base_url(base_url)
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

    def _request(
        self,
        *,
        round_idx: int,
        phase: str,
        operation: str,
        method: str,
        path: str,
        context_id: Optional[str],
        request_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        del round_idx, phase, operation, context_id
        try:
            response = self.session.request(
                method=method,
                url=f"{self.base_url}{path}",
                json=request_body,
                timeout=30,
            )
            response.raise_for_status()
            return parse_http_response_body(response)
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=500,
                detail=f"ContextCake {method} {path} failed: {exc}",
            ) from exc

    def create_context(self, round_idx: int, context_id: str) -> Any:
        return self._request(
            round_idx=round_idx,
            phase="pre",
            operation="create_context",
            method="PUT",
            path=f"/context/{context_id}",
            context_id=context_id,
            request_body={"context_type": "durable"},
        )

    def one_off(self, round_idx: int, context_id: str, msg_index: int) -> Any:
        return self._request(
            round_idx=round_idx,
            phase="pre",
            operation="one_off",
            method="POST",
            path="/one-off",
            context_id=context_id,
            request_body={"context_id": context_id, "msg_index": msg_index},
        )

    def truncate(self, round_idx: int, phase: str, context_id: str, msg_index: int) -> Any:
        return self._request(
            round_idx=round_idx,
            phase=phase,
            operation="truncate",
            method="POST",
            path="/truncate",
            context_id=context_id,
            request_body={"context_id": context_id, "msg_index": msg_index},
        )

    def suspend(self, round_idx: int, phase: str, context_id: str) -> Any:
        return self._request(
            round_idx=round_idx,
            phase=phase,
            operation="suspend",
            method="POST",
            path="/suspend",
            context_id=context_id,
            request_body={"context_id": context_id},
        )

    def delete_context(self, round_idx: int, context_id: str) -> Any:
        return self._request(
            round_idx=round_idx,
            phase="post",
            operation="delete_context",
            method="DELETE",
            path=f"/context/{context_id}",
            context_id=context_id,
            request_body=None,
        )
