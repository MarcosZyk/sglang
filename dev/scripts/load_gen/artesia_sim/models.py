from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CreateContextRequest(BaseModel):
    context_type: str


class ContextIndexRequest(BaseModel):
    context_id: str
    msg_index: int


class ContextOnlyRequest(BaseModel):
    context_id: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(ge=0)
    temperature: float = 0.0
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    context_id: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    stream: bool = False

    model_config = ConfigDict(extra="allow")


class PromptTokensDetails(BaseModel):
    cached_tokens: int


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails


class CompletionMessage(BaseModel):
    role: str
    content: str


class CompletionChoice(BaseModel):
    index: int
    message: CompletionMessage
    finish_reason: str
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage
    prefill_time: float
    decode_time: List[float]
    num_local_cache: int
    num_global_cache: int

    model_config = ConfigDict(extra="allow")
