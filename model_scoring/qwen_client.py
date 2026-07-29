"""Alibaba Cloud Qwen (DashScope) multimodal API — OpenAI-compatible mode."""

from __future__ import annotations

import os
from typing import Any

from openai_compat_client import compat_chat
from provider_registry import get_spec
from pydantic import BaseModel

_spec = get_spec("qwen")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def qwen_base_url() -> str:
    return _spec.default_base_url or ""


def _text_base_url() -> str:
    override = os.getenv("QWEN_LLM_API_BASE_URL", "").strip()
    if override:
        return override
    vlm_base = os.getenv(_spec.base_url_env or "", "").strip() if _spec.base_url_env else ""
    return vlm_base or DASHSCOPE_BASE_URL


def qwen_chat(
    *,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    response_model: type[BaseModel],
    max_tokens: int = 4000,
    text_only: bool = False,
) -> BaseModel:
    if text_only:
        return compat_chat(
            provider_label=_spec.label,
            api_key_env=_spec.api_key_env,
            base_url_env="QWEN_LLM_API_BASE_URL",
            default_base_url=_text_base_url(),
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model=response_model,
            max_tokens=max_tokens,
            alt_api_key_envs=_spec.alt_api_key_envs,
        )
    return compat_chat(
        provider_label=_spec.label,
        api_key_env=_spec.api_key_env,
        base_url_env=_spec.base_url_env,
        default_base_url=_spec.default_base_url or "",
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=max_tokens,
        alt_api_key_envs=_spec.alt_api_key_envs,
    )
