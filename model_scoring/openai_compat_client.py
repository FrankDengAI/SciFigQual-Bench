"""Shared OpenAI-compatible chat client for vision + JSON scoring providers."""

from __future__ import annotations

import json
import os
from typing import Any

import typer
from pydantic import BaseModel

from deepseek_client import _parse_response, _schema_suffix, build_vision_user_content, image_content_part
from live_api_guard import block_live_inference


def compat_chat(
    *,
    provider_label: str,
    api_key_env: str,
    base_url_env: str | None,
    default_base_url: str,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    response_model: type[BaseModel],
    max_tokens: int = 4000,
    alt_api_key_envs: tuple[str, ...] = (),
) -> BaseModel:
    block_live_inference()
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        for alt in alt_api_key_envs:
            api_key = os.environ.get(alt, "").strip()
            if api_key:
                break
    if not api_key:
        raise typer.BadParameter(f"{api_key_env} is required for provider={provider_label}.")

    base_url = default_base_url
    if base_url_env:
        base_url = os.environ.get(base_url_env, "").strip() or default_base_url

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            f"openai package is required when provider={provider_label}. "
            "Run scoring scripts with uv run --script."
        ) from exc

    schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
    system = (
        f"{system_prompt}\n\n"
        "You must respond with one JSON object only. "
        "Do not include markdown fences or extra commentary.\n"
        f"JSON schema:\n{schema_text}"
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    try:
        response = client.chat.completions.create(
            **request_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(**request_kwargs)
    choices = response.choices
    if not choices:
        raise ValueError(
            f"{provider_label} model={model} returned no choices "
            f"(base_url={base_url}). Check model availability on this endpoint."
        )
    return _parse_response(choices[0].message.content, response_model)


__all__ = [
    "build_vision_user_content",
    "image_content_part",
    "compat_chat",
    "_schema_suffix",
]
