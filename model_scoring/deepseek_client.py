"""Shared DeepSeek API helpers (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import os
from base64 import b64encode
from typing import Any, Callable

import typer
from live_api_guard import block_live_inference
from pydantic import BaseModel

ResponseModel = type[BaseModel]

DEFAULT_BASE_URL = "https://api.deepseek.com"


def deepseek_base_url() -> str:
    return os.environ.get("DEEPSEEK_API_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL


def _openai_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("DEEPSEEK_API_KEY is required.")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "openai is required when provider=deepseek. Run this entrypoint with uv run --script."
        ) from exc
    return OpenAI(api_key=api_key, base_url=deepseek_base_url())


def extract_json_object(text: str) -> str:
    import re

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return stripped[start : end + 1]


def _schema_suffix(response_model: type[BaseModel]) -> str:
    schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "\n\nReturn JSON only. Do not include markdown fences.\n"
        f"JSON schema:\n{schema_text}"
    )


def _parse_response(text: str | None, response_model: type[BaseModel]) -> BaseModel:
    if not text:
        raise ValueError("Model returned an empty response.")
    return response_model.model_validate_json(extract_json_object(text))


def deepseek_chat(
    *,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    response_model: type[BaseModel],
    max_tokens: int = 4000,
) -> BaseModel:
    block_live_inference()
    client = _openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return _parse_response(response.choices[0].message.content, response_model)


def image_content_part(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    b64 = b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def build_vision_user_content(
    *,
    rows: list[dict],
    preamble: str,
    response_model: type[BaseModel],
    image_mime_type_fn: Callable[[bytes], str],
    row_prefix_fn: Callable[[dict], str] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": preamble}]
    for row in rows:
        prefix = (
            row_prefix_fn(row)
            if row_prefix_fn is not None
            else f"Target figure image for fig_index={row['fig_index']}"
        )
        content.append({"type": "text", "text": prefix})
        content.append(
            image_content_part(
                row["image_bytes"],
                image_mime_type_fn(row["image_bytes"]),
            )
        )
    content.append({"type": "text", "text": _schema_suffix(response_model)})
    return content
