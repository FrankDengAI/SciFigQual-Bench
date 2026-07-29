from __future__ import annotations

import json
import os
import re
import sys
from base64 import b64encode
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel

_AGENT_DIR = Path(__file__).resolve().parent
_MODEL_SCORING_DIR = _AGENT_DIR.parent
if str(_MODEL_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_SCORING_DIR))

from deepseek_client import _schema_suffix, build_vision_user_content, deepseek_chat
from openai_compat_client import compat_chat
from provider_registry import get_spec
from live_api_guard import block_live_inference
from qwen_client import qwen_chat

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    raise ValueError("Unsupported or unreadable image bytes.")


def vision_user_text(row: dict) -> str:
    return f"Figure index: {row.get('fig_index')}\n"


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return stripped[start : end + 1]


def _gemini_modules():
    try:
        from google import genai
        from google.genai import types as genai_types
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "google-genai is required when provider=gemini. "
            "Run this entrypoint with uv run --script."
        ) from exc
    return genai, genai_types


def _gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("GEMINI_API_KEY is required.")
    genai, _ = _gemini_modules()
    return genai.Client(api_key=api_key)


def _anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY is required.")
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "anthropic is required when provider=claude. Install project dependencies first."
        ) from exc
    return Anthropic(api_key=api_key)


def _compat_vision(
    provider: str,
    row: dict,
    model: str,
    system_prompt: str,
    task_text: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    spec = get_spec(provider)
    user_content = build_vision_user_content(
        rows=[row],
        preamble=vision_user_text(row) + "\n" + task_text,
        response_model=response_model,
        image_mime_type_fn=image_mime_type,
    )
    return compat_chat(
        provider_label=spec.label,
        api_key_env=spec.api_key_env,
        base_url_env=spec.base_url_env,
        default_base_url=spec.default_base_url or "",
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=1800,
        alt_api_key_envs=spec.alt_api_key_envs,
    )


def _compat_vision_batch(
    provider: str,
    rows: list[dict],
    model: str,
    system_prompt: str,
    user_payload_text: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    spec = get_spec(provider)
    user_content = build_vision_user_content(
        rows=rows,
        preamble=user_payload_text,
        response_model=response_model,
        image_mime_type_fn=image_mime_type,
    )
    return compat_chat(
        provider_label=spec.label,
        api_key_env=spec.api_key_env,
        base_url_env=spec.base_url_env,
        default_base_url=spec.default_base_url or "",
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=4000,
        alt_api_key_envs=spec.alt_api_key_envs,
    )


def _compat_text(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    spec = get_spec(provider)
    return compat_chat(
        provider_label=spec.label,
        api_key_env=spec.api_key_env,
        base_url_env=spec.base_url_env,
        default_base_url=spec.default_base_url or "",
        model=model,
        system_prompt=system_prompt,
        user_content=user_prompt + _schema_suffix(response_model),
        response_model=response_model,
        max_tokens=1800,
        alt_api_key_envs=spec.alt_api_key_envs,
    )


def generate_vision(
    *,
    provider: str,
    row: dict,
    model: str,
    system_prompt: str,
    task_text: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    block_live_inference()
    if provider == "gemini":
        _, genai_types = _gemini_modules()
        mime_type = image_mime_type(row["image_bytes"])
        client = _gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=row["image_bytes"], mime_type=mime_type),
                vision_user_text(row) + "\n" + task_text,
            ],
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            },
        )
        return response_model.model_validate_json(response.text)

    if provider == "claude":
        mime_type = image_mime_type(row["image_bytes"])
        schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        response = _anthropic_client().messages.create(
            model=model,
            max_tokens=1800,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64encode(row["image_bytes"]).decode("utf-8"),
                            },
                        },
                        {"type": "text", "text": vision_user_text(row) + "\n" + task_text},
                        {
                            "type": "text",
                            "text": (
                                "Return JSON only. Do not include markdown fences.\n"
                                f"JSON schema:\n{schema_text}"
                            ),
                        },
                    ],
                }
            ],
        )
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return response_model.model_validate_json(extract_json_object(text))

    if provider == "deepseek":
        user_content = build_vision_user_content(
            rows=[row],
            preamble=vision_user_text(row) + "\n" + task_text,
            response_model=response_model,
            image_mime_type_fn=image_mime_type,
        )
        return deepseek_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model=response_model,
            max_tokens=1800,
        )

    if provider == "qwen":
        user_content = build_vision_user_content(
            rows=[row],
            preamble=vision_user_text(row) + "\n" + task_text,
            response_model=response_model,
            image_mime_type_fn=image_mime_type,
        )
        return qwen_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model=response_model,
            max_tokens=1800,
        )

    if provider in {"openai", "doubao", "glm"}:
        return _compat_vision(provider, row, model, system_prompt, task_text, response_model)

    raise typer.BadParameter(f"Unsupported provider: {provider}")


def generate_vision_batch(
    *,
    provider: str,
    rows: list[dict],
    model: str,
    system_prompt: str,
    user_payload_text: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    block_live_inference()
    if provider == "gemini":
        _, genai_types = _gemini_modules()
        contents: list = [user_payload_text]
        for row in rows:
            contents.append(f"Target figure image for fig_index={row['fig_index']}")
            contents.append(
                genai_types.Part.from_bytes(
                    data=row["image_bytes"],
                    mime_type=image_mime_type(row["image_bytes"]),
                )
            )
        client = _gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            },
        )
        return response_model.model_validate_json(response.text)

    if provider == "claude":
        schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        content: list[dict] = [{"type": "text", "text": user_payload_text}]
        for row in rows:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Target figure image for fig_index={row['fig_index']}",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_mime_type(row["image_bytes"]),
                            "data": b64encode(row["image_bytes"]).decode("utf-8"),
                        },
                    },
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "Return JSON only. Do not include markdown fences.\n"
                    f"JSON schema:\n{schema_text}"
                ),
            }
        )
        response = _anthropic_client().messages.create(
            model=model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return response_model.model_validate_json(extract_json_object(text))

    if provider == "deepseek":
        user_content = build_vision_user_content(
            rows=rows,
            preamble=user_payload_text,
            response_model=response_model,
            image_mime_type_fn=image_mime_type,
        )
        return deepseek_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model=response_model,
            max_tokens=4000,
        )

    if provider == "qwen":
        user_content = build_vision_user_content(
            rows=rows,
            preamble=user_payload_text,
            response_model=response_model,
            image_mime_type_fn=image_mime_type,
        )
        return qwen_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model=response_model,
            max_tokens=4000,
        )

    if provider in {"openai", "doubao", "glm"}:
        return _compat_vision_batch(
            provider, rows, model, system_prompt, user_payload_text, response_model
        )

    raise typer.BadParameter(f"Unsupported provider: {provider}")


def generate_text(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModel],
) -> ResponseModel:
    block_live_inference()
    if provider == "gemini":
        client = _gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            },
        )
        return response_model.model_validate_json(response.text)

    if provider == "claude":
        schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        response = _anthropic_client().messages.create(
            model=model,
            max_tokens=1800,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        "Return JSON only. Do not include markdown fences.\n"
                        f"JSON schema:\n{schema_text}"
                    ),
                }
            ],
        )
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return response_model.model_validate_json(extract_json_object(text))

    if provider == "deepseek":
        return deepseek_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_prompt + _schema_suffix(response_model),
            response_model=response_model,
            max_tokens=1800,
        )

    if provider == "qwen":
        return qwen_chat(
            model=model,
            system_prompt=system_prompt,
            user_content=user_prompt + _schema_suffix(response_model),
            response_model=response_model,
            max_tokens=4000,
            text_only=True,
        )

    if provider in {"openai", "doubao", "glm"}:
        return _compat_text(provider, model, system_prompt, user_prompt, response_model)

    raise typer.BadParameter(f"Unsupported provider: {provider}")
