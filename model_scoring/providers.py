from __future__ import annotations

import json
import os
import re
from base64 import b64encode
from pathlib import Path

import typer
from deepseek_client import build_vision_user_content, deepseek_chat
from openai_compat_client import compat_chat
from provider_registry import get_spec
from live_api_guard import block_live_inference
from schema import ResponseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "model_scoring" / "prompts"
PROMPT_PATHS = {
    "baseline": PROMPTS_DIR / "baseline_paper_batch.md",
    "with_features": PROMPTS_DIR / "with_features_paper_batch.md",
}


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


def load_prompt(path: Path | None = None, prompt_mode: str = "baseline") -> str:
    if path is not None:
        target = path
    else:
        target = PROMPT_PATHS[prompt_mode]
    return target.read_text(encoding="utf-8")


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


def user_text(row: dict) -> str:
    return (
        f"Paper title: {row.get('title') or ''}\n"
        f"Caption: {row.get('caption') or ''}\n"
        f"Venue: {row.get('venue')}\n"
        f"Year: {row.get('year')}\n"
        f"Figure index: {row.get('fig_index')}\n"
    )


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


def generate_with_gemini(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("GEMINI_API_KEY is required.")
    genai, genai_types = _gemini_modules()
    client = genai.Client(api_key=api_key)
    mime_type = image_mime_type(row["image_bytes"])
    response = client.models.generate_content(
        model=model,
        contents=[
            genai_types.Part.from_bytes(data=row["image_bytes"], mime_type=mime_type),
            user_text(row) + (f"\n{extra_user_text}" if extra_user_text else ""),
        ],
        config={
            "system_instruction": prompt,
            "response_mime_type": "application/json",
            "response_json_schema": response_model.model_json_schema(),
        },
    )
    return response_model.model_validate_json(response.text)


def generate_with_claude(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY is required.")
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "anthropic is required when provider=claude. Install project dependencies first."
        ) from exc
    client = Anthropic(api_key=api_key)
    mime_type = image_mime_type(row["image_bytes"])
    schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
    full_user_text = user_text(row) + (f"\n{extra_user_text}" if extra_user_text else "")
    response = client.messages.create(
        model=model,
        max_tokens=1400,
        system=prompt,
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
                    {"type": "text", "text": full_user_text},
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
    text_parts = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    return response_model.model_validate_json(extract_json_object("\n".join(text_parts)))


def generate_with_deepseek(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    full_user_text = user_text(row) + (f"\n{extra_user_text}" if extra_user_text else "")
    user_content = build_vision_user_content(
        rows=[row],
        preamble=full_user_text,
        response_model=response_model,
        image_mime_type_fn=image_mime_type,
    )
    return deepseek_chat(
        model=model,
        system_prompt=prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=1400,
    )


def generate_with_qwen(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    return _generate_with_compat_provider(
        "qwen", row, model, prompt, response_model, extra_user_text=extra_user_text, max_tokens=1400
    )


def _generate_with_compat_provider(
    provider_id: str,
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    *,
    extra_user_text: str | None = None,
    max_tokens: int = 1400,
) -> ResponseModel:
    spec = get_spec(provider_id)
    full_user_text = user_text(row) + (f"\n{extra_user_text}" if extra_user_text else "")
    user_content = build_vision_user_content(
        rows=[row],
        preamble=full_user_text,
        response_model=response_model,
        image_mime_type_fn=image_mime_type,
    )
    return compat_chat(
        provider_label=spec.label,
        api_key_env=spec.api_key_env,
        base_url_env=spec.base_url_env,
        default_base_url=spec.default_base_url or "",
        model=model,
        system_prompt=prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=max_tokens,
        alt_api_key_envs=spec.alt_api_key_envs,
    )


def _generate_paper_batch_with_compat_provider(
    provider_id: str,
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
    max_tokens: int = 4000,
) -> ResponseModel:
    spec = get_spec(provider_id)
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
        system_prompt=prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=max_tokens,
        alt_api_key_envs=spec.alt_api_key_envs,
    )


def generate_with_openai(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    return _generate_with_compat_provider(
        "openai", row, model, prompt, response_model, extra_user_text=extra_user_text, max_tokens=1400
    )


def generate_with_doubao(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    return _generate_with_compat_provider(
        "doubao", row, model, prompt, response_model, extra_user_text=extra_user_text, max_tokens=1400
    )


def generate_with_glm(
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    return _generate_with_compat_provider(
        "glm", row, model, prompt, response_model, extra_user_text=extra_user_text, max_tokens=1400
    )


def generate_paper_batch_with_gemini(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("GEMINI_API_KEY is required.")
    genai, genai_types = _gemini_modules()
    contents: list = [user_payload_text]
    for row in rows:
        contents.append(f"Target figure image for fig_index={row['fig_index']}")
        contents.append(
            genai_types.Part.from_bytes(
                data=row["image_bytes"],
                mime_type=image_mime_type(row["image_bytes"]),
            )
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "system_instruction": prompt,
            "response_mime_type": "application/json",
            "response_json_schema": response_model.model_json_schema(),
        },
    )
    return response_model.model_validate_json(response.text)


def generate_paper_batch_with_deepseek(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    user_content = build_vision_user_content(
        rows=rows,
        preamble=user_payload_text,
        response_model=response_model,
        image_mime_type_fn=image_mime_type,
    )
    return deepseek_chat(
        model=model,
        system_prompt=prompt,
        user_content=user_content,
        response_model=response_model,
        max_tokens=4000,
    )


def generate_paper_batch_with_qwen(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    return _generate_paper_batch_with_compat_provider(
        "qwen",
        rows=rows,
        model=model,
        prompt=prompt,
        response_model=response_model,
        user_payload_text=user_payload_text,
    )


def generate_paper_batch_with_openai(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    return _generate_paper_batch_with_compat_provider(
        "openai",
        rows=rows,
        model=model,
        prompt=prompt,
        response_model=response_model,
        user_payload_text=user_payload_text,
    )


def generate_paper_batch_with_doubao(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    return _generate_paper_batch_with_compat_provider(
        "doubao",
        rows=rows,
        model=model,
        prompt=prompt,
        response_model=response_model,
        user_payload_text=user_payload_text,
    )


def generate_paper_batch_with_glm(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    return _generate_paper_batch_with_compat_provider(
        "glm",
        rows=rows,
        model=model,
        prompt=prompt,
        response_model=response_model,
        user_payload_text=user_payload_text,
    )


def generate_paper_batch_with_claude(
    *,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY is required.")
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            "anthropic is required when provider=claude. Install project dependencies first."
        ) from exc
    schema_text = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
    content: list[dict] = [{"type": "text", "text": user_payload_text}]
    for row in rows:
        content.extend(
            [
                {"type": "text", "text": f"Target figure image for fig_index={row['fig_index']}"},
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
                f"Return JSON only. Do not include markdown fences.\nJSON schema:\n{schema_text}"
            ),
        }
    )
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=prompt,
        messages=[{"role": "user", "content": content}],
    )
    text_parts = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    return response_model.model_validate_json(extract_json_object("\n".join(text_parts)))


def generate_paper_batch_with_provider(
    *,
    provider: str,
    rows: list[dict],
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    user_payload_text: str,
) -> ResponseModel:
    block_live_inference()
    if provider == "gemini":
        return generate_paper_batch_with_gemini(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "claude":
        return generate_paper_batch_with_claude(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "deepseek":
        return generate_paper_batch_with_deepseek(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "qwen":
        return generate_paper_batch_with_qwen(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "openai":
        return generate_paper_batch_with_openai(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "doubao":
        return generate_paper_batch_with_doubao(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    if provider == "glm":
        return generate_paper_batch_with_glm(
            rows=rows,
            model=model,
            prompt=prompt,
            response_model=response_model,
            user_payload_text=user_payload_text,
        )
    raise typer.BadParameter(f"Unsupported provider: {provider}")


def generate_with_provider(
    provider: str,
    row: dict,
    model: str,
    prompt: str,
    response_model: type[ResponseModel],
    extra_user_text: str | None = None,
) -> ResponseModel:
    block_live_inference()
    if provider == "gemini":
        return generate_with_gemini(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "claude":
        return generate_with_claude(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "deepseek":
        return generate_with_deepseek(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "qwen":
        return generate_with_qwen(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "openai":
        return generate_with_openai(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "doubao":
        return generate_with_doubao(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    if provider == "glm":
        return generate_with_glm(
            row,
            model,
            prompt,
            response_model,
            extra_user_text=extra_user_text,
        )
    raise typer.BadParameter(f"Unsupported provider: {provider}")
