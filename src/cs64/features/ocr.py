"""OCR backend abstraction.

The first iteration supports a `none` backend so the feature pipeline can ship
before OCR dependencies are finalized.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np

OCRBackendName = Literal["none", "paddleocr"]

STOPWORDS_PATH = Path(__file__).with_name("stop_words_english.txt")


def _load_stopwords(path: Path = STOPWORDS_PATH) -> set[str]:
    words: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        word = raw_line.strip().lower()
        if not word:
            continue
        words.add(word)
    return words


STOPWORDS = _load_stopwords()

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")
NUMERIC_TOKEN_RE = re.compile(
    r"""
    (?<![A-Za-z])
    [+-]?
    (?:
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    %?
    """,
    flags=re.VERBOSE,
)
PANEL_LABEL_RE = re.compile(r"^\(?[a-zA-Z]\)?$")


@dataclass(slots=True)
class OCRFeatures:
    has_readable_text: bool | None
    ocr_mean_confidence: float | None
    text_box_count: int | None
    text_region_ratio: float | None
    panel_label_count: int | None
    caption_ocr_token_overlap_ratio: float | None
    caption_numeric_overlap_ratio: float | None


class BaseOCRBackend:
    name: str

    def extract(self, image_rgb: np.ndarray, caption: str) -> OCRFeatures:
        raise NotImplementedError


class NoneOCRBackend(BaseOCRBackend):
    name = "none"

    def extract(self, image_rgb: np.ndarray, caption: str) -> OCRFeatures:
        _ = image_rgb, caption
        return OCRFeatures(
            has_readable_text=None,
            ocr_mean_confidence=None,
            text_box_count=None,
            text_region_ratio=None,
            panel_label_count=None,
            caption_ocr_token_overlap_ratio=None,
            caption_numeric_overlap_ratio=None,
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set[str]:
    tokens = {m.group(0).lower() for m in TOKEN_RE.finditer(text)}
    return {token for token in tokens if token not in STOPWORDS}


def _extract_numeric_tokens(text: str) -> set[str]:
    return {m.group(0) for m in NUMERIC_TOKEN_RE.finditer(text)}


def _is_panel_label(text: str) -> bool:
    stripped = text.strip()
    return bool(PANEL_LABEL_RE.fullmatch(stripped))


def _polygon_area(points: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


class PaddleOCRBackend(BaseOCRBackend):
    name = "paddleocr"

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault(
            "PADDLE_OCR_BASE_DIR",
            str(project_root / ".cache" / "paddleocr"),
        )

        if platform.system() == "Windows":
            # PaddleOCR imports `paddle` first, which can disturb the DLL search
            # state enough that a later `import torch` fails on Windows. Preloading
            # torch keeps the transitive albumentations -> torch import stable.
            import torch  # noqa: F401

        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "paddleocr backend requested, but PaddleOCR is not "
                "installed in the current environment."
            ) from exc

        self.engine = PaddleOCR(lang="en", use_textline_orientation=True)

    def extract(self, image_rgb: np.ndarray, caption: str) -> OCRFeatures:
        raw_result = self.engine.ocr(image_rgb, cls=False)
        lines = raw_result[0] if raw_result else []
        items: list[dict[str, object]] = []
        for line in lines or []:
            if not line or len(line) != 2:
                continue
            box, payload = line
            if not payload or len(payload) != 2:
                continue

            text, confidence = payload
            normalized_text = _normalize_text(str(text))
            if not normalized_text:
                continue
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = 0.0
            items.append({"box": box, "text": normalized_text, "conf": conf})

        if not items:
            return OCRFeatures(
                has_readable_text=False,
                ocr_mean_confidence=None,
                text_box_count=0,
                text_region_ratio=0.0,
                panel_label_count=0,
                caption_ocr_token_overlap_ratio=None,
                caption_numeric_overlap_ratio=None,
            )

        image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
        total_text_area = sum(_polygon_area(item["box"]) for item in items)
        ocr_text = " ".join(str(item["text"]) for item in items)
        ocr_tokens = _tokenize(ocr_text)
        caption_tokens = _tokenize(caption)
        ocr_numeric = _extract_numeric_tokens(ocr_text)
        caption_numeric = _extract_numeric_tokens(caption)

        panel_label_count = sum(1 for item in items if _is_panel_label(str(item["text"])))
        token_overlap = _ratio(len(caption_tokens & ocr_tokens), len(caption_tokens))
        numeric_overlap = _ratio(len(caption_numeric & ocr_numeric), len(caption_numeric))

        return OCRFeatures(
            has_readable_text=True,
            ocr_mean_confidence=float(mean(float(item["conf"]) for item in items)),
            text_box_count=len(items),
            text_region_ratio=min(total_text_area / image_area, 1.0) if image_area else None,
            panel_label_count=panel_label_count,
            caption_ocr_token_overlap_ratio=token_overlap,
            caption_numeric_overlap_ratio=numeric_overlap,
        )


def get_ocr_backend(name: OCRBackendName = "paddleocr") -> BaseOCRBackend:
    if name == "none":
        return NoneOCRBackend()
    if name == "paddleocr":
        return PaddleOCRBackend()
    raise ValueError(f"Unsupported OCR backend: {name}")
