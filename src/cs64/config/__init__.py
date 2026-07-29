"""Canonical constants for the cs64 package.

This is the single source of truth for shared constants. Other modules in this
codebase that cannot import from cs64 (PEP 723 standalone scripts, Streamlit
bare-import apps) keep a local copy; those copies should stay in sync with
this definition.
"""

# The five evaluation dimensions used throughout scoring, annotation, and reporting.
DIMENSIONS: list[str] = [
    "visual_clarity",
    "structure_layout",
    "caption_consistency",
    "context_consistency",
    "misleading_risk",
]
