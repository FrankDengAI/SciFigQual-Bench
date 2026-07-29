# `scripts/features/` — OCR/CV feature dataset

Build and upload the OCR + visual feature dataset: read `figures_clean.parquet`,
extract ~14 feature columns (OCR confidence, text density, blur,
caption-OCR overlap, panel/box counts, …) into `figures_features.parquet`. The
implementation lives in `src/cs64/features/`; this folder holds the two CLI
entry points. Features feed the scorer's "with-features" workflow.

📖 **Full documentation:** [`wiki/02-scoring/04-features.md`](../../wiki/02-scoring/04-features.md)
— what each signal means, why PaddleOCR, and how features are routed.

> See [`wiki/index.md`](../../wiki/index.md) for the whole project.
