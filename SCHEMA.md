# eval1200 JSONL schema

One JSON object per line in `data/eval1200/figures.jsonl`. Gold scores are **dimension-first means** over `annotations[]` (paper Eqs. for dim-mean and gated overall). Null means the dimension was gated out.

| Field | Type | Description |
|-------|------|-------------|
| `figure_id` | string | Unique id, e.g. `0001_2020.acl-main.1__fig0001` |
| `paper_id` | string | Paper identifier in this release |
| `paper_id_norm` | string | Venue-native paper id |
| `fig_index` | int | Figure index in the source PDF |
| `venue` / `year` | string / int | ACL, EMNLP, ICML, or NeurIPS · 2020–2025 |
| `title` | string | Paper title (already public) |
| `caption` | string | Figure caption (may be empty) |
| `section` | string | Nearby section heading |
| `context_texts` | list[str] | Index-resolved citing paragraphs T |
| `context_count` | int | Number of citing paragraphs |
| `domain_l1` / `domain_l2` | string | Topic tags when present |
| `width` / `height` | int | Crop size in pixels |
| `annotator_count` | int | Number of human ratings aggregated |
| `image` | string | Relative path `images/<figure_id>.png` (file present only if listed in `included_images.json`) |
| `human_visual_clarity` | float / null | Mean VC |
| `human_structure_layout` | float / null | Mean SL |
| `human_caption_consistency` | float / null | Mean CC (null if no caption) |
| `human_context_consistency` | float / null | Mean CTX (null if no citing text) |
| `human_misleading_risk` | float / null | Mean MR (higher = lower risk) |
| `human_overall_score` | float / null | Gated mean of available dimensions |
| `annotations` | list[object] | Per-rater scores and evidence-tied rationales |

### `annotations[]` fields

| Field | Type | Description |
|-------|------|-------------|
| `source_type` | string | Always `human` in this file |
| `source_name` | string | Anonymized `Rater_XX` |
| `visual_clarity` … `misleading_risk` | float / null | Integer 1–10 or null under L1 gating |
| `overall_score` | float | That rater's gated overall |
| `summary` / `suggestion` | string | Short overall note |
| `visual_clarity_reason` … `misleading_risk_reason` | string / null | Evidence-tied rationale |
| `annotated_at` | string | Timestamp |

Join keys: `figure_id` or `(paper_id, fig_index)`.
