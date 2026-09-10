Changes from v2 → v3: all patches are based on failure patterns identified in annotation data. The existing rubric structure is completely unchanged; new rules are inserted at the end of the corresponding dimension sections only.

**P3 — Results figure caption missing metric/legend (highest frequency, 18 hits)**
In the `caption_consistency` section, add a mandatory rule: when the figure is a results or comparison figure, the caption must include both the evaluation metric name (e.g. F1, accuracy) and the comparison group name (model name, dataset name, etc.) to score 7 or above. Captions that only name the experimental variable (e.g. "Effect of K") but provide no metric and no legend are forced to 5–6. This is the highest-frequency pattern and the rule is worded most firmly.

**P2 — Dense figures with small fonts incorrectly penalised for low clarity (2 hits)**
In the `visual_clarity` section, add a dense figure rule: small but readable labels in Sankey diagrams, heatmaps, multi-panel figures and similar dense layouts are normal design — scores must not be pushed below 7 on this basis alone. A score of 5 or below is only warranted when text remains unrecognisable even after zooming (truly blurry, character overlap, low contrast).

**P1 — Bare references that merely name the figure treated as complete context (1 hit)**
In the `context_consistency` section, add a rule: if the body context consists of only one or two sentences that merely name the figure (e.g. "We provide … in Figure X") without discussing the figure's structure, trends, or conclusions, force a score of 5–6; do not exceed 7. Topical relevance does not equal content consistency.

**P4 — Reason text overstates degree of alignment (soft patch)**
In the General Scoring Rules section, add wording constraints: absolute adjectives such as "perfectly / fully / completely / accurately" are prohibited when alignment is incomplete; use phrases such as "consistent with" or "covers the main topic" instead. This does not change scores, but makes reason text more accurate and easier to verify in round 2 annotation.
