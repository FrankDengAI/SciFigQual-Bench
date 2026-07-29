# README figures

Paper figures exported for the GitHub README (Figures 2–4 from the AAAI 2027 submission).

| File | Paper figure | Source |
|------|--------------|--------|
| `fig2_construction_pipeline.png` | Figure 2 — construction pipeline | `Fig2.pdf` |
| `fig3_sfq_agent.png` | Figure 3 — SFQ-Agent | `Fig3.pdf` |
| `fig4_dataset_statistics.png` | Figure 4 — dataset statistics | `Fig4.pdf` |

Regenerate from PDF:

```bash
python scripts/export_readme_figures.py
```

Optional: pass a custom paper directory:

```bash
python scripts/export_readme_figures.py --paper-dir ../../AAAI2027/paper
```
