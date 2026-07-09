# Research Materials

| Path | Description |
|------|-------------|
| `references.bib` | BibTeX bibliography (22 references covering C-MAPSS, Brayton cycle, EKF, conformal prediction, tree ensemble methods, SHAP) |
| `figures/` | Publication-quality figures for paper/poster |
| `experiments/` | Experiment configs and run logs |
| `ablations/` | Subsystem contribution ablation studies |

## Quick Links

- **[Chapter 2: Theory](../docs/Theory.md)** — all methods referenced here
- **[Chapter 6: Validation](../docs/Validation.md)** — quantitative model comparison with per-target metrics
- **Paper draft** — available as `paper/paper.md`
- **Poster** — available as `poster/poster.md`

## External Validation

The project currently validates on a proprietary engine dataset with 300 engines, 14 sensor channels, and 6 health targets. For public benchmarking against C-MAPSS (FD001–FD004), see the validation roadmap in [Chapter 6: Validation](../docs/Validation.md#roadmap).

## Citation

```bibtex
@misc{dtwin2025,
  title = {Turbojet Digital Twin},
  howpublished = {\url{https://github.com/anomalyco/turbojet-dtwin}},
  year = {2025}
}
```

## Figures Checklist

- [ ] T–s diagram of Brayton cycle with station labels
- [ ] Engine cross-section with annotated components
- [ ] Health indicator evolution over life (all 4 subsystems)
- [ ] RUL prediction vs actual (with conformal intervals)
- [ ] SHAP summary plot for top-10 features per target
- [ ] Confusion matrix for failure mode classification
- [ ] Model comparison bar chart (RMSE per target, all models)
