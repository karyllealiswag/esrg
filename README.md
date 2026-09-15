# ESRG — Enhanced Seeded Region Growing for MRI Brain Tumor Segmentation

An enhancement of the Adams & Bischof (1994) Seeded Region Growing algorithm for
localization and delineation of brain tumors on 2D contrast-enhanced T1-weighted MRI.

Three enhancements over the classical algorithm:
1. Automatic, deterministic seed selection (Objective 1).
2. A local log-domain difference measure, insensitive to the MRI bias field (Objective 2).
3. An adaptive stopping criterion that halts growth at weak boundaries (Objective 3).

## Install and run

```bash
pip install -r requirements.txt
python -m gui.app                       # desktop app (per-stage inspection)
python tests/test_pipeline.py           # unit tests (or: pytest tests/)
```

Batch evaluation and analysis:

```bash
python -m experiments.run_batch      --root <split> --method esrg   # scores + CSV
python -m experiments.run_attribution --root <split>                # per-stage failure buckets
python -m experiments.ablation       --root <split>                 # Objective 2/3 ablation
```

## Layout

```
esrg/
  config.py           All parameters, frozen after TRAIN tuning.
  io_utils.py         Image/mask loading, BRISC filename parsing.
  preprocessing.py    Normalize, head mask, N4 (ablation), log transform, noise floor.
  seeds.py            Objective 1 — automatic seed selection, hard-fail on no candidate.
  seed_ranking.py     Phase 1 — contrast/shape candidate ranking (contrast weight 0.5).
  anatomy_exclusion.py Phase 2a — orbit/skull-base exclusion (ablation, off by default).
  growing.py          Objectives 2 & 3 — grow_esrg; grow_srg baseline.
  postprocess.py      Hole fill, opening, seed-connected component.
  metrics.py          DSC, IoU, precision, recall, HD95, ASSD, seed hit, leakage.
  visualize.py        Stage rendering and annotated overlays.
  pipeline.py         Stage orchestration; retains every intermediate.
gui/app.py            Desktop app with one inspection button per stage.
experiments/          run_batch, run_attribution, attribution, ablation.
tests/                Correctness and smoke tests.
sample_data/          BRISC slices with masks for a quick trial.
```

## Findings that shaped the final configuration

Measured on the full 860-image BRISC test split:

- Meningioma delineates reliably and automatically (DSC median 0.86, seed-hit 81%).
- Pituitary delineates well when seeded (DSC 0.77) but automatic seeding is
  unreliable (33%); use manual seeding for these cases.
- Glioma is the honest limit of intensity-based region growing (seed-correct DSC 0.44),
  since the tumor is heterogeneous with weak infiltrative edges.
- Three targeted improvements — contrast/shape ranking, orbit exclusion, and per-image
  adaptive stopping — were implemented and evaluated; none beat the tuned baseline,
  indicating the configuration approaches the practical ceiling of this method class.
  They remain as switchable ablation options.

## Notes

- Stage 2 masks the HEAD, not the brain: on coronal/sagittal slices the brain cannot be
  separated from face/neck, so interior filtering is delegated to seed selection.
- Deep learning is deliberately out of scope (interpretability, no training, CPU-only);
  it belongs in future work.
