# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Code carries no comments or docstrings; documentation lives in `README.md`,
  `docs/`, `PLAN.md` and `benchmarks/results/`.
- Linting uses pinned black 24.10.0 and flake8 7.1.1 (CI and pre-commit).
- `canopy_cover` default is now `exg`: 5×5-smoothed Excess Green, with Otsu used
  only when the histogram is clearly bimodal and a fixed 0.02 threshold otherwise.
  On a held-out GWFSS sample, plant IoU rose from 0.485 to 0.637 and cover MAE fell
  from 0.33 to 0.13 compared with the previous `exg_otsu` default
  (`benchmarks/results/gwfss_canopy_cover_sample.md`).

### Added
- Verified `segment_organs` end to end with the GWFSS SegFormer-B1. On a 24-image
  sanity check it scored 0.78 mIoU (stem 0.53) at 1.3 s per image on CPU; training-set
  overlap is likely, so this is not a benchmark. Added the README figure and a
  network-marked test.
- Dataset registry with 42 wheat image datasets. Each entry has tier, licence group,
  licence conflicts across mirrors, and pinned Zenodo md5s.
- Licence-gated, checksum-verified downloads from Zenodo and Hugging Face.
- GWHD readers, dHash near-duplicate grouping, group and leave-one-group-out splits,
  and a leakage assertion.
- Evaluation: Global Wheat Average Domain Accuracy, per-domain count metrics, mIoU.
- Split-conformal and per-group prediction intervals.
- Tasks: canopy cover (ExG+Otsu), GWFSS SegFormer organ segmentation wrapper, and
  tiled head counting with any detector.
- torchvision detector fine-tuning loop.
- Trait tables, thermal time, and logistic event dating for heading and senescence.
- SILO and NASA POWER clients.
- Tiered decision support:
  - cited rules (stripe rust in AU, septoria T1/T2 in UK, FHB weather);
  - safeguards against low-confidence and out-of-distribution inputs;
  - outcome ranges;
  - breeder ranking with P(top k).
- BrAPI v2 and MIAPPE-style exports, and a CLI.
- Out-of-distribution detection (`OODDetector`). It uses model-free image features
  or any embedding, with a conformally calibrated alarm threshold, and feeds
  `Observation.out_of_distribution`.
- Growth-stage fusion (`fuse_stage`): a thermal-time prior is combined with
  image-classifier probabilities into a Zadoks interval and a confidence score;
  `Observation.from_stage_estimate` passes the result to the advisory layer.
- Hourly NASA POWER client and `humid_hours` (the FHB humidity-hour covariate).
- Registry check uses DataCite metadata for DOI-backed entries, including licence
  drift. UMN hyperspectral DOI added.
