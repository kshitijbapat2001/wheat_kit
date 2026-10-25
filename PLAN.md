# wheatkit: build and launch plan

> **Mission.** Become the open, trusted *observation layer* for wheat. Turn public and
> private wheat imagery into plot-level traits with honest error bars. Then hand those
> observations to the breeding databases and decision tools people already use.

This plan comes from a structured review of about 60 public wheat image datasets, the
2020–2026 model literature, existing wheat decision-support systems, and evidence on
what makes scientific open-source tools get adopted. The full research report, with
sources, is in [docs/research/wheat-image-ml-research-report.md](docs/research/wheat-image-ml-research-report.md).

---

## 1. Why this project, why now

| Finding | Implication for wheatkit |
|---|---|
| GWHD 2021 has had **~9,600 unique downloads**, yet the official GWFSS models get **67–164 downloads/month** and the best wheat Hugging Face demo has **4 likes** | Demand for wheat *data* far exceeds use of wheat *models*. Ship models with demos. |
| AgML (the nearest complement) gets **~350 PyPI downloads/month**; it has one model and no demo apps | The gap is pretrained models, demos and trait extraction, not another data loader. Integrate with AgML; don't compete. |
| DeepForest's tree model got **140,852 Hugging Face downloads in 30 days**, against 6,417 PyPI installs | Hosting weights on Hugging Face reaches users who never visit GitHub. Make HF downloads the north-star metric. |
| Head detectors fall from **93% → 74% mAP@50** on GWHD's unseen-region split | Default to honest splits; that is our credibility. |
| Licences conflict across mirrors (GWHD: CC BY on Zenodo, MIT in the paper, CC BY-SA on AgML, MIT on Roboflow) | A machine-readable licence manifest is a real, unique feature. |
| No peer-reviewed wheat system feeds image-derived severity or growth stage into fungicide decision tools | The observation layer for tools like StripeRustWM, Yield Prophet and xarvio is an open niche. |
| No conformal-prediction study exists for wheat counting or traits | Cheap to add, and it sets wheatkit apart. |

## 2. Users and the jobs they need done

| User | Job | Feature that wins them |
|---|---|---|
| Phenotyping platforms (APPN nodes, breeding companies) | Turn image runs into per-plot traits for genetics | `traits` + `io.brapi` export, conformal intervals, plot-level aggregation |
| Breeders | Rank lines using image traits that are actually heritable | `decide.rank_lines` (heritability filter, heading-date correction, P(top k)) |
| Agronomy and extension researchers | Monitor crops; link observations to spray windows | `decide.advise` (cited rules), growth-stage and heading dating |
| ML researchers | Fair benchmarks on wheat | Registry, official splits, ADA metric, leaderboard |
| DSS developers | A reliable observation feed | Stable API, model cards with validity ranges |

## 3. Status of this repository (v0.1.0.dev0)

| Module | What works today | Tests |
|---|---|---|
| `data.registry` | 42 datasets, tiers, licence groups, conflicts, pinned md5s | ✅ |
| `data.fetch` | Licence-gated Zenodo/HF downloads, resume, md5 check, dry-run | ✅ (mocked network) |
| `data.gwhd`, `dedup`, `splits` | GWHD CSV/HF readers, dHash duplicate groups, group/LOGO splits, leakage assertion | ✅ |
| `eval` | IoU, greedy matching, **Average Domain Accuracy**, per-group count metrics, mIoU | ✅ |
| `uncertainty` | Split-conformal and per-group (Mondrian) intervals, coverage | ✅ |
| `ood` | Out-of-distribution flag from image features or embeddings, conformal threshold | ✅ |
| `tasks` | Canopy cover (smoothed ExG, held-out IoU 0.64), GWFSS SegFormer wrapper (verified: 0.78 mIoU sanity check, 1.3 s/image CPU), tiled head counting with any detector | ✅ (SegFormer test is network-marked, skipped in CI) |
| `train.detector` | torchvision Faster R-CNN/RetinaNet fine-tuning loop | ✅ (1-step CPU smoke test) |
| `traits`, `phenology` | Tidy trait tables, image aggregation, density, thermal time, logistic event dating, **thermal-time + image stage fusion** | ✅ |
| `env` | NASA POWER (daily and hourly) and SILO clients with provenance; FHB humidity hours | ✅ (parsing) |
| `decide` | Tier 1–2 advisory with cited rules and safeguards, tier-3 outcome ranges, breeder ranking | ✅ |
| `io` | BrAPI v2 observations, MIAPPE-style wide table | ✅ |
| `cli` | `datasets list/info/download`, `cover`, `advise` | ✅ |
| Released model weights | **None yet**: this is milestone v0.1 | — |

## 4. Architecture and design principles

```
src/wheatkit/
  data/       registry.yaml · fetch (licence gate, md5) · gwhd · dedup · splits
  eval/       detection (ADA) · counting (per-domain) · segmentation (mIoU)
  tasks/      canopy · organs · heads (task API: 1–3 lines for biologists)
  train/      detector (torchvision, permissive)
  traits.py   tidy observation tables (MIAPPE-shaped)
  phenology.py thermal time · logistic event dating
  uncertainty.py split-conformal, per region
  env/        SILO · NASA POWER (+ Open-Meteo, SLGA, STAC planned)
  decide/     advisory (tiers 1–2) · economics (tier 3) · ranking (breeders)
  io/         BrAPI · MIAPPE
```

1. **Wrap, don't rebuild.** Use torchvision, timm, 🤗 transformers/datasets, SAHI,
   supervision and torchgeo. Ultralytics stays an optional extra because its AGPL-3.0
   licence extends to trained weights.
2. **The honest split is the default.** Random splits exist only as labelled diagnostics.
3. **Provenance travels with every output:** dataset licence group, model version,
   interval bounds.
4. **Core installs without a GPU or the internet;** heavy dependencies are extras.
5. **Describe before advising.** Every rule is cited, region-scoped and rate-free.

## 5. Data strategy

- **Tiers.** Tier 1 loads by default; tier 2 is opt-in with a printed reason; tier 3 is
  refused unless the user passes `allow_tier3`.
- **Licence groups decide what model weights can be:**
  - **permissive** (CC BY, CC0, MIT) → Apache-2.0 weights;
  - **share-alike** → flagged;
  - **non-commercial** → weights marked non-commercial.
- **Hygiene rules, enforced in loaders:**
  1. Hash-deduplicate across *all* pooled datasets before splitting.
  2. Strip pre-augmented copies (Yellow-Rust-19, WGSP, Bangladesh set).
  3. Split by plot, leaf, farmer or session, never by image.
  4. Publish known label issues on each dataset card.
- **Maintenance.** A monthly CI job re-queries Zenodo, HF and DataCite to detect dead
  links, licence changes and file changes (`scripts/check_registry.py`, backlog item).
- **Settle the open licence questions** by emailing the maintainers: GWHD 2021
  (CC BY vs MIT), Anderegg 2026, Zenkl 2025, RootNav 2.0 and WGSP.
- **Contribute data back.** Releasing even a modest, permissively licensed Australian UAV
  dataset with per-plot yield across sites would fill the biggest open gap worldwide.
  It would also make wheatkit somewhere people come for data. APPN nodes are the
  natural source.

## 6. Model plan

| Task | Training data (tier 1) | Default model (permissive) | Stretch | Honest protocol and v-target |
|---|---|---|---|---|
| Head detection and counting | GWHD 2021, Madec WED, CFWED | torchvision Faster R-CNN v2 / HF RT-DETR + centre-point counter | session-split ensembles, test-time augmentation, box fusion, pseudo-labels (the Global Wheat Challenge winners' recipe); FoMo4Wheat backbone | GWHD official test: ADA, mAP@50:95, per-domain count rRMSE and bias, ≥3 seeds. v0.1 target ≥ 70 mAP@50 |
| Organ segmentation | GWFSS (CC BY subset) | SegFormer-B1/B4 (existing GWFSS weights) | Mask2Former + ViT-Adapter on FoMo4Wheat; distil to B1 | GWFSS test mIoU + stem IoU (~0.71 → 0.75) |
| Canopy cover | EWS*, GWFSS | ExG+Otsu baseline → SegFormer-B0 | VegAnn-style multi-source training | cross-dataset IoU |
| Growth stage and heading | FIP 1.0, WheatPhenology | ConvNeXt via timm + thermal-time prior | temporal model over per-date embeddings | heading date error in days, leave-one-year-out |
| Disease severity | NWRD, UMN FHB, (Anderegg*, Zenkl* once licences are confirmed) | U-Net/SegFormer lesion masks | foundation-model backbone | % area error vs visual scores; JIC-999 as external test |
| Plot traits | FIP 1.0 | LightGBM on image traits + thermal time | transformer on DINOv2/FoMo4Wheat embeddings | R²/RMSE on unseen genotypes, environments and GxE; **first published comparison of foundation-model features vs hand-built features for wheat yield** |

\* share-alike or licence pending.

**Compute.** Every v0.x default trains on a single 24 GB GPU in under a day.
Foundation-model *pretraining* (about 9,000 A100-hours for FoMo4Wheat-G) is out of
scope. Use frozen backbones and distillation instead.

## 7. Decision support plan

| Tier | Ships in | Gate to release |
|---|---|---|
| 1 Describe (stage, severity, cover, density ± interval) | v0.1 | Interval coverage published per region |
| 2 Cited rules (stripe rust Z31/Z39 AU; septoria T1/T2 UK; FHB weather around flowering) | v0.1 (library) → v0.5 (fed by models) | Every rule has a source, a region and an agronomist reviewer. Alerts fire only when a growth-stage window is open, risk changes and the model is confident |
| 3 Outcome ranges and breeder ranking | v0.5–0.8, opt-in | Regional calibration; P(loss) shown; validated on later years and other sites |
| 4 Learned policies (RL/bandits) | never as advice | Simulation research only |

**Pilot (v0.6).** Run with one APPN node or National Variety Trial site for a full
season. Log every alert and whether the agronomist confirmed it. Publish precision and
recall per region and season.

## 8. Benchmarks and leaderboard

Papers with Code was reportedly retired in 2025, so the field lacks a default place to
report wheat results. `benchmarks/` defines tasks on official splits (GWHD ADA, GWFSS
mIoU, FIP leave-environment-out). Results are submitted as PRs containing a predictions
file and a model card. CI recomputes the metric. Invite the Global Wheat consortium to
co-host.

## 9. Roadmap (18 months)

| Phase | Months | Build | Launch and community | Exit criteria |
|---|---|---|---|---|
| **0 Foundations** ✅ mostly done | 0–2 | This skeleton; registry; CI; licence gate | Reserve PyPI, HF and GitHub names; contact Global Wheat and AgML; recruit a 2nd maintainer | CI green; tier-1 manifest complete |
| **1 First models (v0.1)** | 2–5 | GWHD detector + per-domain eval; SegFormer canopy/organs; FIP canopy cover and head density; HF model cards | **Public launch:** GIF README, Colab per task, HF Spaces demo, Zenodo DOI, posts on X, Bluesky, LinkedIn and HN; partner announcements | Results reproduced on official splits; JOSS 6-month clock running |
| **2 Traits and time (v0.2–0.4)** | 5–10 | Heading/senescence dating on FIP; growth stage; disease severity; orthomosaic plot extraction (ODM → torchgeo/SAHI); conformal everywhere; FoMo4Wheat/DINOv2 adapters with low-label learning curves | Starter issues with fast reviews; **JOSS submission (month ≥ 8)**; benchmark paper to *Plant Phenomics* or *Plant Methods*; first workshop tutorial | Leave-year-out trait accuracy and interval coverage published per region |
| **3 Decision support (v0.5–0.8)** | 10–15 | Tiers 1–3 wired to models; breeder module end to end; growth-stage-gated alerts; INVITA opt-in loader under agreement | Season-long pilot with agronomists and breeders; webinar series | Alert precision and recall per region; every rule traceable |
| **4 Scale (v1.0+)** | 15–18+ | Stable 1.0 API; multispectral, hyperspectral and 3D modules; distilled edge models (ONNX); new open Australian dataset | Leaderboard submissions from outside groups; conference tutorials; Hacktoberfest | Independent institutional adopters; outside maintainers |

## 10. Launch and engagement playbook

The evidence: the strongest predictors of a popular academic AI repo are links in the
README, images in the README, having a licence and reproducibility. Promotion brings
stars but few contributors. Responsive maintainers keep newcomers.

**Before launch (checklist)**

- [ ] README hero GIF: a field image going to heads, organs and a trait table (`docs/assets/`)
- [ ] Three Colab notebooks: quick start, train-and-evaluate on GWHD, FIP trait time series
- [ ] Hugging Face org `wheatkit` holding models with machine-readable cards (licence group, training data, per-region metrics, validity range)
- [ ] Gradio demo on HF Spaces (`app/`), linked from the README and the arXiv paper page
- [ ] MkDocs-Material site with executed notebooks
- [ ] Zenodo–GitHub integration for a DOI per release; `CITATION.cff`
- [ ] 15–20 labelled starter issues from `.github/backlog.yaml` (`scripts/create_github_issues.py`)
- [ ] PyPI trusted publishing via `release.yml`

**Launch week**

- Day 1: release v0.1 on PyPI and HF, with a short thread on X, Bluesky and LinkedIn (GIF plus one honest number: "93 → 74 mAP on unseen regions; here's why").
- Days 2–3: posts on Hacker News (Show HN) and r/MachineLearning; posts in the plant-phenotyping communities (IPPN, EPPN, APPN newsletters).
- Days 3–5: joint posts with partners (Global Wheat, AgML, APPN); PRs adding wheatkit to awesome-lists (awesome-agriculture, awesome-plant-phenotyping).

**Sustained cadence**

- A monthly minor release with a changelog post.
- A quarterly registry refresh.
- A response to every issue within 48 hours.
- Tutorials at CVPPP, IPPS, Plant Phenomics and APPN events.
- Integrations contributed upstream (an AgML loader, a SAHI example, a FiftyOne zoo entry). These create the cross-links that drive discovery.

**Broaden the frame.** Present wheatkit as *small, dense organ counting and
segmentation in field images, with wheat as the flagship crop*, so that barley and
other cereal researchers can adopt it.

## 11. Metrics

| Metric | Why it matters | 12-month target |
|---|---|---|
| **HF model downloads / month (north star)** | real usage, including users who never see GitHub | 5,000 |
| PyPI + conda installs / month | library adoption | 2,000 |
| Citing papers (Zenodo DOI + JOSS) | academic credit, funding | 15 |
| Leaderboard submissions from outside groups | becoming the reference benchmark | 5 |
| Institutions using wheatkit | sustainability | 5 |
| External contributors with ≥1 merged PR | community health | 10 |
| GitHub stars | lagging signal (plant-science tools usually plateau in the hundreds) | 500 |

The targets are ambitions, not forecasts. Review them at month 6.

## 12. Governance and sustainability

- At least **two named maintainers** from different institutions before launch. See
  `GOVERNANCE.md`.
- An **institutional home**, for example an APPN node or university lab. Projects tied
  to one paper or competition die; PlantCV (Danforth), torchgeo (its own organisation)
  and AgML (UC Davis AIFS) survive.
- **Funding options:** ARDC, GRDC and APPN/NCRIS project lines, university research
  software engineering funds, and open-source science grants.
- Decisions by lazy consensus among maintainers; Contributor Covenant code of conduct.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Licence ambiguity (GWHD, unverified sets) | Original-record licence plus a conflicts list; email maintainers; GWHD is permissive under either stated licence |
| AGPL contamination | Ultralytics optional only; permissive defaults; weights tagged by framework licence |
| Advice liability | Tier gating, citations, no products or rates, suppression on low confidence, prominent disclaimer, opt-in tiers 3–4 |
| Over-claiming accuracy | Official splits, ≥3 seeds, per-region tables, conformal coverage reported |
| Maintainer burnout | Two or more maintainers, an institutional home, limited scope (wrap, don't rebuild) |
| Foundation models unreplicated (FoMo4Wheat, SPROUT) | Optional backbones, benchmarked with learning curves |
| Dependency decay (MMDetection, detectron2 stagnant) | Depend on torchvision, timm and HF only |

## 14. The next eight weeks

| Week | Do |
|---|---|
| 1 | Publish the repo (Section 15); reserve PyPI and HF names; email Global Wheat about the GWHD licence and a co-hosted leaderboard |
| 2 | `scripts/check_registry.py` (link and licence monitor); HF dataset loaders for GWHD and GWFSS (tier-1 streaming) |
| 3–4 | Train the Faster R-CNN v2 baseline on GWHD 2021 (3 seeds); evaluate with ADA and per-domain count metrics; write the first model card |
| 5 | Add a FIP 1.0 loader and evaluate canopy cover against FIP ground truth |
| 6 | Gradio demo plus the Colab notebooks; record the README GIF |
| 7 | Docs site; open the starter issues; recruit a 2nd maintainer |
| 8 | **v0.1 launch** (Section 10) |

## 15. Publishing to GitHub (browser upload, no command-line login)

1. On github.com, signed in as **kshitijbapat2001**, open <https://github.com/new>.
   - Repository name `wheatkit`, **Public**.
   - Leave *Add a README*, *.gitignore* and *license* **unticked**: the folder already has them.
2. On the empty repository page, click **uploading an existing file**.
3. In Finder, open `wheatkit_upload/` and press **Cmd + Shift + .** to show hidden
   files (`.github`, `.gitignore`, `.flake8`, `.pre-commit-config.yaml`).
4. Press **Cmd + A** and drag everything onto the upload page. There are 93 files
   (under GitHub's 100-file limit for one upload) and the largest is 288 KB.
5. Commit message: `wheatkit 0.1.0.dev0`, then **Commit changes**.
6. Check the **Actions** tab: CI runs lint, tests on Python 3.10–3.13, and a CPU
   training smoke test.

Afterwards:
- Add a PyPI trusted publisher for `release.yml`.
- Turn on the Zenodo GitHub integration.
- Optionally create the starter issues with `scripts/create_github_issues.py`, which
  needs a token.
