# Canopy cover baseline on a GWFSS sample (2026-10-22)

**Question:** how well do model-free colour-index methods separate plant from
background on real, diverse wheat images?

**Data:** 24 labelled images from GWFSS v1.0 (Global Wheat consortium, CC BY 4.0),
taken at evenly spaced offsets through the 1,096-image labelled split so that they span
several sites. Ground truth is "plant" = any organ class (head, stem, leaf), including
senescent tissue; "background" = soil, weeds and other material.

**Protocol:** even-indexed images (n = 12) were used to choose settings; odd-indexed
images (n = 12) are the held-out set reported below. Reproduce with
`python scripts/eval_canopy_cover_gwfss.py`.

| Method | Held-out IoU (plant) | Cover MAE | Cover bias | Cover R² |
|---|---|---|---|---|
| `exg` (default: 5×5-smoothed ExG; Otsu only if bimodal, η ≥ 0.8; else fixed 0.02) | **0.637** | **0.132** | −0.049 | **0.60** |
| `exg_otsu` (ExG + Otsu, previous default) | 0.485 | 0.328 | −0.156 | −1.00 |

Settings compared on the development half (none chosen on the held-out half):

| Variant | Dev IoU | Held-out IoU |
|---|---|---|
| ExG + Otsu | 0.453 | 0.485 |
| ExG − ExR > 0 (Meyer & Neto 2008) | 0.527 | 0.377 |
| ExG > 0 | 0.661 | 0.664 |
| ExG > 0.02 | 0.666 | 0.635 |
| 5×5 mean(ExG) > 0.02 | 0.672 | 0.637 |
| 5×5 mean(ExG); Otsu if η ≥ 0.8 else 0.02 (**selected**) | 0.672 | 0.637 |

## What failed, and why

- **Otsu on unimodal images.** On near-full canopies (true cover 0.94–0.98), Otsu
  still splits the image in two and reports about 0.2 cover. On a soil-only image it
  reported 0.98 cover.
- **Senescent tissue.** GWFSS labels yellowed organs as plant, but colour indices only
  see green, so cover is under-estimated late in the season. Use the GWFSS SegFormer
  (`method="segformer"`) for senescing crops.
- **Neutral grey or brown backgrounds** sit near ExG ≈ 0. With a fixed 0.02 threshold,
  about a third of pixels in a noisy synthetic grey-soil image were misclassified as
  plant. Smoothing and the bimodality guard mitigate this but do not remove it.

## Caveats

- 12 held-out images from one dataset is a small sample: treat ±0.05 IoU as noise.
- The dataset viewer serves masks as JPEG, so class boundaries are slightly blurred.
  Plant = brightest channel ≥ 64 absorbs most of the compression noise.
- This is a *baseline*. The next benchmark (backlog) evaluates on the full GWFSS test
  split and on EWS, and adds SegFormer.
