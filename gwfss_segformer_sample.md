# Organ segmentation sanity check: GWFSS SegFormer-B1 (2026-10-25)

`wheatkit.segment_organs` wraps the Global Wheat consortium's
`GlobalWheat/GWFSS_Segformer_b1` (CC BY 4.0). This run checks that the wrapper works
end to end and puts numbers next to the colour-index baseline. It is **not** a
benchmark (see the caveat).

**Data:** the same 24 GWFSS v1.0 labelled images as
[gwfss_canopy_cover_sample.md](gwfss_canopy_cover_sample.md) (evenly spaced through the
1,096-image labelled split). Masks were mapped to classes by nearest palette colour:
background (0,0,0), head (50,255,132), stem (50,132,255), leaf (214,255,50).
**Hardware:** laptop CPU, about 1.3 s per 512×512 image.

| Class | IoU (all 24) | IoU (12 odd-indexed) |
|---|---|---|
| background | 0.880 | 0.920 |
| head | 0.853 | 0.843 |
| **stem** | **0.533** | **0.540** |
| leaf | 0.856 | 0.837 |
| **mean** | **0.780** | **0.785** |

| Canopy cover (12 odd-indexed images) | MAE | bias | R² |
|---|---|---|---|
| `canopy_cover(method="segformer")` | **0.019** | +0.016 | **0.99** |
| `canopy_cover()` default (ExG) | 0.133 | −0.049 | 0.60 |

![examples](../../docs/assets/organ_segmentation_examples.jpg)

## Caveat: probable training-set overlap

The Hugging Face `labelled` split contains **all** 1,096 labelled GWFSS images,
including the images this model was trained on. These scores are therefore optimistic.
Published GWFSS *test* results for SegFormer-class models are about 0.71 mIoU, and the
2025 challenge winner reached 0.747. The honest benchmark (backlog) needs the official
test split from the ETH Research Collection deposit.

## What it does show

- The wrapper, preprocessing and class mapping are correct: per-class IoUs match the
  published pattern, and the full pipeline runs on a CPU.
- **Stems remain the hard class** (IoU about 0.53), as every GWFSS study reports.
- For canopy cover, the learned model handles senescent tissue and dark soil that
  colour indices cannot. Use `method="segformer"` for anything past flowering.
