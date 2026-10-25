from __future__ import annotations

import gradio as gr
import numpy as np

import wheatkit as wk

PALETTE = np.array([[0, 0, 0], [230, 180, 40], [120, 70, 200], [40, 170, 60]], dtype=np.uint8)


def analyse(image: np.ndarray, use_segformer: bool):
    cover = wk.canopy_cover(image)
    rows = [["canopy cover (ExG)", f"{cover.cover:.1%}"]]
    overlay = image.copy()
    overlay[cover.mask] = (0.5 * overlay[cover.mask] + 0.5 * np.array([40, 200, 60])).astype(np.uint8)
    if use_segformer:
        seg = wk.segment_organs(image)
        rows += [[f"{name} fraction (GWFSS SegFormer)", f"{v:.1%}"] for name, v in seg.fractions.items()]
        overlay = (0.55 * image + 0.45 * PALETTE[np.clip(seg.mask, 0, 3)]).astype(np.uint8)
    return overlay, rows


demo = gr.Interface(
    fn=analyse,
    inputs=[
        gr.Image(type="numpy", label="Wheat canopy image (close-range RGB)"),
        gr.Checkbox(value=True, label="Organ segmentation (GWFSS SegFormer-B1, CC BY 4.0)"),
    ],
    outputs=[gr.Image(label="Overlay"), gr.Dataframe(headers=["trait", "value"], label="Traits")],
    title="🌾 wheatkit: wheat canopy traits",
    description="Canopy cover and organ fractions from one image. Research tool; see the model cards for "
    "validity ranges. Not agronomic advice.",
)

if __name__ == "__main__":
    demo.launch()
