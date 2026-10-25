from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from wheatkit.data.gwhd import load_gwhd_hf
from wheatkit.eval import average_domain_accuracy, metrics_by_group
from wheatkit.tasks import torchvision_detector
from wheatkit.train.detector import TrainConfig, train_detector


def load_split(split: str):
    images, boxes, domains = [], [], []
    for ex in load_gwhd_hf(split):
        images.append(np.asarray(ex["image"].convert("RGB")))
        boxes.append(ex["boxes"])
        domains.append(ex["domain"])
    return images, boxes, domains


def main(cfg_path: str) -> None:
    import torch

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tr_img, tr_box, _ = load_split(cfg["train_split"])
    te_img, te_box, te_dom = load_split(cfg["test_split"])
    results = []
    for seed in cfg["seeds"]:
        torch.manual_seed(seed)
        tc = TrainConfig(
            arch=cfg["model"]["arch"], pretrained_backbone=cfg["model"]["pretrained_backbone"], **cfg["train"]
        )
        model = train_detector(tr_img, tr_box, tc)
        torch.save(model.state_dict(), out / f"model_seed{seed}.pt")
        det = torchvision_detector(model, score_threshold=0.5)
        preds = [det(im) for im in te_img]
        ada, per_domain = average_domain_accuracy(
            [p[0] for p in preds], te_box, te_dom, [p[1] for p in preds], cfg["eval"]["iou_threshold"]
        )
        counts = metrics_by_group([len(b) for b in te_box], [len(p[0]) for p in preds], te_dom)
        results.append(
            {
                "seed": seed,
                "ada": ada,
                "per_domain_accuracy": per_domain,
                "count_metrics": counts.to_dict(orient="records"),
            }
        )
        print(f"seed {seed}: ADA={ada:.3f}")
    adas = [r["ada"] for r in results]
    summary = {
        "ada_mean": float(np.mean(adas)),
        "ada_sd": float(np.std(adas, ddof=1)) if len(adas) > 1 else None,
        "runs": results,
        "config": cfg,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"ADA {summary['ada_mean']:.3f} ± {summary['ada_sd']}  -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train/detector_gwhd.yaml")
