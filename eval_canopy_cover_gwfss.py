from __future__ import annotations

import argparse
import io
import json
import urllib.request

import numpy as np
from PIL import Image

from wheatkit.eval import regression_metrics
from wheatkit.tasks import canopy_cover

ROWS = "https://datasets-server.huggingface.co/rows?dataset=GlobalWheat/GWFSS_v1.0&config=default&split=labelled"


def fetch(n: int = 24):
    total = json.load(urllib.request.urlopen(f"{ROWS}&offset=0&length=1", timeout=60))["num_rows_total"]
    for i in range(n):
        off = int(i * (total - 1) / (n - 1))
        row = json.load(urllib.request.urlopen(f"{ROWS}&offset={off}&length=1", timeout=60))["rows"][0]["row"]
        img = Image.open(io.BytesIO(urllib.request.urlopen(row["image"]["src"], timeout=60).read())).convert("RGB")
        mask = Image.open(io.BytesIO(urllib.request.urlopen(row["mask"]["src"], timeout=60).read())).convert("RGB")
        yield off, np.asarray(img), np.asarray(mask).max(axis=2) >= 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["exg", "exg_otsu"])
    args = ap.parse_args()
    data = list(fetch())
    halves = {"development": data[0::2], "held_out": data[1::2]}
    report = {}
    for m in args.methods:
        report[m] = {}
        for name, subset in halves.items():
            ious, t, p = [], [], []
            for _, img, truth in subset:
                pred = canopy_cover(img, method=m).mask
                u = (pred | truth).sum()
                ious.append((pred & truth).sum() / u if u else 1.0)
                t.append(truth.mean())
                p.append(pred.mean())
            report[m][name] = {"iou": float(np.mean(ious)), **regression_metrics(t, p)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
