import numpy as np
import pytest

from wheatkit.tasks import ModelNotReleasedError, canopy_cover, count_heads, load_detector
from wheatkit.tiling import detect_tiled, nms, tile_windows


def test_canopy_cover_half(green_on_soil):
    r = canopy_cover(green_on_soil)
    assert r.cover == pytest.approx(0.5, abs=0.03)
    assert r.mask[:, :100].mean() > 0.95 and r.mask[:, 160:].mean() < 0.05


def test_canopy_cover_accepts_path(tmp_path, green_on_soil):
    from PIL import Image

    p = tmp_path / "plot.png"
    Image.fromarray(green_on_soil).save(p)
    assert canopy_cover(p).cover == pytest.approx(0.5, abs=0.03)


def test_otsu_fails_on_full_canopy_but_fixed_threshold_does_not():
    rng = np.random.default_rng(3)
    full = np.clip(np.full((128, 128, 3), (60, 140, 50)) + rng.integers(-15, 16, (128, 128, 3)), 0, 255).astype(
        np.uint8
    )
    assert canopy_cover(full).cover > 0.95
    assert canopy_cover(full, method="exg_otsu").cover < 0.8


def test_bad_method(green_on_soil):
    with pytest.raises(ValueError):
        canopy_cover(green_on_soil, method="magic")


def test_tile_windows_cover_image():
    h, w = 2500, 3000
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in tile_windows(h, w, tile=1024, overlap=0.2):
        assert x1 - x0 <= 1024 and y1 - y0 <= 1024
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]))
    assert list(keep) == [0, 2]


def _bright_box_detector(rgb):
    from scipy import ndimage

    labels, n = ndimage.label(rgb.mean(axis=2) > 150)
    boxes = [[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in ndimage.find_objects(labels)]
    return np.asarray(boxes, float).reshape(-1, 4), np.ones(n)


def test_count_heads_with_tiling_merges_split_objects(heads_image):
    img, gt = heads_image
    boxes, _ = detect_tiled(img, _bright_box_detector, tile=128, overlap=0.5)
    r = count_heads(img, _bright_box_detector, tile=1024, ground_area_m2=0.5)
    assert r.count == 3 and r.density_per_m2 == pytest.approx(6.0)
    assert len(boxes) >= 3


def test_unreleased_model_message():
    with pytest.raises(ModelNotReleasedError):
        load_detector("wheatkit/fasterrcnn-gwhd")
