import numpy as np
import pandas as pd
import pytest
from PIL import Image

from wheatkit.data import (
    assert_no_leakage,
    dhash,
    duplicate_groups,
    find_near_duplicates,
    format_boxes_string,
    group_split,
    leave_one_group_out,
    parse_boxes_string,
    read_gwhd_csv,
)
from wheatkit.data.gwhd import xywh_to_xyxy


def test_parse_boxes_string_roundtrip():
    b = parse_boxes_string("10 20 30 40;50 60 70 90")
    assert b.shape == (2, 4)
    assert format_boxes_string(b) == "10 20 30 40;50 60 70 90"
    assert parse_boxes_string("no_box").shape == (0, 4)
    assert parse_boxes_string(float("nan")).shape == (0, 4)


def test_degenerate_boxes_dropped():
    assert parse_boxes_string("10 10 10 20").shape == (0, 4)


def test_malformed_box_raises():
    with pytest.raises(ValueError):
        parse_boxes_string("1 2 3")


def test_read_gwhd_csv(tmp_path):
    p = tmp_path / "competition_train.csv"
    pd.DataFrame(
        {"image_name": ["a.png", "b.png"], "BoxesString": ["0 0 5 5", "no_box"], "domain": ["Arvalis_1", "Ukyoto_1"]}
    ).to_csv(p, index=False)
    samples = read_gwhd_csv(p)
    assert [s.count for s in samples] == [1, 0]
    assert samples[1].domain == "Ukyoto_1"


def test_dhash_finds_resized_duplicate():
    rng = np.random.default_rng(1)
    base = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    other = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    resized = base.resize((96, 96))
    assert dhash(base) != dhash(other)
    pairs = find_near_duplicates([("base", base), ("resized", resized), ("other", other)], max_distance=6)
    ids = {(a, b) for a, b, _ in pairs}
    assert ("base", "resized") in ids
    assert all("other" not in pair for pair in ids)
    groups = duplicate_groups(["base", "resized", "other"], pairs)
    assert groups["base"] == groups["resized"] != groups["other"]


def test_group_split_keeps_groups_whole():
    groups = [f"plot{i // 5}" for i in range(100)]
    tr, te = group_split(groups, test_fraction=0.3, seed=2)
    g = np.asarray(groups)
    assert_no_leakage(g[tr], g[te])
    assert 0.2 <= len(te) / 100 <= 0.4


def test_leave_one_group_out():
    folds = list(leave_one_group_out(["a", "a", "b", "c"]))
    assert [f[0] for f in folds] == ["a", "b", "c"]
    assert list(folds[0][2]) == [0, 1]


def test_leakage_detected():
    with pytest.raises(AssertionError):
        assert_no_leakage(["a", "b"], ["b", "c"])


def test_hf_mirror_boxes_are_converted_from_xywh():
    np.testing.assert_array_equal(xywh_to_xyxy([[227, 3, 58, 47]]), [[227, 3, 285, 50]])
    assert xywh_to_xyxy([]).shape == (0, 4)
