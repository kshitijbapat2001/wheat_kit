import numpy as np
import pandas as pd
import pytest

from wheatkit.decide import Observation, advise
from wheatkit.env import humid_hours
from wheatkit.env.power import parse_power_hourly_json, power_hourly_url
from wheatkit.ood import FEATURE_NAMES, OODDetector, image_features
from wheatkit.phenology import fuse_stage, zadoks_from_thermal_time


def _canopy(rng, green=(60, 140, 50), soil=(120, 95, 70)):
    img = np.empty((64, 64, 3), np.uint8)
    split = rng.integers(20, 44)
    img[:, :split], img[:, split:] = green, soil
    return np.clip(img + rng.integers(-10, 11, img.shape), 0, 255).astype(np.uint8)


def test_image_features_shape(green_on_soil):
    f = image_features(green_on_soil)
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


def test_ood_flags_unlike_images_but_not_similar_ones():
    rng = np.random.default_rng(0)
    ref = np.stack([image_features(_canopy(rng)) for _ in range(120)])
    det = OODDetector(alpha=0.05).fit(ref)
    new_in = np.stack([image_features(_canopy(rng)) for _ in range(60)])
    assert det.is_ood(new_in).mean() <= 0.15
    night = np.stack([image_features(_canopy(rng, green=(10, 12, 30), soil=(5, 5, 15))) for _ in range(10)])
    snow = np.stack([image_features(_canopy(rng, green=(240, 240, 245), soil=(250, 250, 250))) for _ in range(10)])
    assert det.is_ood(night).all() and det.is_ood(snow).all()


def test_ood_needs_enough_reference():
    with pytest.raises(ValueError):
        OODDetector().fit(np.zeros((5, 3)))


def test_zadoks_interpolation_needs_anchors():
    anchors = [(0, 0), (500, 30), (900, 39), (1200, 65)]
    assert zadoks_from_thermal_time(700, anchors) == pytest.approx(34.5)
    with pytest.raises(ValueError):
        zadoks_from_thermal_time(700, [(0, 0)])


def test_fusion_agreeing_evidence_is_confident():
    prior_only = fuse_stage(35, 5)
    agree = fuse_stage(35, 5, {(31, 39): 0.9, (40, 49): 0.1})
    assert agree.confidence > prior_only.confidence
    assert agree.interval[1] - agree.interval[0] <= prior_only.interval[1] - prior_only.interval[0]


def test_fusion_conflict_lowers_confidence_and_feeds_advice():
    conflict = fuse_stage(35, 4, {(31, 39): 0.5, (50, 59): 0.5})
    assert conflict.confidence < fuse_stage(35, 4, {(31, 39): 1.0}).confidence
    obs = Observation.from_stage_estimate(
        fuse_stage(36, 2, {(31, 39): 0.95}), region="AU", diseases={"stripe_rust": 1.0}, variety_rating="S"
    )
    assert "au_stripe_rust_susceptible_z31_z39" in [a.rule_id for a in advise(obs)]


def test_humid_hours_window():
    idx = pd.date_range("2026-10-01", periods=10 * 24, freq="h")
    df = pd.DataFrame({"T2M": 20.0, "RH2M": 80.0}, index=idx)
    df.loc["2026-10-08 00:00":"2026-10-08 05:00", "RH2M"] = 95.0
    df.loc["2026-10-02 00:00":"2026-10-02 23:00", "RH2M"] = 95.0
    df.loc["2026-10-08 03:00", "T2M"] = 35.0
    assert humid_hours(df, end="2026-10-10 00:00") == 5


def test_power_hourly_parsing():
    assert "temporal/hourly" in power_hourly_url(-35, 148, "2026-10-01", "2026-10-02")
    df = parse_power_hourly_json(
        {
            "properties": {
                "parameter": {
                    "T2M": {"2026100100": 12.0, "2026100101": 11.5},
                    "RH2M": {"2026100100": 91.0, "2026100101": -999.0},
                }
            }
        }
    )
    assert df.index[1] == pd.Timestamp("2026-10-01 01:00") and df["RH2M"].isna().sum() == 1
