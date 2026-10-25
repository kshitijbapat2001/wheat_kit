import numpy as np
import pandas as pd
import pytest

from wheatkit.phenology import cumulative_thermal_time, daily_thermal_time, event_date, fit_logistic, logistic
from wheatkit.traits import Observation, aggregate_images, footprint_m2, heads_per_m2, trait_table


def test_thermal_time():
    np.testing.assert_allclose(daily_thermal_time([5, -5, 20], [15, 3, 40]), [10, 0, 23])
    assert cumulative_thermal_time([5, 5], [15, 15])[-1] == 20


def test_logistic_fit_recovers_heading():
    t = np.linspace(0, 1500, 30)
    y = logistic(t, 0.0, 1.0, 900.0, 0.02) + np.random.default_rng(0).normal(0, 0.02, 30)
    fit = fit_logistic(t, y)
    assert fit.t50 == pytest.approx(900, abs=25)
    assert event_date(t, y, 0.5) == pytest.approx(900, abs=25)


def test_decreasing_curve_senescence():
    t = np.linspace(1000, 2200, 25)
    y = logistic(t, 0.05, 0.9, 1700.0, -0.015)
    fit = fit_logistic(t, y)
    assert fit.rate < 0
    onset = fit.time_at(0.9)
    assert onset < 1700


def test_trait_table_and_aggregation():
    per_image = pd.DataFrame(
        {"observation_unit_id": ["p1", "p1", "p1", "p2"], "date": ["2026-10-01"] * 4, "cover": [0.40, 0.42, 0.90, 0.10]}
    )
    agg = aggregate_images(per_image, "cover", trait="canopy_cover", unit="fraction")
    p1 = agg[agg.observation_unit_id == "p1"].iloc[0]
    assert p1.value == pytest.approx(0.42) and p1.n_images == 3
    tt = trait_table([Observation("p1", "2026-10-01", "head_density", 420.0, "heads/m2", 380, 460)])
    assert list(tt.columns)[:4] == ["observation_unit_id", "date", "trait", "value"]


def test_density_helpers():
    area = footprint_m2(1024, 1024, gsd_mm=0.4)
    assert area == pytest.approx(0.1678, abs=1e-3)
    assert heads_per_m2(70, area) == pytest.approx(417, abs=2)
