import numpy as np
import pytest

from wheatkit.uncertainty import ConformalRegressor, conformal_quantile, coverage


def test_quantile_too_few_points_is_infinite():
    assert conformal_quantile(np.array([1.0, 2.0]), alpha=0.1) == float("inf")


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_marginal_coverage(alpha):
    rng = np.random.default_rng(0)
    y = rng.normal(100, 20, 4000)
    yhat = y + rng.normal(0, 10, 4000)
    cr = ConformalRegressor(alpha=alpha).fit(y[:2000], yhat[:2000])
    lo, hi = cr.predict(yhat[2000:])
    assert coverage(y[2000:], lo, hi) >= 1 - alpha - 0.02


def test_group_conditional_widths_differ_and_cover():
    rng = np.random.default_rng(1)
    n = 3000
    g = np.where(rng.random(n) < 0.5, "easy_site", "hard_site")
    noise = np.where(g == "easy_site", 2.0, 15.0)
    y = rng.normal(50, 10, n)
    yhat = y + rng.normal(0, 1, n) * noise
    cal, test = slice(0, 1500), slice(1500, None)
    cr = ConformalRegressor(alpha=0.1).fit(y[cal], yhat[cal], groups=g[cal])
    assert cr.q_group_["hard_site"] > 3 * cr.q_group_["easy_site"]
    lo, hi = cr.predict(yhat[test], groups=g[test])
    for site in ("easy_site", "hard_site"):
        m = g[test] == site
        assert coverage(y[test][m], lo[m], hi[m]) >= 0.87


def test_unseen_group_falls_back_to_global():
    cr = ConformalRegressor(alpha=0.1, min_group_size=5).fit(np.arange(50.0), np.arange(50.0) + 1, groups=["a"] * 50)
    lo, hi = cr.predict([10.0], groups=["new_site"])
    assert hi[0] - lo[0] == pytest.approx(2 * cr.q_global_)
