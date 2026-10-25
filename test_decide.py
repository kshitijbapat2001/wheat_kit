import pandas as pd
import pytest

from wheatkit.decide import Observation, advise, load_rules, option_outcome_range, rank_lines
from wheatkit.decide.advisory import RATE_PATTERN


def test_rules_are_cited_and_rate_free():
    for r in load_rules():
        assert r["citations"] and all(c["url"].startswith("https://") for c in r["citations"])
        assert not RATE_PATTERN.search(r["message"])


def test_rate_pattern_catches_rates():
    assert RATE_PATTERN.search("apply 500 mL/ha")
    assert RATE_PATTERN.search("use 1.5 L/ha")
    assert not RATE_PATTERN.search("flag-leaf window Z39")


def test_stripe_rust_window_fires_for_susceptible_in_au():
    obs = Observation(region="AU", zadoks=(36, 39), diseases={"stripe_rust": 3.0}, variety_rating="S")
    ids = [a.rule_id for a in advise(obs)]
    assert "au_stripe_rust_susceptible_z31_z39" in ids
    assert ids[0] == "describe"


def test_resistant_variety_gets_no_window_rule():
    obs = Observation(region="AU", zadoks=(36, 39), diseases={"stripe_rust": 3.0}, variety_rating="R")
    ids = [a.rule_id for a in advise(obs)]
    assert "au_stripe_rust_susceptible_z31_z39" not in ids
    assert "au_stripe_rust_economic_tool" in ids


def test_region_gating():
    obs = Observation(region="UK", zadoks=(36, 39), diseases={"stripe_rust": 3.0}, variety_rating="S")
    ids = [a.rule_id for a in advise(obs)]
    assert not any(i.startswith("au_") for i in ids)
    assert "uk_septoria_t2_window" in ids


def test_low_confidence_suppresses_rules():
    obs = Observation(
        region="AU", zadoks=(30, 45), diseases={"stripe_rust": 3.0}, variety_rating="S", stage_confidence=0.3
    )
    advice = advise(obs)
    assert [a.rule_id for a in advice] == ["describe", "inspect"]
    assert all(a.tier == 1 for a in advice)


def test_out_of_distribution_suppresses_rules():
    obs = Observation(
        region="AU", zadoks=(36, 39), diseases={"stripe_rust": 3.0}, variety_rating="S", out_of_distribution=True
    )
    assert advise(obs)[-1].rule_id == "inspect"


def test_weather_rule_needs_weather():
    base = dict(region="US", zadoks=(60, 62))
    assert "fhb_weather_around_flowering" not in [a.rule_id for a in advise(Observation(**base))]
    obs = Observation(**base, weather={"hours_rh90_15_30c_7d": 41})
    msg = next(a.message for a in advise(obs) if a.rule_id == "fhb_weather_around_flowering")
    assert "41 hours" in msg


def test_outcome_range():
    r = option_outcome_range(4.0, 350, 30, loss_without=(0.0, 0.1, 0.3), loss_with=(0.0, 0.02, 0.05))
    assert r.worst < r.most_likely < r.best
    assert 0 <= r.p_negative <= 1
    with pytest.raises(ValueError):
        option_outcome_range(4.0, 350, 30, loss_without=(0.3, 0.1, 0.0), loss_with=(0, 0, 0))


def test_rank_lines_heritability_filter_and_uncertainty():
    df = pd.DataFrame(
        {
            "line": ["A", "B", "C", "D"],
            "heads_m2": [520, 480, 400, 450],
            "senescence_gdd": [1800, 1750, 1650, 1700],
            "noisy_trait": [1, 5, 2, 9],
            "heads_m2_se": [10, 10, 10, 10],
            "heading_date": [120, 121, 119, 122],
        }
    )
    ranking, dropped = rank_lines(
        df,
        {"heads_m2": 1.0, "senescence_gdd": 0.5, "noisy_trait": 1.0},
        se={"heads_m2": "heads_m2_se"},
        heritability={"noisy_trait": 0.1},
        top_k=2,
    )
    assert dropped == ["noisy_trait"]
    assert ranking.iloc[0]["line"] == "A"
    assert ranking["p_top_2"].sum() == pytest.approx(2.0)
