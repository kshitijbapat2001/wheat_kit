import json

import pytest

from wheatkit.cli import main
from wheatkit.env.power import parse_power_json, power_url
from wheatkit.env.silo import parse_silo_csv, silo_url
from wheatkit.io import to_brapi_observations, to_miappe_wide, write_brapi_json
from wheatkit.traits import Observation, trait_table


@pytest.fixture
def table():
    return trait_table(
        [
            Observation("plot_1", "2026-10-01", "canopy_cover", 0.42, "fraction", 0.38, 0.46, "exg_otsu", "0.1.0"),
            Observation("plot_1", "2026-10-01", "head_density", 410.0, "heads/m2"),
        ]
    )


def test_brapi_export(table, tmp_path):
    obs = to_brapi_observations(table, "study_7", variable_ids={"canopy_cover": "CO_321:0000001"})
    assert obs[0]["observationUnitDbId"] == "plot_1"
    assert obs[0]["observationVariableDbId"] == "CO_321:0000001"
    assert obs[0]["additionalInfo"]["interval_lower"] == 0.38
    assert "observationVariableDbId" not in obs[1]
    p = write_brapi_json(table, tmp_path / "obs.json", "study_7")
    assert len(json.loads(p.read_text())["observations"]) == 2


def test_miappe_wide(table):
    wide = to_miappe_wide(table)
    assert set(wide.columns) == {"observation_unit_id", "date", "canopy_cover", "head_density"}


def test_power_parsing():
    url = power_url(-34.97, 138.63, "2026-06-01", "2026-06-03")
    assert "community=AG" in url and "start=20260601" in url
    payload = {
        "header": {"fill_value": -999.0},
        "properties": {
            "parameter": {
                "T2M_MAX": {"20260601": 17.1, "20260602": -999.0},
                "T2M_MIN": {"20260601": 6.2, "20260602": 5.0},
            }
        },
    }
    df = parse_power_json(payload)
    assert df.loc["2026-06-01", "T2M_MAX"] == 17.1
    assert df["T2M_MAX"].isna().sum() == 1


def test_silo_url_and_parsing():
    url = silo_url("me@example.org", "2026-01-01", "2026-01-31", lat=-27.5, lon=151.0)
    assert "DataDrillDataset.php" in url and "password=apirequest" in url
    with pytest.raises(ValueError):
        silo_url("me@example.org", "2026-01-01", "2026-01-31")
    df = parse_silo_csv("latitude,longitude,YYYY-MM-DD,daily_rain,max_temp\n-27.5,151,2026-01-01,0.0,31.2\n")
    assert df.loc["2026-01-01", "max_temp"] == 31.2


def test_cli_datasets_list(capsys):
    assert main(["datasets", "list", "--task", "head_counting", "--max-tier", "1"]) == 0
    out = capsys.readouterr().out
    assert "gwhd_2021" in out and "CC-BY-4.0" in out


def test_cli_info(capsys):
    main(["datasets", "info", "fip_1"])
    assert "CC0-1.0" in capsys.readouterr().out


def test_cli_advise(capsys):
    main(["advise", "--region", "AU", "--stage", "36-39", "--disease", "stripe_rust=2", "--rating", "S"])
    out = capsys.readouterr().out
    assert "Z31" in out and "Decision support only" in out
