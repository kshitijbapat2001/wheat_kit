import pytest

from wheatkit.data import get_dataset, list_datasets, load_registry
from wheatkit.data.registry import LICENCE_GROUPS

PERMISSIVE = {"CC-BY-4.0", "CC0-1.0", "MIT"}


def test_registry_loads_and_ids_unique():
    reg = load_registry()
    assert len(reg) >= 30
    assert "gwhd_2021" in reg and "fip_1" in reg and "gwfss_v1" in reg


def test_every_entry_has_source_and_valid_group():
    for d in load_registry().values():
        assert d.licence_group in LICENCE_GROUPS
        assert d.url, f"{d.id} has no resolvable URL"
        assert d.tasks, f"{d.id} has no tasks"


def test_permissive_group_matches_licence():
    for d in load_registry().values():
        if d.licence_group == "permissive":
            assert d.licence in PERMISSIVE, f"{d.id}: {d.licence} is not a permissive licence"


def test_tier3_and_restricted_need_reasons():
    for d in load_registry().values():
        if d.tier == 3 or d.licence_group in ("unknown",):
            assert d.tier_reason, f"{d.id} must explain its tier"


def test_zenodo_entries_pin_checksums_when_files_listed():
    for d in load_registry().values():
        for f in d.source.get("files", []):
            assert len(f["md5"]) == 32


def test_filters():
    heads = list_datasets(task="head_counting", max_tier=1)
    assert heads and all(d.tier == 1 and "head_counting" in d.tasks for d in heads)
    nc = list_datasets(licence_group="non-commercial")
    assert all(d.licence_group == "non-commercial" for d in nc)


def test_summary_mentions_conflicts():
    s = get_dataset("gwhd_2021").summary()
    assert "MIT" in s and "CC-BY-4.0" in s


def test_unknown_id():
    with pytest.raises(KeyError):
        get_dataset("not_a_dataset")
