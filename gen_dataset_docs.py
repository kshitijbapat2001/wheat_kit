from __future__ import annotations

from pathlib import Path

from wheatkit.data import list_datasets
from wheatkit.data.registry import LICENCE_NOTES

OUT = Path(__file__).resolve().parents[1] / "docs" / "datasets.md"


def main() -> None:
    lines = [
        "# Datasets",
        "",
        "Generated from `src/wheatkit/data/registry.yaml` by `scripts/gen_dataset_docs.py`. "
        "Licences are those on the **original** record; mirror conflicts are listed. "
        "Corrections are welcome as PRs.",
        "",
        "## Licence groups",
        "",
        *[f"- **{g}**: {note}" for g, note in LICENCE_NOTES.items()],
        "",
    ]
    for tier, title in ((1, "Tier 1: loaded by default"), (2, "Tier 2: opt-in"), (3, "Tier 3: not for training")):
        lines += [f"## {title}", "", "| id | dataset | licence | images | tasks | notes |", "|---|---|---|---|---|---|"]
        for d in list_datasets(tier=tier):
            notes = "; ".join([*d.licence_conflicts, *([d.tier_reason] if d.tier_reason else []), *d.known_issues])
            n = f"{d.n_images:,}" if d.n_images else "-"
            name = f"[{d.name}]({d.url})" if d.url else d.name
            lines.append(
                f"| `{d.id}` | {name} | {d.licence or 'not stated'} | {n} | {', '.join(d.tasks)} | "
                f"{notes.replace('|', '/')} |"
            )
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
