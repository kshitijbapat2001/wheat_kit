from __future__ import annotations

import sys

import requests

from wheatkit.data import load_registry

ZENODO_LICENCE = {"cc-by-4.0": "CC-BY-4.0", "mit-license": "MIT", "cc-zero": "CC0-1.0", "cc0-1.0": "CC0-1.0"}


def check(ds, s: requests.Session) -> tuple[str, str]:
    src = ds.source
    try:
        if src["type"] == "zenodo":
            r = s.get(f"https://zenodo.org/api/records/{src['record']}", timeout=60)
            if r.status_code != 200:
                return "❌", f"HTTP {r.status_code}"
            meta = r.json()
            lic = ZENODO_LICENCE.get((meta.get("metadata", {}).get("license") or {}).get("id", ""), "?")
            files = {f["key"]: f.get("checksum", "").removeprefix("md5:") for f in meta.get("files", [])}
            notes = []
            if ds.licence and lic != "?" and lic != ds.licence:
                notes.append(f"licence now {lic}")
            for f in src.get("files", []):
                if f["key"] not in files:
                    notes.append(f"missing {f['key']}")
                elif files[f["key"]] != f["md5"]:
                    notes.append(f"md5 changed for {f['key']}")
            if not files:
                notes.append("no files listed")
            return ("⚠️", "; ".join(notes)) if notes else ("✅", f"{len(files)} files")
        if src["type"] == "huggingface":
            r = s.get(f"https://huggingface.co/api/datasets/{src['repo']}", timeout=60)
            return ("✅", "repo resolves") if r.status_code == 200 else ("❌", f"HTTP {r.status_code}")
        if ds.doi:
            r = s.get(f"https://api.datacite.org/dois/{ds.doi}", timeout=60)
            if r.status_code != 200:
                return "❌", f"DataCite HTTP {r.status_code}"
            rights = [x.get("rightsIdentifier") for x in r.json()["data"]["attributes"].get("rightsList", [])]
            rights = [x.upper() for x in rights if x]
            if ds.licence and rights and ds.licence.upper() not in rights:
                return "⚠️", f"DataCite licence {rights} differs from registry {ds.licence}"
            return "✅", f"DOI resolves; licence {rights or 'not in DataCite'}"
        url = ds.url
        r = s.head(url, timeout=60, allow_redirects=True)
        if r.status_code >= 400:
            r = s.get(url, timeout=60, stream=True)
        return ("✅", f"HTTP {r.status_code}") if r.status_code < 400 else ("⚠️", f"HTTP {r.status_code}")
    except requests.RequestException as e:
        return "❌", type(e).__name__


def main() -> int:
    s = requests.Session()
    s.headers["User-Agent"] = "wheatkit-registry-check"
    print("| dataset | tier | status | detail |\n|---|---|---|---|")
    bad = 0
    for ds in load_registry().values():
        status, detail = check(ds, s)
        bad += status == "❌"
        print(f"| {ds.id} | {ds.tier} | {status} | {detail} |")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
