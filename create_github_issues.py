from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import yaml

API = "https://api.github.com/repos/{repo}"
BACKLOG = Path(__file__).resolve().parents[1] / ".github" / "backlog.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="OWNER/NAME")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    plan = yaml.safe_load(BACKLOG.read_text(encoding="utf-8"))

    if not args.apply:
        print(f"DRY RUN for {args.repo}")
        print(f"labels: {[label['name'] for label in plan['labels']]}")
        print(f"milestones: {[m['title'] for m in plan['milestones']]}")
        for i in plan["issues"]:
            print(f"- [{', '.join(i['labels'])}] {i['title']}  ({i['milestone']})")
        print("\nRe-run with --apply and GITHUB_TOKEN set to create them.")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set", file=sys.stderr)
        return 1
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    base = API.format(repo=args.repo)

    for label in plan["labels"]:
        r = s.post(f"{base}/labels", json=label, timeout=30)
        print("label", label["name"], r.status_code)
    numbers = {}
    for m in plan["milestones"]:
        r = s.post(f"{base}/milestones", json=m, timeout=30)
        if r.status_code == 201:
            numbers[m["title"]] = r.json()["number"]
        print("milestone", m["title"], r.status_code)
    if len(numbers) < len(plan["milestones"]):
        for m in s.get(f"{base}/milestones?state=all", timeout=30).json():
            numbers.setdefault(m["title"], m["number"])
    for i in plan["issues"]:
        body = {"title": i["title"], "body": i["body"], "labels": i["labels"], "milestone": numbers.get(i["milestone"])}
        r = s.post(f"{base}/issues", json=body, timeout=30)
        print("issue", i["title"], r.status_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
