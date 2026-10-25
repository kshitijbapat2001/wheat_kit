# Governance

## Maintainers

| Name | Affiliation | Areas |
|---|---|---|
| _TBD (founder)_ | _TBD_ | project lead, data registry |
| _TBD (2nd maintainer, required before v0.1)_ | _different institution_ | models and evaluation |

Long-lived open-source science tools have an institutional home and more than one
maintainer. Recruiting the second maintainer is a release blocker for v0.1.

## Decisions

- **Day-to-day:** lazy consensus on PRs. One maintainer approval merges a PR; changes to
  `registry.yaml` licences or `rules.yaml` need **two** approvals.
- **Larger changes** (new task family, a new tier-3 or tier-4 capability, changes to
  the API before 1.0) go in an RFC issue, which stays open for at least 7 days.
- **Decision-support rules** additionally need a reviewer with agronomy expertise, who
  is named in the PR.

## Becoming a maintainer

Make sustained, high-quality contributions (about 5 merged PRs or equivalent review
work). Any maintainer can nominate you, and the others confirm by consensus.

## Releases

Releases follow semantic versioning, and a release goes out at least monthly while the
project is in 0.x. Each release is tagged on GitHub, which triggers PyPI (trusted
publishing) and Zenodo (DOI).
