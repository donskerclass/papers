# Papers

A personal daily ranking of new arXiv preprints, published at
<https://donskerclass.github.io/papers/>.

Every weekday night a GitHub Actions job fetches the last week of new submissions
in the watched arXiv categories and scores each paper on four criteria:

| criterion | compared against | profile file |
|---|---|---|
| your work | my own papers | `profile/own_papers.yaml` |
| favorites | papers from my year-end "papers I liked" posts and Bluesky | `profile/favorites.yaml` |
| interests | short written descriptions of what I want to see (and what to push down) | `profile/interests.yaml` |
| reading | my whole Zotero library: a tf-idf classifier plus nearest neighbours | `data/library/` (built locally) |

The page shows a few papers from the latest listing under each criterion, then
everything from the week, ranked. `profile/config.yaml` sets the categories,
weights and the embedding model. The site's "How it works" page describes the
method in full.

## Editing the profile

Edit the YAML files in `profile/` (on GitHub or locally) and push. The workflow
reruns on every push to `profile/`, looks up abstracts for any new arXiv ids,
DOIs or titles, and commits the lookups to `data/profile_cache.json`.

## Refreshing the reading library (laptop only)

The library model is fitted on the laptop, because it reads the local Zotero
database, and only derived data is committed. Refresh it now and then:

```sh
uv run paperboard build-library          # re-export Zotero, fetch background sample, refit
git add data/library && git commit -m "Refresh library model" && git push
```

`--skip-background` reuses the cached background sample under `local/`.
Nothing under `local/` (the Zotero export, with titles and abstracts) is ever
committed.

## Running locally

```sh
uv sync
uv run paperboard run      # writes site/index.html
open site/index.html
```

## Optional API keys

Everything works without keys. Abstract lookups use arXiv, Crossref and
Semantic Scholar's anonymous tiers, which are slow but sufficient. If they get
too slow, free keys for [OpenAlex](https://openalex.org) and
[Semantic Scholar](https://www.semanticscholar.org/product/api) can be set as
environment variables `OPENALEX_API_KEY` / `S2_API_KEY` (and as repository
secrets for the workflow).

## Planned sources

EconPapers/RePEc (computational economics, econometrics) and Semantic Scholar
recommendations.
