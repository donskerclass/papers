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

## Daily email

After each nightly run the workflow emails the day's page: the four highlight
sections with abstracts and the top of the latest listing's ranked list in the body, and the
full page attached as `papers-YYYY-MM-DD.html` (dated by arXiv listing) for the
record. It is sent through Resend or Gmail (see below); pushes that only
edit the profile don't send mail. A manual run sends mail only if "Also send the
email" is ticked.

One-time setup, either option (the workflow uses Resend if its key is set,
otherwise Gmail; with neither, the email job skips itself). Each `gh secret set`
prompts for the value, so nothing lands in shell history or in the repo.

**Option A, Resend** (free tier, 3,000 emails/month; works for any account):

1. Sign up at <https://resend.com> with the address the email should go to, and
   create an API key (permission "Sending access" is enough).
2. `gh secret set RESEND_API_KEY --repo donskerclass/papers` and
   `gh secret set MAIL_TO --repo donskerclass/papers` (the sign-up address).

**Option B, Gmail SMTP** (needs 2-Step Verification, which is what makes the
app-passwords page available):

1. Create an app password at <https://myaccount.google.com/apppasswords>.
2. `gh secret set MAIL_USERNAME`, `MAIL_TO` and `MAIL_PASSWORD` (each with
   `--repo donskerclass/papers`): the Gmail address, the destination, and the
   16-character app password.

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
