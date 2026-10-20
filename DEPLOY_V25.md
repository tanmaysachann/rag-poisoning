# V25 hosted scanner deployment

The public research console at `https://rag-poisoning.vercel.app` proxies
`/api/v25/*` to the separate container project
`sentinel-rag-v25`. Its production hostname is
`https://sentinel-rag-v25-tanmaysachan0005-gmailcoms-projects.vercel.app`.
The shorter `https://sentinel-rag-v25.vercel.app` alias also points to it.
The same container also serves the frozen V24 scanner for paired comparisons.
The existing V24 admitted-source RAG answer path remains on the main site.

## Source and evidence bindings

`deploy/build_v24_bundle.py` stages the V25 scanner, V24 baseline, pinned
model artifacts, frozen V25 rule, and first NFCorpus holdout report. At image
build, `Dockerfile.vercel` downloads the official BEIR NQ archive, checks its
SHA-256, and constructs the 2,681,468-passage FTS5 lookup index. The scanner
refuses to report itself available if model, calibration, rule, or holdout
bindings have changed. The site bundle is staged separately by
`deploy/build_site_bundle.py`; its `V25_REMOTE_URL` environment variable
points to the scanner project's public HTTPS origin.

## Rebuild and verify

Build bundles into new directories from the repository root:

```powershell
python -m deploy.build_v24_bundle --output tmp/v25_service_release
python -m deploy.build_site_bundle --output tmp/site_v25_release
```

Run `vercel deploy` **from inside each staged bundle** so the CLI reads that
bundle's `vercel.json`. The scanner bundle uses a container, and the site
bundle uses the FastAPI function in its own configuration. Link the scanner
bundle to `sentinel-rag-v25` and the site bundle to `rag-poisoning` before
deploying. Set `V25_REMOTE_URL` for both preview and production on the site
project to the scanner hostname above. Confirm that SSO deployment protection
is disabled for `sentinel-rag-v25` or the site's server-side proxy will be
unable to call it.

Verify these endpoints after each deployment:

```text
GET  /api/v25/status     available: true
GET  /api/v25/evidence   493/600 V25 attacks; 14/240 clean flags
POST /api/v25/scan       query, reference, candidate, optional expected_sha256
GET  /api/v24/status     baseline available
POST /api/v24/scan       paired baseline response
```

`python -m evaluation.prepare_v25_smoke` writes a saved attack and clean
request to `tmp/v25_smoke_{attack,clean}.json`. The chosen attack is a pinned
NFCorpus case that V24 passes and V25 reviews; both scanners pass the clean
case. The saved report and per-case trace are in
`results/v25_nfcorpus_holdout.json` and `.cases.jsonl`. Test the public site
API as well as the scanner service, then inspect the scanner selector in the
site UI. The NFCorpus report measures passage screening on deterministic
synthetic insertions; it is not a live answer-quality result.

## Verified production release (20 October 2026)

The V25 scanner deployment `dpl_2XwAcgX5mQFkBtoixShq8PvZwbGr` and the
main site deployment `dpl_GgshumoXweeCurCMCadmNrGiQrc2` reached `Ready`.
Direct requests to `https://rag-poisoning.vercel.app` returned V25
`available: true`, the frozen **493/600** attack and **14/240** clean report,
and V24 `available: true`. The saved NFCorpus attack returned V25 `review`
while its paired V24 baseline returned `pass`; the saved clean passage
returned V25 `pass`. The HTML contained the V25 holdout section, and
`frontend/v24.js` passed Node's syntax check. A rendered browser screenshot
was not available in this session.
