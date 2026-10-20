# V24 hosted research scanner

Public site: https://rag-poisoning.vercel.app (open **V24 scanner**).
The separate scanner service is at https://sentinel-rag-v24.vercel.app.
The service project's Vercel Authentication is disabled so the main site's
server can call it. Preview deployments of the main site remain protected;
the site's canonical production domain is public.

The main Vercel site exposes V24 in its own tab and at `POST /api/v24/scan`.
That route proxies to a separate Vercel container carrying the trained model,
calibration, and full BEIR Natural Questions source index. The main site's
existing V3 workbench remains available. The V24 tab now has an explicitly
selected research answer replay with source admission and pinned clean
recovery; Secure Query retains its own policy. PPO ran during offline adversarial training and evaluation; the
hosted scanner performs inference only.

## Hosted API

The service exposes `GET /api/v24/status`, `GET /api/v24/evidence`,
`GET /api/v24/sample`, and `POST /api/v24/scan`. The POST JSON body has required
`query`, `reference`, and `candidate` strings and an optional `expected_sha256`.
An expected digest must come from an independently pinned document ID/hash
record. A digest typed by the same untrusted party that supplied a document
does not authenticate it. The response returns `pass` or `review`, scores, and
reason codes. `review` calls for human investigation; `pass` is not a proof of
truth or safety.

The sample is a pinned published PoisonArena development example, not an
unseen test. Evidence figures are from local evaluation on previously inspected
questions. A hosted scan of a few examples checks that packaging, model
loading, snapshot search, and API wiring work; it does not reproduce all 1,066
evaluation cases in the deployed Linux image. The Linux image rebuilds the
SQLite full-text index from the exact BEIR archive pinned by SHA-256 and
verifies the source manifest, schema, and passage count. Its database bytes may
differ from the Windows-built evaluation snapshot.

On the public production site, `/api/v24/status` reported all assets present
and 2,681,468 snapshot passages. Through the site's proxy, a published attack
returned `review`, an unchanged pinned passage returned `pass`, and a changed
passage with the original expected digest returned `review`. The original
`/api/health` also remained operational. These are serving checks, not a
hosted benchmark run.

## Rebuild and deploy

From a checkout with the V24 artifacts and reports present:

```powershell
python -m deploy.build_v24_bundle --output tmp/v24_release
vercel.cmd link --cwd tmp/v24_release --project sentinel-rag-v24 --yes
vercel.cmd deploy --cwd tmp/v24_release --yes --logs
```

The V24 bundle contains only serving assets and evaluation summaries. Its
`Dockerfile.vercel` fetches the official BEIR NQ archive during the build,
checks its SHA-256, and builds all 2,681,468 passages into the SQLite FTS index.
The container image uses CPU PyTorch and local MiniLM weights. The first cold
build can take several minutes. Verify `/api/v24/status` and a sample scan on
the preview before promotion.

The main site is staged from `git archive HEAD` plus an explicit V24 serving
overlay, so unrelated local experiments are excluded:

```powershell
python -m deploy.build_site_bundle --output tmp/site_v24_release
vercel.cmd link --cwd tmp/site_v24_release --project rag-poisoning --yes
vercel.cmd env add V24_REMOTE_URL preview
vercel.cmd env add V24_REMOTE_URL production
vercel.cmd deploy --cwd tmp/site_v24_release --yes --logs
```

Set `V24_REMOTE_URL` to the deployed scanner's HTTPS origin. Verify that the
site's `/api/v24/status` reports `available: true` and that its scan route
returns the same decision as the direct scanner preview before promoting the
site. A redeployment is needed after changing a Vercel environment variable.
Deployment credentials and project links are local to the operator and are
not included in the source bundle.

## Current public answer replay

The 2026-10-20 production release `dpl_7vmwJFzw5CpoCXB3dg1y3F3Xn5XP`
was built from `tmp/site_v24_admitted_release2` and aliased to
https://rag-poisoning.vercel.app. The site's `/api/v24/research-evidence`
serves the frozen SciFact transfer, initial and revised answer replays, and
three-seed PPO report. `POST /api/lab/run` with `v24_gate: true` runs the
admitted-source answer path on a selected validation case. Public production
checks confirmed health, available scanner assets, the evidence reports, and
a stealth substitution replay where V3 returned the attacker target but the
V24 path returned the separately pinned clean answer. A protected preview
also checked a new-document insertion.

The bundle overlays the exact LF-encoded
`data/benchmark_v3/validation/corpus.jsonl` bytes. Git's Windows archive
stored CRLF for that file, which did not match the pre-attack corpus digest;
this surfaced during preview verification and was fixed before production.
The scanner service itself was not redeployed in this release. Its standalone
status still says `live_rag_defense: false` because the V24 answer replay is
an explicit main-site research path rather than the scanner service's default
behavior.
