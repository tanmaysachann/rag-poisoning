# V26 hosted research scanner

The main site is `https://rag-poisoning.vercel.app`. It sends `/api/v26/*`
requests server-side to the separate public container service at
`https://sentinel-rag-v26.vercel.app` through its `V26_REMOTE_URL` setting.
The V24 and V25 selectors remain available. The V26 container includes the
frozen trained role component, the earlier V25/V24 models, and the pinned
2.68-million-passage BEIR NQ source lookup rebuilt from the official archive
at image build. The lookup is for source containment, not supervised labels.

## Rebuild

From the repository root, stage new bundle directories:

```powershell
python -m deploy.build_v24_bundle --output tmp/v26_service_release
python -m deploy.build_site_bundle --output tmp/site_v26_release
```

Run `vercel deploy` **from inside** the staged service bundle linked to
`sentinel-rag-v26`, and from inside the site bundle linked to
`rag-poisoning`. The service project needs SSO deployment protection disabled
so the site's server-side proxy can reach it. Set `V26_REMOTE_URL` on the
site's Production and Preview environments to the service HTTPS origin.

The container's `/api/v26/status` checks the frozen code, model, V25
calibration, FiQA report, targeted role stress, and answer replay bindings.
The browser cannot trigger training or benchmark scoring.

## Verify

Check both the service origin and main site:

```text
GET  /api/v26/status       available: true
GET  /api/v26/evidence     first FiQA: 600/600 for V25 and V26; targeted role: 0/120 V25, 120/120 V26
POST /api/v26/scan         query, reference, candidate, optional expected_sha256
GET  /api/v25/status       available: true
GET  /api/v24/status       available: true
```

`python -m evaluation.prepare_v26_smoke` saves a pinned targeted FiQA
role-delimiter attack and its original clean reference in
`tmp/v26_smoke_{attack,clean}.json`. The attack should pass V25 and request
review on V26; the clean reference should pass both. Confirm the site
selector, evidence text, and API responses. A browser rendering check is
useful, but API and static checks are sufficient to establish the deployment
route when browser automation is unavailable.

The first FiQA scanner result has no V26 improvement over V25. The targeted
role improvement is a post-hoc known-pattern stress, and the answer replay
has poor clean utility (119/120 gated abstentions). Present both limitations
with the favorable role result.

## Verified production release (20 October 2026)

Commit `31145c3` was pushed to `origin/main`. The V26 container deployment
`dpl_XeHnbvDDCz1WxpovfTQWssETk1tU` and main site deployment
`dpl_6EjQd78SzXMpPKfmdSuiy7HoWJfh` reached `READY` and were aliased to
the production hostnames above. The container rebuilt the pinned NQ snapshot
with 2,681,468 passages and the expected database SHA-256.

Direct service and main-site proxy requests both returned V26
`available: true`. The live evidence response returned 600/600 for V25 and
V26 on the first FiQA challenge, 0/120 versus 120/120 on the follow-up role
stress, and 119/120 V26 answer-path abstentions. The saved attack returned
V25 `pass` and V26 `review`; its clean reference returned `pass` on both,
through both hostnames. Main-site V24 and V25 status endpoints also returned
`available: true`; the page HTML and served JavaScript contained the V26
selector and evidence loader. The JavaScript passed `node --check`, and the
new scanner and answer-gate unit tests passed. A rendered browser screenshot
was not available in this session.
