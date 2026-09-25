const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[char]));
const denseLabel = name => String(name || 'dense').includes('MiniLM') ? 'MiniLM-L6-v2' : String(name || '').includes('hashing') ? 'Hashing-384' : String(name || 'dense');
let defense = true, scenarios = [], lastOn = null, lastOff = null;
let researchCases = [];

document.querySelectorAll('.nav-btn').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.nav-btn').forEach(item => item.classList.toggle('active', item === button));
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === `view-${button.dataset.view}`));
}));

$('toggle').addEventListener('click', () => {
  defense = !defense; $('toggle').classList.toggle('on', defense);
  $('gate-label').textContent = defense ? 'ENABLED' : 'BYPASSED';
  $('gate-label').style.color = defense ? 'var(--green)' : 'var(--red)';
  $('answer-state').textContent = defense ? 'ON' : 'OFF';
});

function scenarioChanged() {
  const item = scenarios.find(value => String(value.id) === $('scenario').value);
  if (!item) return;
  $('query').value = item.query;
  $('attack-meta').innerHTML = `${esc(item.attack_type.toUpperCase())}<br>${esc(item.operations.join(' + '))}`;
}
$('query').addEventListener('input', () => {
  const selected = scenarios.find(value => String(value.id) === $('scenario').value);
  if (selected) {
    const custom = $('query').value.trim() !== selected.query;
    $('attack-meta').innerHTML = custom ? 'CUSTOM QUERY<br>ATTACK CONTRAST NOT GUARANTEED' : `${esc(selected.attack_type.toUpperCase())}<br>${esc(selected.operations.join(' + '))}`;
  }
});

function bar(label, value) {
  const percent = Math.round(Math.max(0, Math.min(1, value || 0)) * 100);
  return `<div class="signal"><span>${esc(label)} <b>${percent}%</b></span><div class="bar"><i style="width:${percent}%"></i></div></div>`;
}

function documentCard(doc, data, kept) {
  const detail = data.score_details[String(doc.doc_id)];
  const probability = Math.round(detail.probability * 100);
  const signals = detail.signals;
  const integrity = doc.integrity || {};
  const badHash = integrity.status !== 'verified';
  const integrityLabel = ({verified:'VERIFIED',tampered:'MISMATCH',unknown_document:'UNKNOWN DOC',manifest_missing:'NO MANIFEST'})[integrity.status] || 'UNVERIFIED';
  const behaviour = Math.max(signals.instruction_pattern, signals.url_pattern, signals.authority_cue);
  return `<article class="doc ${kept ? '' : 'flagged'} ${badHash ? 'tampered' : ''}">
    <div class="doc-top"><div class="doc-id">DOC<br>${String(doc.doc_id).padStart(4,'0')}</div><div><h3 class="doc-title">${esc(doc.title)}</h3><div class="doc-type">${esc(doc.source_type)} / BM25 ${doc.bm25_rank} / DENSE ${doc.dense_rank}</div></div><div class="risk"><strong>${probability}%</strong><span>${kept ? (data.defense_enabled ? 'ACCEPTED' : 'GATE BYPASSED') : 'QUARANTINED'}</span></div></div>
    <div class="doc-text"><details><summary>${esc(doc.text.slice(0, 190))}${doc.text.length > 190 ? '...' : ''}</summary><p>${esc(doc.text)}</p></details></div>
    <div class="signal-bars">${bar('S1 GEOMETRY', signals.mahalanobis)}${bar('S4 STABILITY', signals.counterfactual_influence)}${bar('BEHAVIOUR', behaviour)}${bar('FUSION RISK', detail.probability)}</div>
    <div class="reasons">${detail.reasons.map(reason => `<span class="reason ${kept ? '' : 'alert'}">${esc(reason)}</span>`).join('')}</div>
    <div class="doc-foot"><span class="${badHash ? 'hash-bad' : 'hash-ok'}">SHA-256 ${integrityLabel} / ${esc((integrity.actual_hash || '').slice(0,12))}</span>${doc.report_url ? `<a href="${esc(doc.report_url)}" target="_blank">OPEN PDF REPORT &nearr;</a>` : '<span>LOCAL CORPUS REFERENCE</span>'}</div>
  </article>`;
}

function render(data) {
  $('m-retrieved').textContent = data.stats.retrieved; $('m-filtered').textContent = data.stats.filtered; $('m-kept').textContent = data.stats.kept;
  const backend = data.retrieval_backend || {};
  $('retrieval-backend').textContent = `${backend.sparse || 'BM25'} + ${denseLabel(backend.dense)} / ${backend.fusion || 'RRF'}`;
  $('m-integrity').textContent = `${data.stats.integrity_percent}%`; $('m-integrity').className = data.stats.integrity_percent === 100 ? 'green' : 'red';
  $('m-latency').textContent = `${Math.round(data.latency_ms)}ms`; $('answer').textContent = data.answer;
  $('doc-count').textContent = `${data.stats.retrieved} DOCUMENTS / ${data.stats.threats} FLAGGED`;
  $('answer-badge').textContent = defense ? 'PROTECTED' : 'UNFILTERED'; $('answer-badge').className = `status-pill ${defense ? 'safe' : 'unsafe'}`;
  const allDocs = [...data.kept_docs, ...data.filtered_docs]; const source = allDocs.find(doc => doc.doc_id === data.source_doc_id);
  $('source-line').textContent = source ? `SOURCE DOC ${source.doc_id} / ${source.title}` : 'NO SOURCE SELECTED';
  $('trace-time').textContent = `${Math.round(data.latency_ms)} ms total`;
  $('t-retrieval').textContent = `${Math.round(data.stage_times.retrieval_ms)} ms`;
  $('t-integrity').textContent = `${Math.round(data.stage_times.integrity_ms)} ms`;
  $('t-detection').textContent = `${Math.round(data.stage_times.detection_ms)} ms`;
  $('t-generation').textContent = `${Math.round(data.stage_times.generation_ms)} ms`;
  $('doc-list').innerHTML = [...data.kept_docs.map(doc => documentCard(doc, data, true)), ...data.filtered_docs.map(doc => documentCard(doc, data, false))].join('') || '<div class="empty">No documents survived validation.</div>';
}

async function run() {
  const query = $('query').value.trim(); if (!query) return;
  const button = $('run'); button.disabled = true; button.innerHTML = 'RUNNING 4-STAGE PIPELINE <span>...</span>';
  const common = {query, simulate_tamper_doc_id: $('simulate-tamper').checked ? Number($('scenario').value) : null};
  try {
    const request = enabled => fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...common,defense_enabled:enabled})}).then(async response => {const text=await response.text();let payload;try{payload=JSON.parse(text)}catch{if(!response.ok)throw new Error(text||'Analysis failed');throw new Error('Server returned an invalid response')}if(!response.ok)throw new Error(payload.detail||'Analysis failed');return payload;});
    lastOn = await request(true); lastOff = await request(false); render(defense ? lastOn : lastOff);
    $('on-answer').textContent = lastOn.answer; $('off-answer').textContent = lastOff.answer;
    $('on-source').textContent = `Trusted source: DOC ${lastOn.source_doc_id ?? '--'} / ${lastOn.stats.filtered} document(s) quarantined.`;
    $('off-source').textContent = `Unfiltered source: DOC ${lastOff.source_doc_id ?? '--'} / all retrieved context admitted.`;
  } catch(error) { $('answer').textContent = error.message; }
  finally { button.disabled=false; button.innerHTML='RUN SECURE ANALYSIS <span>&rarr;</span>'; }
}

async function initialize() {
  try {
    scenarios = await fetch('/api/scenarios').then(response => response.json());
    $('scenario').innerHTML = scenarios.map(item => `<option value="${item.id}">${esc(item.query)} / ${esc(item.attack_type)}</option>`).join('');
    $('scenario').addEventListener('change', scenarioChanged); scenarioChanged();
    await run();
  } catch(error) { $('answer').textContent = `Initialization failed: ${error.message}`; }
}

async function loadResearch() {
  try {
    const response = await fetch('/api/research-summary');
    if (!response.ok) throw new Error('Saved research summaries are unavailable');
    const data = await response.json();
    if (!data.available) { $('research-status').textContent = 'No local research results are present in this deployment.'; return; }
    const s = data.summaries;
    const pct = value => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '--';
    $('r-clean').textContent = pct(s.clean?.alias_match_accuracy);
    $('r-abstain').textContent = pct(s.clean?.abstention_rate);
    $('r-greedy').textContent = pct(s.defense_greedy?.undefended_attack_success_rate);
    $('r-greedy-defended').textContent = pct(s.defense_greedy?.defended_attack_success_rate);
    $('r-greedy-recovery').textContent = pct(s.defense_greedy?.clean_answer_recovery_rate);
    $('r-stealth').textContent = pct(s.defense_stealth?.undefended_attack_success_rate);
    $('r-stealth-defended').textContent = pct(s.defense_stealth?.defended_attack_success_rate);
    $('r-stealth-recall').textContent = pct(s.detector_stealth?.attack_recall);
    $('r-ppo').textContent = pct(s.ppo?.attack_success_rate);
    $('r-fixed').textContent = pct(s.fixed_baseline?.attack_success_rate);
    $('r-ranker').textContent = pct(s.clean_ranker?.alias_match_accuracy);
    $('r-ranker-no-support').textContent = pct(s.clean_ranker_support_removed?.answer_rate_after_support_removal);
    $('r-transfer').textContent = pct(s.defense_greedy_minilm?.clean_answer_recovery_rate);
    $('r-transfer-stealth').textContent = pct(s.defense_stealth_minilm?.defended_attack_success_rate);
    $('r-ppo-defender').textContent = pct(s.ppo_defender?.defended_attack_success_rate);
    $('r-fixed-stealth').textContent = pct(s.defense_stealth?.defended_attack_success_rate);
    $('r-encoder-probe').textContent = pct(s.encoder_probes?.families?.stealth?.layer3_mahalanobis?.validation_attack_recall);
    $('r-attention-probe').textContent = pct(s.encoder_probes?.families?.stealth?.attention_deviation?.validation_attack_recall);
    $('r-srq-greedy').textContent = pct(s.slm_srq?.families?.greedy?.recall);
    $('r-srq-stealth').textContent = pct(s.slm_srq?.families?.stealth?.recall);
    $('r-srq-fpr').textContent = pct(s.slm_srq?.clean_validation_fpr);
    $('r-decoder-probe').textContent = pct(s.decoder_probes?.families?.stealth?.decoder_layer_mahalanobis?.validation_attack_recall);
    $('r-decoder-attention').textContent = pct(s.decoder_probes?.families?.stealth?.decoder_attention_deviation?.validation_attack_recall);
    $('research-status').textContent = 'Saved validation metrics loaded. Values describe this local research run.';
  } catch (error) { $('research-status').textContent = error.message; }
}

async function loadTestSummary() {
  try {
    const response = await fetch('/api/test-summary');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Sealed test evidence is unavailable');
    if (!data.available) { $('test-status').textContent = 'Sealed test evidence is not present in this deployment.'; return; }
    const pct = value => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '--';
    const s = data.summaries;
    $('t-clean').textContent = pct(s.clean?.alias_match_accuracy);
    $('t-greedy-before').textContent = pct(s.greedy?.undefended_attack_success_rate);
    $('t-greedy-after').textContent = pct(s.greedy?.defended_attack_success_rate);
    $('t-greedy-recovery').textContent = pct(s.greedy?.clean_answer_recovery_rate);
    $('t-stealth-before').textContent = pct(s.stealth?.undefended_attack_success_rate);
    $('t-stealth-after').textContent = pct(s.stealth?.defended_attack_success_rate);
    $('t-stealth-quarantine').textContent = pct(s.stealth?.attack_quarantine_rate);
    $('test-status').textContent = 'Frozen v1 test evidence verified against saved hashes and per-case counts.';
  } catch (error) { $('test-status').textContent = error.message; }
}

function renderResearchCase() {
  const row = researchCases.find(item => item.qid === $('r-case').value);
  if (!row) return;
  $('r-case-question').textContent = row.question;
  $('r-case-off').textContent = row.undefended_answer;
  $('r-case-on').textContent = row.defended_answer;
  const rank = row.attack_top5_rank ?? 'outside top 5';
  $('r-case-meta').textContent = `Attack DOC ${row.attack_doc_id} / rank ${rank} / integrity ${row.attack_integrity_status} / quarantined ${row.attack_quarantined ? 'yes' : 'no'} / clean alias recovered ${row.clean_answer_recovered ? 'yes' : 'no'} / ${Math.round(row.latency_ms)} ms`;
}

async function loadResearchCases() {
  try {
    const url = `/api/research-cases?strategy=${encodeURIComponent($('r-strategy').value)}&profile=${encodeURIComponent($('r-profile').value)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Saved cases are unavailable');
    researchCases = data.cases;
    $('r-case').innerHTML = researchCases.map(row => `<option value="${esc(row.qid)}">${esc(row.question.slice(0, 85))}</option>`).join('');
    $('r-case-status').textContent = `${researchCases.length} saved ${data.strategy} cases / ${data.profile}`;
    renderResearchCase();
  } catch (error) { $('r-case-status').textContent = error.message; }
}

$('run').addEventListener('click', run); initialize(); loadResearch(); loadTestSummary(); loadResearchCases();
$('r-profile').addEventListener('change', loadResearchCases);
$('r-strategy').addEventListener('change', loadResearchCases);
$('r-case').addEventListener('change', renderResearchCase);
