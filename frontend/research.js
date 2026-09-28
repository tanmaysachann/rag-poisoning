/* Interactive validation lab. The server executes retrieval and detection; this file only renders it. */
(() => {
  const el = id => document.getElementById(id);
  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percent = value => `${Math.round(Number(value || 0) * 100)}%`;
  const count = (rate, n = 75) => `${Math.round(Number(rate || 0) * n)}/${n}`;
  let cases = [];

  async function jsonRequest(url, options) {
    const response = await fetch(url, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    return body;
  }

  function selectedCase() { return cases.find(row => row.qid === el('lab-case').value); }

  function resetPayload() {
    const item = selectedCase();
    if (!item) return;
    el('lab-payload').value = el('lab-strategy').value === 'stealth' ? item.stealth_text : item.greedy_text;
  }

  function chooseCase() {
    const item = selectedCase();
    if (!item) return;
    el('lab-original').textContent = item.original_text;
    el('lab-case-note').textContent = item.saved_stealth_failure
      ? 'Saved hashing run: the answer substitution passed the defense. Edit the payload to probe why.'
      : 'Saved hashing run: this substitution did not produce a defended wrong answer. You can edit the payload and rerun.';
    resetPayload();
    loadPpoCase(item.qid);
  }

  function featureCard(name, value) {
    return `<div class="lab-feature"><span>${escapeHtml(name.replaceAll('_', ' ').toUpperCase())}</span><strong>${Number(value).toFixed(3)}</strong></div>`;
  }

  function documentCard(doc) {
    const risk = Math.max(0, Math.min(100, Math.round(doc.risk * 100)));
    const featureRows = Object.entries(doc.features).map(([key, value]) => featureCard(key, value)).join('');
    return `<details class="lab-doc ${doc.is_attack ? 'attack' : ''} ${doc.decision === 'quarantine' ? 'quarantined' : 'accepted'}" ${doc.is_attack ? 'open' : ''}>
      <summary><span>#${doc.rank}</span><strong>DOC ${doc.doc_id}${doc.is_attack ? ' / ALTERED' : ''}</strong><span>RISK ${risk}%</span><em>${doc.decision.toUpperCase()}</em></summary>
      <div class="lab-doc-body"><div class="lab-doc-meta"><span>BM25 RANK ${doc.bm25_rank}</span><span>DENSE RANK ${doc.dense_rank}</span><span>RRF ${Number(doc.rrf_score).toFixed(4)}</span><span>SHA-256 ${escapeHtml(doc.integrity.toUpperCase())}</span></div><p>${escapeHtml(doc.text)}</p><div class="lab-risk-track"><i style="width:${risk}%"></i></div><div class="lab-feature-grid">${featureRows}</div><div class="lab-doc-meta" style="margin-top:12px">SHA-256 ${escapeHtml(doc.sha256)}</div></div>
    </details>`;
  }

  function renderRun(data) {
    el('lab-time').textContent = `${Math.round(data.latency_ms)} MS / ${data.execution.replaceAll('_', ' ').toUpperCase()}`;
    el('lab-status').dataset.state = 'ok';
    el('lab-status').textContent = `${data.documents.length} passages retrieved · attack ${data.attack_retrieved ? 'entered top five' : 'missed top five'} · threshold ${Number(data.detector_threshold).toFixed(3)}`;
    el('lab-off-answer').textContent = data.undefended.answer;
    el('lab-on-answer').textContent = data.defended.answer;
    el('lab-clean-answer').textContent = data.clean.answer;
    el('lab-off-source').textContent = `SOURCE DOC ${data.undefended.source_doc_id ?? 'NONE'}`;
    el('lab-on-source').textContent = `SOURCE DOC ${data.defended.source_doc_id ?? 'NONE'}`;
    el('lab-clean-source').textContent = `SOURCE DOC ${data.clean.source_doc_id ?? 'NONE'} · DEFENDED ANSWER ${data.defended_answer_matches_clean ? 'MATCHES' : 'DIFFERS'}`;
    const verdict = el('lab-verdict');
    if (data.defended_attack_success) {
      verdict.className = 'lab-verdict failure';
      verdict.textContent = 'Observed failure: the altered passage supplied the defended wrong answer. The integrity check and detector did not remove it.';
    } else if (data.attack_quarantined) {
      verdict.className = 'lab-verdict';
      verdict.textContent = `Attack document quarantined. ${data.surface === 'post_index_tamper' ? 'The protected clean snapshot detected a post-index change.' : 'The research detector flagged the accepted document.'}`;
    } else if (data.defended_source_is_attack) {
      verdict.className = 'lab-verdict failure';
      verdict.textContent = 'The altered document remained the defended answer source. The saved target phrase was not returned; inspect the answer and compare with the original passage.';
    } else {
      verdict.className = 'lab-verdict';
      verdict.textContent = data.attack_retrieved ? 'The altered document was retrieved but did not supply the defended answer.' : 'The altered document did not reach the top five retrieved passages.';
    }
    const citation = data.defended.citations?.[0];
    const timing = data.stage_times || {};
    const changed = (data.counterfactuals || []).filter(row => row.answer_changed);
    el('lab-inference-details').innerHTML = `<p><b>Selected evidence:</b> ${citation ? `DOC ${escapeHtml(citation.doc_id)} / ${escapeHtml(citation.span)}` : 'Abstained - no accepted source span.'}</p>
      <p><b>Backend:</b> ${escapeHtml(data.defended.backend)} · <b>Stage time:</b> setup + indexing ${Number(timing.setup_and_index_ms || 0).toFixed(1)} ms, retrieval ${Number(timing.retrieval_ms || 0).toFixed(1)} ms, integrity ${Number(timing.integrity_ms || 0).toFixed(1)} ms, detector ${Number(timing.detection_ms || 0).toFixed(1)} ms, answer + leave-one-out ${Number(timing.answer_and_loo_ms || 0).toFixed(1)} ms.</p>
      <p><b>Independent-origin check:</b> ${data.provenance?.enforced ? escapeHtml(data.provenance.reason) : 'Not applied; this benchmark has no reviewed independent source origins.'}</p>
      <details><summary>${changed.length} of ${(data.counterfactuals || []).length} accepted passages changed the answer when removed</summary><ul>${(data.counterfactuals || []).map(row => `<li>Remove DOC ${escapeHtml(row.removed_doc_id)}: ${row.answer_changed ? 'answer changed to' : 'answer unchanged'} ${escapeHtml(row.answer_after_removal)}</li>`).join('')}</ul></details>`;
    el('lab-candidate-count').textContent = `${data.documents.length} CANDIDATES`;
    el('lab-documents').innerHTML = data.documents.map(documentCard).join('');
  }

  async function runLab() {
    const button = el('lab-run');
    button.disabled = true;
    el('lab-status').dataset.state = 'loading';
    el('lab-status').textContent = 'Staging document, building index, scoring eight features…';
    try {
      const data = await jsonRequest('/api/lab/run', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          qid: el('lab-case').value,
          strategy: el('lab-strategy').value,
          surface: el('lab-surface').value,
          attack_text: el('lab-payload').value,
        }),
      });
      renderRun(data);
    } catch (error) {
      el('lab-status').dataset.state = 'error';
      el('lab-status').textContent = error.message;
    } finally { button.disabled = false; }
  }

  async function loadCases() {
    try {
      const data = await jsonRequest('/api/lab/cases');
      cases = data.cases;
      el('lab-case').innerHTML = cases.map(row => `<option value="${escapeHtml(row.qid)}">${row.saved_stealth_failure ? '● ' : ''}${escapeHtml(row.question)} / ${escapeHtml(row.qid)}</option>`).join('');
      el('lab-case').value = data.default_qid;
      chooseCase();
      await runLab();
    } catch (error) {
      el('lab-status').dataset.state = 'error';
      el('lab-status').textContent = error.message;
    }
  }

  function barRow(label, value, isAfter = false) {
    const width = Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));
    return `<div class="lab-bar-row ${isAfter ? 'after' : ''}"><span>${label}</span><div><i style="width:${width}%"></i></div><b>${width}%</b></div>`;
  }

  async function loadTest() {
    try {
      const data = await jsonRequest('/api/test-summary');
      if (!data.available) throw new Error('Sealed test evidence is unavailable');
      const {clean, greedy, stealth} = data.summaries;
      el('lab-greedy-result').textContent = `${count(greedy.undefended_attack_success_rate)} → ${count(greedy.defended_attack_success_rate)}`;
      el('lab-stealth-result').textContent = `${count(stealth.undefended_attack_success_rate)} → ${count(stealth.defended_attack_success_rate)}`;
      el('lab-clean-result').textContent = count(clean.alias_match_accuracy);
      el('lab-greedy-bars').innerHTML = barRow('BEFORE', greedy.undefended_attack_success_rate) + barRow('AFTER', greedy.defended_attack_success_rate, true);
      el('lab-stealth-bars').innerHTML = barRow('BEFORE', stealth.undefended_attack_success_rate) + barRow('AFTER', stealth.defended_attack_success_rate, true);
      el('lab-clean-bars').innerHTML = barRow('MATCH', clean.alias_match_accuracy, true);
      el('lab-test-status').textContent = 'First frozen test setup (v1): 75 questions per attack family. The server verified file hashes and case counts. This test is historical evidence for that setup.';
    } catch (error) { el('lab-test-status').textContent = error.message; }
  }

  function plotPpo(plain, defended) {
    const x = index => 47 + index * 69;
    const y = value => 192 - Math.max(0, Math.min(1, Number(value || 0))) * 159;
    const path = (rows, key) => rows.map((row, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(row[key])}`).join(' ');
    const grid = [0, .5, 1].map(v => `<line x1="47" y1="${y(v)}" x2="690" y2="${y(v)}" stroke="#d5dce2" stroke-dasharray="${v ? '3 4' : '0'}"/><text x="5" y="${y(v) + 4}" fill="#5d6b76" font-size="10">${Math.round(v * 100)}%</text>`).join('');
    const ticks = plain.map((_, index) => `<text x="${x(index)}" y="218" text-anchor="middle" fill="#5d6b76" font-size="10">${(index + 1) * 20}</text>`).join('');
    const points = (rows, key, color) => rows.map((row, index) => `<circle cx="${x(index)}" cy="${y(row[key])}" r="4" fill="${color}"><title>Batch ${index + 1}: ${Math.round(row[key] * 100)}%</title></circle>`).join('');
    el('lab-ppo-chart').innerHTML = `${grid}<path d="${path(plain, 'attack_success_rate')}" fill="none" stroke="#155a8a" stroke-width="3"/>${points(plain, 'attack_success_rate', '#155a8a')}<path d="${path(defended, 'defended_attack_success_rate')}" fill="none" stroke="#a63732" stroke-width="3"/>${points(defended, 'defended_attack_success_rate', '#a63732')}${ticks}<text x="688" y="238" text-anchor="end" fill="#5d6b76" font-size="10">EPISODES</text>`;
  }

  async function loadPpo() {
    try {
      const data = await jsonRequest('/api/lab/ppo');
      plotPpo(data.runs.undefended, data.runs.defender_aware);
      el('lab-ppo-plain').textContent = count(data.validation.undefended.attack_success_rate);
      el('lab-ppo-defender').textContent = count(data.validation.defender_aware.defended_attack_success_rate);
      el('lab-ppo-fixed').textContent = count(data.validation.fixed_insertion.attack_success_rate);
      // The matched fixed substitution is the frozen hashing defense baseline.
      const summary = await jsonRequest('/api/research-summary');
      el('lab-ppo-stealth-fixed').textContent = count(summary.summaries.defense_stealth?.defended_attack_success_rate);
      const multi = summary.summaries.ppo_multiseed;
      if (multi) el('lab-ppo-multiseed').textContent = `Three matched validation seeds, 75 questions each: PPO ${multi.ppo_successes}/${multi.total_paired_cases} defended successes; fixed answer substitution ${multi.fixed_substitution_successes}/${multi.total_paired_cases}; random edits ${multi.random_edit_successes}/${multi.total_paired_cases}. PPO minus fixed: ${Math.round(multi.paired_difference_rate * 1000) / 10} percentage points (query-cluster bootstrap 95% interval ${multi.query_cluster_bootstrap_95pct_difference.map(value => `${Math.round(value * 1000) / 10}`).join(' to ')}). PPO beat random edits but did not beat fixed substitution.`;
      const ablation = summary.summaries.ppo_detection_ablation;
      if (ablation) el('lab-ppo-ablation').textContent = `Reward ablation, same three seeds and questions: with detector-risk step shaping ${ablation.original_successes}/${ablation.total_paired_cases} defended successes; without it ${ablation.no_detection_reward_successes}/${ablation.total_paired_cases}. The terminal defended-success reward remained in both runs. This development result supports the shaping term for this PPO setup, not a general policy advantage.`;
      const proxyAblation = summary.summaries.ppo_proxy_ablation;
      if (proxyAblation) el('lab-ppo-proxy-ablation').textContent = `Auxiliary critic ablation, same seeds and questions: ${proxyAblation.original_successes}/${proxyAblation.total_paired_cases} defended successes with its training loss, ${proxyAblation.no_proxy_value_successes}/${proxyAblation.total_paired_cases} without. No deterministic validation case outcome changed.`;
      const cacheAblation = summary.summaries.ppo_cache_ablation;
      if (cacheAblation) el('lab-ppo-cache-ablation').textContent = `Edit-effect cache ablation, same seeds and questions: ${cacheAblation.original_successes}/${cacheAblation.total_paired_cases} defended successes without cache shaping, ${cacheAblation.cache_successes}/${cacheAblation.total_paired_cases} with it. No deterministic validation case outcome changed. The cache predicted step reward, not final attack success.`;
      const headAblation = summary.summaries.ppo_head_ablation;
      if (headAblation) el('lab-ppo-head-ablation').textContent = `Action-head conditioning ablation, same seeds and questions: ${headAblation.original_successes}/${headAblation.total_paired_cases} defended successes with chosen-action inputs, ${headAblation.no_head_conditioning_successes}/${headAblation.total_paired_cases} without. No deterministic validation case outcome changed; action masks remained active.`;
      const v2 = summary.summaries.v2_paired_gate;
      if (v2) el('lab-v2-summary').textContent = `On 30 validation questions, the full-context cosine gate left attack success at ${v2.attack_success_before_gate}/${v2.cases} → ${v2.attack_success_after_gate}/${v2.cases} while clean alias matches fell ${v2.clean_answer_alias_before_gate}/${v2.cases} → ${v2.clean_answer_alias_after_gate}/${v2.cases}. It was not deployed.`;
      const layouts = summary.summaries.unseen_templates?.styles;
      if (layouts) el('lab-unseen-summary').textContent = `Three overt answer layouts were tested on the same 75 development questions. All ${Object.values(layouts).reduce((total, row) => total + row.cases, 0)} altered documents were quarantined; defended attack success was ${Object.values(layouts).reduce((total, row) => total + row.defended_attack_successes, 0)}. These probes do not establish unseen-source robustness.`;
      const budgets = summary.summaries.poison_budget?.budgets;
      if (budgets) el('lab-budget-summary').textContent = `With one, two, or three identical accepted stealth passages, defended wrong answers occurred in ${budgets['1'].defended_attack_successes}/75, ${budgets['2'].defended_attack_successes}/75, and ${budgets['3'].defended_attack_successes}/75 validation cases. Copies share one attacker origin; this is not a varied-attack guarantee.`;
    } catch (error) { el('lab-ppo-chart').outerHTML = `<p class="lab-evidence-caption">${escapeHtml(error.message)}</p>`; }
  }

  async function loadPpoCase(qid) {
    el('lab-ppo-case-label').textContent = qid;
    try {
      const data = await jsonRequest(`/api/lab/ppo-case?qid=${encodeURIComponent(qid)}`);
      el('lab-ppo-traces').innerHTML = Object.entries(data.runs).map(([name, run]) => {
        const steps = run.trace.map(step => `<li>${escapeHtml(step.operation)} / position ${escapeHtml(step.position)} / payload ${escapeHtml(step.payload)} / reward ${Number(step.reward).toFixed(2)}</li>`).join('');
        const result = run.defended_attack_success == null ? `Undefended attack success: ${run.attack_success ? 'yes' : 'no'}` : `Defended attack success: ${run.defended_attack_success ? 'yes' : 'no'} · detector flagged: ${run.attack_doc_detected ? 'yes' : 'no'}`;
        return `<div class="lab-policy-trace"><span>${escapeHtml(name.replaceAll('_', ' ').toUpperCase())}</span><h4>${escapeHtml(result)}</h4><ol>${steps}</ol><details><summary>Show final edited passage</summary><p>${escapeHtml(run.final_document)}</p></details></div>`;
      }).join('');
    } catch (error) { el('lab-ppo-traces').textContent = error.message; }
  }

  el('lab-case').addEventListener('change', chooseCase);
  el('lab-strategy').addEventListener('change', resetPayload);
  el('lab-reset').addEventListener('click', resetPayload);
  el('lab-run').addEventListener('click', runLab);
  loadCases(); loadTest(); loadPpo();
})();
