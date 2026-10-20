/* Paired V24/V25 scanners plus an explicitly selected research answer replay. */
(() => {
  const el = id => document.getElementById(id);
  const percent = value => `${(100 * Number(value)).toFixed(1)}%`;
  const available = {v24: false, v25: false};

  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }

  async function loadStatus() {
    const results = await Promise.allSettled([
      request('/api/v24/status'), request('/api/v25/status'),
    ]);
    available.v24 = results[0].status === 'fulfilled' && results[0].value.available;
    available.v25 = results[1].status === 'fulfilled' && results[1].value.available;
    for (const version of ['v24', 'v25']) {
      el('v24-model').querySelector(`option[value="${version}"]`).disabled = !available[version];
    }
    if (!available.v25 && available.v24) el('v24-model').value = 'v24';
    updateVersionAvailability();
  }

  function updateVersionAvailability() {
    const version = el('v24-model').value;
    const ready = Boolean(available[version]);
    el('v24-ready').textContent = ready ? `${version.toUpperCase()} model + snapshot ready` : `${version.toUpperCase()} assets unavailable`;
    el('v24-run').disabled = !ready;
    if (!ready) {
      el('v24-status').dataset.state = 'error';
      el('v24-status').textContent = `The selected ${version.toUpperCase()} scanner is unavailable.`;
    }
  }

  async function loadEvidence() {
    try {
      const data = await request('/api/v24/evidence');
      const combined = data.product.combined;
      el('v24-hero-result').textContent = `${combined.poison_caught}/${combined.poison_total} attacks flagged; ${combined.clean_flags}/${combined.clean_total} clean passages flagged.`;
      el('v24-recall').textContent = percent(combined.poison_recall);
      el('v24-recall-detail').textContent = `${combined.poison_caught} of ${combined.poison_total} published attack passages`;
      el('v24-clean').textContent = `${combined.clean_flags}/${combined.clean_total}`;
      const rounds = data.adaptive.rounds;
      el('v24-ppo').textContent = `Adaptive PPO: two 200-episode attacker runs on 350 MS MARCO training questions. On 75 validation questions, the first policy had ${rounds[0].frozen_policy_against_training_defender.defended_successes}/75 defended successes; replaying the same attacks against source containment gave ${rounds[0].same_policy_against_next_defender.defended_successes}/75. A new policy trained against containment had ${rounds[1].frozen_policy_against_training_defender.defended_successes}/75; the known-ID hash rule reduced that same-policy replay to ${rounds[1].same_policy_against_next_defender.defended_successes}/75. The hash rule applies only when the document's expected digest was pinned separately.`;
    } catch (error) {
      el('v24-hero-result').textContent = 'Saved report unavailable';
      el('v24-ppo').textContent = String(error.message);
    }
  }

  async function loadExample() {
    // A sample is only a published benchmark record, never an authenticated source.
    const button = el('v24-example');
    button.disabled = true;
    try {
      const row = await request('/api/v24/sample');
      el('v24-query').value = row.query;
      el('v24-reference').value = row.reference;
      el('v24-candidate').value = row.candidate;
      el('v24-hash').value = '';
      el('v24-status').dataset.state = 'ready';
      el('v24-status').textContent = 'Loaded a published PoisonArena attack and its benchmark reference. The reference is a benchmark label, not an independently authenticated source.';
    } catch (error) {
      el('v24-status').dataset.state = 'error';
      el('v24-status').textContent = String(error.message);
    } finally { button.disabled = false; }
  }

  async function loadResearchEvidence() {
    try {
      const data = await request('/api/v24/research-evidence');
      const transfer = data.transfer.metrics.v24;
      const initial = data.answer_initial.summary.overall;
      const admitted = data.answer_admitted.summary.overall;
      el('v24-transfer-attack').textContent = `${transfer.attack_recall.flagged}/${transfer.attack_recall.total}`;
      el('v24-transfer-clean').textContent = `${transfer.clean_false_alarm.flagged}/${transfer.clean_false_alarm.total} clean passages flagged; inserted documents have no known-ID hash`;
      el('v24-seeds').textContent = data.ppo_multiseed.rounds.map(row => `${row.v23_defended_successes}/75`).join(' · ');
      el('v24-answer-initial').textContent = `${initial.v24_attack_successes}/${initial.cases}`;
      el('v24-answer-admitted').textContent = `${admitted.v24_attack_successes}/${admitted.cases}`;
      el('v24-research-note').textContent = 'SciFact uses synthetic fixed-template attack labels, not independent factual review. The admitted answer gate was revised after the initial failure and replayed on previously inspected validation cases. PPO trains the attacker; the defender is supervised plus source integrity.';
    } catch (error) {
      el('v24-research-note').textContent = `Research report unavailable: ${error.message}`;
    }
  }

  async function loadV25Evidence() {
    try {
      const data = await request('/api/v25/evidence');
      const current = data.holdout.metrics.v25;
      const baseline = data.holdout.metrics.v24;
      el('v25-holdout-attack').textContent = `${current.attack_recall.flagged}/${current.attack_recall.total}`;
      el('v25-holdout-clean').textContent = `${current.clean_false_alarm.flagged}/${current.clean_false_alarm.total} original clean passages flagged`;
      el('v25-baseline-attack').textContent = `${baseline.attack_recall.flagged}/${baseline.attack_recall.total}`;
      el('v25-baseline-clean').textContent = `${baseline.clean_false_alarm.flagged}/${baseline.clean_false_alarm.total} original clean passages flagged`;
    } catch (error) {
      el('v25-holdout-clean').textContent = `V25 report unavailable: ${error.message}`;
    }
  }

  async function scan() {
    const version = el('v24-model').value;
    const body = {
      query: el('v24-query').value.trim(),
      reference: el('v24-reference').value.trim(),
      candidate: el('v24-candidate').value.trim(),
    };
    const hash = el('v24-hash').value.trim();
    if (hash) body.expected_sha256 = hash;
    if (!body.query || !body.reference || !body.candidate) {
      el('v24-status').dataset.state = 'error';
      el('v24-status').textContent = 'Enter a question, trusted reference, and candidate passage.';
      return;
    }
    const button = el('v24-run');
    button.disabled = true;
    el('v24-status').dataset.state = '';
    el('v24-status').textContent = `Running ${version.toUpperCase()} and the pinned source lookup. The first request may take longer while assets load.`;
    try {
      const data = await request(`/api/${version}/scan`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
      });
      el('v24-result').hidden = false;
      el('v24-mode').textContent = version === 'v25' ? 'V25 reference adaptive' : 'V24 baseline';
      el('v24-decision').textContent = data.decision === 'review' ? 'REVIEW THIS PASSAGE' : 'PASS — NO FLAG AT THIS THRESHOLD';
      el('v24-decision').dataset.decision = data.decision;
      el('v24-learned').textContent = data.learned.flag ? 'Flagged' : 'No flag';
      el('v24-learned-detail').textContent = `Fusion score ${data.learned.fusion_score.toFixed(3)}; threshold ${data.learned.threshold.toFixed(3)}. Model score is a source-label signal, not a probability that the facts are false.`;
      el('v24-integrity').textContent = data.integrity.flag ? 'Changed source found' : 'No changed source found';
      el('v24-integrity-detail').textContent = `${data.integrity.status}; similarity ${Number(data.integrity.near_copy_score || 0).toFixed(3)}; ${data.integrity.match_method || 'no match method'}.`;
      el('v24-identity').textContent = data.identity.status === 'not_supplied' ? 'Not supplied' : data.identity.flag ? 'Digest mismatch' : 'Digest match';
      el('v24-identity-detail').textContent = data.identity.status === 'not_supplied' ? 'Requires an independently pinned hash for this document ID.' : 'Compares the candidate bytes with the supplied expected SHA-256.';
      el('v25-reasons').hidden = version !== 'v25';
      if (version === 'v25') {
        el('v25-v3').textContent = data.v3.flag ? 'Flagged' : 'No flag';
        el('v25-v3-detail').textContent = `Behavioral risk ${data.v3.risk.toFixed(3)}; threshold ${data.v3.threshold.toFixed(3)}.`;
        el('v25-relative').textContent = data.reference_adaptive.flag ? 'Flagged' : 'No flag';
        el('v25-relative-detail').textContent = `Candidate minus reference fusion score ${data.reference_adaptive.margin.toFixed(3)}; required margin ${data.reference_adaptive.required_margin.toFixed(3)}. The reference must be authenticated separately.`;
      }
      el('v24-status').dataset.state = 'ready';
      el('v24-status').textContent = 'Scan complete. Inspect the reasons before deciding what to do with the passage.';
    } catch (error) {
      el('v24-result').hidden = true;
      el('v24-status').dataset.state = 'error';
      el('v24-status').textContent = String(error.message);
    } finally { button.disabled = false; }
  }

  async function loadRagCases() {
    try {
      const data = await request('/api/lab/cases');
      const select = el('v24-rag-case');
      for (const row of data.cases) {
        const option = document.createElement('option');
        option.value = row.qid;
        option.textContent = `${row.qid} · ${row.question}`;
        select.appendChild(option);
      }
      select.value = data.default_qid;
      el('v24-rag-status').textContent = `${data.cases.length} saved validation questions available. Select one and run the paired answer path.`;
    } catch (error) {
      el('v24-rag-status').dataset.state = 'error';
      el('v24-rag-status').textContent = String(error.message);
      el('v24-rag-run').disabled = true;
    }
  }

  async function runRag() {
    const button = el('v24-rag-run');
    button.disabled = true;
    el('v24-rag-result').hidden = true;
    el('v24-rag-status').dataset.state = '';
    el('v24-rag-status').textContent = 'Building an isolated retrieval index, then screening its top five passages with V24...';
    try {
      const data = await request('/api/lab/run', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          qid: el('v24-rag-case').value,
          strategy: el('v24-rag-strategy').value,
          surface: 'accepted_ingest', attack_budget: 1, v24_gate: true,
        }),
      });
      const v24 = data.v24_answer_path;
      if (!v24) throw new Error('V24 answer path was not returned.');
      el('v24-rag-off').textContent = data.undefended.answer;
      el('v24-rag-v3').textContent = data.defended.answer;
      el('v24-rag-on').textContent = v24.defended.answer;
      el('v24-rag-off-source').textContent = `Source ${data.undefended.source_doc_id ?? 'none'}`;
      el('v24-rag-v3-source').textContent = `Source ${data.defended.source_doc_id ?? 'none'}`;
      el('v24-rag-on-source').textContent = `Source ${v24.defended.source_doc_id ?? 'none'}`;
      const recovered = v24.snapshot_recovery.decision === 'recover' ? 'yes' : 'no';
      el('v24-rag-detail').textContent = `${v24.retrieved_attack_docs_reviewed}/${data.attack_retrieved_count} retrieved attack passages reviewed or denied admission. Pinned snapshot recovery: ${recovered}. Attack-sourced target answer: V3 ${data.defended_attack_success ? 'yes' : 'no'}, admitted V24 ${v24.attack_source_success ? 'yes' : 'no'}. Clean answer alias recovered: V3 ${data.clean_alias_recovered ? 'yes' : 'no'}, admitted V24 ${v24.clean_alias_recovered ? 'yes' : 'no'}.`;
      el('v24-rag-time').textContent = `${Math.round(data.latency_ms)} ms total`;
      el('v24-rag-result').hidden = false;
      el('v24-rag-status').dataset.state = 'ready';
      el('v24-rag-status').textContent = 'Paired replay complete. The reference came from this benchmark’s pinned pre-attack corpus.';
    } catch (error) {
      el('v24-rag-status').dataset.state = 'error';
      el('v24-rag-status').textContent = String(error.message);
    } finally { button.disabled = false; }
  }

  el('v24-example').addEventListener('click', loadExample);
  el('v24-model').addEventListener('change', updateVersionAvailability);
  el('v24-run').addEventListener('click', scan);
  if (el('v24-rag-run')) el('v24-rag-run').addEventListener('click', runRag);
  loadStatus();
  loadEvidence();
  if (el('v24-transfer-attack')) loadResearchEvidence();
  if (el('v25-holdout-attack')) loadV25Evidence();
  if (el('v24-rag-case')) loadRagCases();
})();
