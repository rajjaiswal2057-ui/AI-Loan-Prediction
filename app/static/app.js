/* AI Loan Prediction - front-end. All numbers shown come from the Flask API / saved training results. */
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const S = {data: null, last: null, input: null, inited: {}, charts: {}, simKey: ""};
const COL = {pri: "#1d4ed8", ok: "#15803d", bad: "#b91c1c", warn: "#d97706", mut: "#94a3b8", pur: "#7c3aed"};
const LVL = {LOW: COL.ok, MEDIUM: COL.warn, HIGH: COL.bad};
Chart.defaults.font.family = 'Inter,-apple-system,"Segoe UI",sans-serif'; Chart.defaults.font.size = 11.5; Chart.defaults.color = "#64748b";
Chart.defaults.maintainAspectRatio = false; Chart.defaults.plugins.legend.labels.boxWidth = 10;

const money = v => v >= 1e7 ? `₹${(v / 1e7).toFixed(2)} Cr` : `₹${(v / 1e5).toFixed(1)} L`;
const pct = v => (v * 100).toFixed(1) + "%";
function mk(id, cfg) { if (S.charts[id]) S.charts[id].destroy(); S.charts[id] = new Chart($("#" + id), cfg); }
const bar = (id, labels, sets, extra = {}) => mk(id, {type: "bar", data: {labels, datasets: sets}, options: {scales: {y: {beginAtZero: true, grid: {color: "#eef1f7"}}, x: {grid: {display: false}, stacked: !!extra.stack}, ...(extra.stack ? {y: {stacked: true, grid: {color: "#eef1f7"}}} : {})}, ...(extra.o || {})}});
const ds = (label, data, color) => ({label, data, backgroundColor: color, borderRadius: 4});

/* ---------- routing ---------- */
const INIT = {dashboard: initDash, performance: initPerf, insights: initIns, model: initModel, risk: renderRisk, simulator: initSim};
function route() {
  const p = (location.hash.replace("#/", "") || "dashboard"); const page = $("#p-" + p) ? p : "dashboard";
  $$(".page").forEach(e => e.classList.toggle("on", e.id === "p-" + page));
  $$(".side a").forEach(a => a.classList.toggle("on", a.dataset.p === page));
  $("#side").classList.remove("open"); window.scrollTo(0, 0);
  if (S.data && INIT[page]) { if (page === "risk" || page === "simulator") INIT[page](); else if (!S.inited[page]) { S.inited[page] = 1; INIT[page](); } }
}
window.addEventListener("hashchange", route); $("#menu").onclick = () => $("#side").classList.toggle("open");
fetch("/api/bootstrap").then(r => r.json()).then(d => { S.data = d; route(); });

/* ---------- dashboard ---------- */
function initDash() {
  const {eda, results: R} = S.data, d = eda.dash;
  const k = [["Total Applications", d.total.toLocaleString()], ["Approved", d.approved.toLocaleString()], ["Rejected", d.rejected.toLocaleString()],
    ["Approval Rate", d.approval_rate + "%"], ["Average CIBIL", d.avg_cibil], ["Avg Loan Amount", money(d.avg_loan)]];
  $("#kpis").innerHTML = k.map(([l, v]) => `<div class="card h kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
  mk("c-appr", {type: "doughnut", data: {labels: ["Approved", "Rejected"], datasets: [{data: [d.approved, d.rejected], backgroundColor: [COL.ok, COL.bad], borderWidth: 0}]}, options: {cutout: "66%", plugins: {legend: {position: "bottom"}}}});
  bar("c-cibil", eda.cibil.labels, [ds("Approved", eda.cibil.approved, COL.ok), ds("Rejected", eda.cibil.rejected, COL.bad)], {stack: 1});
  bar("c-inc", eda.income.labels, [ds("Applicants", eda.income.all, COL.pri)]);
  bar("c-loan", eda.loan.labels, [ds("Applicants", eda.loan.all, COL.pur)]);
  bar("c-prob", R.prob_bins, [ds("Applicants", R.prob_hist_all, COL.pri)]);
  mk("c-risk", {type: "doughnut", data: {labels: ["Low", "Medium", "High"], datasets: [{data: [R.risk_counts.LOW, R.risk_counts.MEDIUM, R.risk_counts.HIGH], backgroundColor: [COL.ok, COL.warn, COL.bad], borderWidth: 0}]}, options: {cutout: "66%", plugins: {legend: {position: "bottom"}}}});
}

/* ---------- prediction ---------- */
const IDS = ["dependents", "education", "self_employed", "income", "loan_amount", "loan_term", "cibil", "residential", "commercial", "luxury", "bank"];
const SAMPLES = {strong: [1, "Graduate", "No", 8500000, 22000000, 8, 790, 12000000, 6000000, 20000000, 7000000], weak: [3, "Not Graduate", "Yes", 3500000, 11000000, 16, 480, 1500000, 800000, 4000000, 1200000]};
const getForm = () => Object.fromEntries(IDS.map(i => [i, $("#" + i).value]));
$$("[data-sample]").forEach(b => b.onclick = () => IDS.forEach((i, n) => $("#" + i).value = SAMPLES[b.dataset.sample][n]));
const post = (u, b) => fetch(u, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(b)}).then(async r => ({ok: r.ok, j: await r.json()}));

function ring(v, label, invert) {
  const c = 2 * Math.PI * 40, col = (invert ? 100 - v : v) >= 66 ? COL.ok : (invert ? 100 - v : v) >= 40 ? COL.warn : COL.bad;
  return `<div><div class="ring"><svg width="96" height="96"><circle cx="48" cy="48" r="40" fill="none" stroke="#e8edf6" stroke-width="9"/><circle cx="48" cy="48" r="40" fill="none" stroke="${col}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${c * v / 100} ${c}"/></svg><b>${v}</b></div><small>${label}</small></div>`;
}
const summ = r => `<div class="sumg">${[["CIBIL Score", r.summary.cibil], ["Annual Income", money(r.summary.income)], ["Loan Amount", money(r.summary.loan_amount)], ["Loan-to-Income", r.summary.loan_to_income + "x"], ["Total Assets", money(r.summary.total_assets)], ["Loan Term", r.summary.loan_term + " yrs"]].map(([l, v]) => `<div><small>${l}</small><b>${v}</b></div>`).join("")}</div>`;
const warnBox = r => r.warnings && r.warnings.length ? `<div class="warn"><b>Outside training range:</b> ${r.warnings.join("; ")}. The model is extrapolating, so treat this prediction with caution.</div>` : "";

$("#analyze").onclick = async () => {
  const input = getForm(), box = $("#presult"), err = $("#formerr"); err.style.display = "none";
  const msgs = ["Analyzing applicant profile...", "Evaluating financial indicators...", "Calculating approval probability...", "Generating risk assessment..."];
  box.innerHTML = `<div class="card loader"><div class="spin"></div><p id="lm">${msgs[0]}</p><small>Progress display only · prediction runs locally on the custom model</small></div>`;
  const req = post("/api/predict", input);
  for (let i = 1; i < msgs.length; i++) { await new Promise(r => setTimeout(r, 450)); $("#lm").textContent = msgs[i]; }
  await new Promise(r => setTimeout(r, 350)); const {ok, j} = await req;
  if (!ok) { box.innerHTML = `<div class="card empty">Your analysis will appear here.</div>`; err.textContent = j.error; err.style.display = "block"; return; }
  S.last = j; S.input = input; renderResult(j);
};
function renderResult(r) {
  const f = r.factors, ok = r.decision === "APPROVED";
  $("#presult").innerHTML = `<div class="card result"><div class="t">AI LOAN PREDICTION</div><div class="d ${r.decision}">${ok ? "✓ LOAN APPROVED" : "✕ LOAN REJECTED"}</div>
   <div class="p">${r.probability.toFixed(2)}% Approval Probability</div>
   <div class="rrow"><div><small>Risk Score</small><b>${r.risk_score}/100</b></div><div><small>Risk Level</small><b><span class="chip ${r.risk_level}">${r.risk_level}</span></b></div><div><small>Confidence</small><b>${r.confidence}</b></div></div></div>
  ${warnBox(r)}<div class="card" style="margin-top:16px"><h3>Financial Summary</h3>${summ(r)}</div>
  <div class="card" style="margin-top:16px"><h3>Financial Health</h3><div class="rings">${ring(r.health.credit, "CIBIL Health")}${ring(r.health.income, "Income Strength")}${ring(r.health.burden, "Loan Burden", true)}${ring(r.health.assets, "Asset Strength")}</div></div>
  <div class="card" style="margin-top:16px"><h3>Why this loan was ${ok ? "approved" : "rejected"}</h3><ul class="fl">${f.positive.map(x => `<li class="p">${x}</li>`).join("")}${f.negative.map(x => `<li class="n">${x}</li>`).join("")}${!f.positive.length && !f.negative.length ? "<li>No individual indicator is strongly positive or negative.</li>" : ""}</ul>
   <p class="hint">Rules compare your values with percentiles of the training data. Model drivers below show each feature's actual contribution (weight × standardised value) to the log-odds.</p>
   <div class="grid g2" style="margin-top:10px"><div><h3>Pushing toward approval</h3>${f.drivers_up.map(d => `<div class="brow"><div><span>${d.feature}</span><b>+${d.impact}</b></div></div>`).join("") || "<small>None</small>"}</div><div><h3>Pushing toward rejection</h3>${f.drivers_down.map(d => `<div class="brow"><div><span>${d.feature}</span><b>${d.impact}</b></div></div>`).join("") || "<small>None</small>"}</div></div></div>
  <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap"><a href="#/risk"><button class="btn">View Risk Analysis</button></a><a href="#/simulator"><button class="btn o">Try What-If Simulator</button></a></div>`;
}

/* ---------- risk page ---------- */
function renderRisk() {
  const r = S.last, el = $("#riskbody");
  if (!r) { el.innerHTML = `<div class="card empty">No applicant analysed yet.<br><a href="#/predict"><button class="btn">Analyze a Loan</button></a></div>`; return; }
  const col = LVL[r.risk_level], p = r.risk_formula.parts, w = r.risk_formula.weights, f = r.factors;
  const names = {model: "Model rejection probability", credit: "Credit weakness (CIBIL)", burden: "Loan burden (loan ÷ income)", assets: "Asset weakness", income: "Income weakness"};
  el.innerHTML = `<div class="grid g2"><div class="card" style="text-align:center"><h3>Risk Score</h3>
   <svg viewBox="0 0 200 120" width="260"><path d="M10 100A90 90 0 0 1 190 100" pathLength="100" fill="none" stroke="#e8edf6" stroke-width="16" stroke-linecap="round"/><path d="M10 100A90 90 0 0 1 190 100" pathLength="100" fill="none" stroke="${col}" stroke-width="16" stroke-linecap="round" stroke-dasharray="${r.risk_score} 100"/><text x="100" y="92" text-anchor="middle" font-size="32" font-weight="800" fill="#0f172a">${r.risk_score}</text><text x="100" y="112" text-anchor="middle" font-size="11" fill="#64748b">out of 100</text></svg>
   <div><span class="chip ${r.risk_level}" style="font-size:14px">${r.risk_level} RISK</span></div><p class="hint" style="margin-top:8px">0–30 Low · 31–60 Medium · 61–100 High</p></div>
  <div class="card"><h3>Strength profile</h3><div class="cc s"><canvas id="c-strength"></canvas></div></div></div>
  ${warnBox(r)}
  <div class="grid g2" style="margin-top:16px"><div class="card"><h3>Positive risk factors</h3><ul class="fl">${f.positive.map(x => `<li class="p">${x}</li>`).join("") || "<li>None identified</li>"}</ul></div>
  <div class="card"><h3>Negative risk factors</h3><ul class="fl">${f.negative.map(x => `<li class="n">${x}</li>`).join("") || "<li>None identified</li>"}</ul></div></div>
  <div class="card" style="margin-top:16px"><h3>How this score was calculated</h3>
   ${Object.keys(p).map(k => `<div class="brow"><div><span>${names[k]} <small>(weight ${(w[k] * 100).toFixed(0)}%)</small></span><b>${p[k]} pts</b></div><div class="bar"><i style="width:${p[k] / (w[k] * 100) * 100}%;background:${col}"></i></div></div>`).join("")}
   <pre>Risk = 100 × [0.45(1−P) + 0.25(1−Credit) + 0.15·Burden + 0.10(1−Assets) + 0.05(1−Income)]</pre>
   <p class="hint">Project-generated transparent score. Burden, Assets and Income are scaled between the 5th and 95th percentile of the training data. It is NOT an official banking or CIBIL score.</p></div>`;
  mk("c-strength", {type: "bar", data: {labels: ["Credit", "Income", "Asset", "Loan affordability"], datasets: [{data: [r.health.credit, r.health.income, r.health.assets, 100 - r.health.burden], backgroundColor: [COL.pri, COL.pur, COL.ok, COL.warn], borderRadius: 6}]}, options: {indexAxis: "y", plugins: {legend: {display: false}}, scales: {x: {max: 100, grid: {color: "#eef1f7"}}, y: {grid: {display: false}}}}});
}

/* ---------- simulator ---------- */
const SIMF = [["cibil", "CIBIL Score", 1], ["income", "Annual Income (₹)", 100000], ["loan_amount", "Loan Amount (₹)", 100000], ["loan_term", "Loan Term (years)", 1], ["residential", "Residential Assets (₹)", 100000], ["commercial", "Commercial Assets (₹)", 100000], ["luxury", "Luxury Assets (₹)", 100000], ["bank", "Bank Assets (₹)", 100000]];
function initSim() {
  const cur = S.input || getForm(), key = JSON.stringify(cur); if (key === S.simKey) return; S.simKey = key; S.simCur = cur;
  $("#simtable tbody").innerHTML = SIMF.map(([k, l, st]) => `<tr><td>${l}</td><td>${Number(cur[k]).toLocaleString("en-IN")}</td><td><input id="s-${k}" type="number" step="${st}" value="${cur[k]}" style="max-width:200px;text-align:right"></td></tr>`).join("");
  $("#simout").innerHTML = "";
}
$("#simbtn").onclick = async () => {
  const cur = S.simCur, imp = {...cur}; SIMF.forEach(([k]) => imp[k] = $("#s-" + k).value); $("#simerr").style.display = "none";
  const {ok, j} = await post("/api/simulate", {current: cur, improved: imp});
  if (!ok) { $("#simerr").textContent = j.error; $("#simerr").style.display = "block"; return; }
  const c = j.current, i = j.improved, up = j.change >= 0;
  $("#simout").innerHTML = `<div class="grid g3"><div class="card"><small class="hint">CURRENT</small><div class="big">${c.probability.toFixed(1)}%</div><div>Approval probability · Risk <span class="chip ${c.risk_level}">${c.risk_level}</span> (${c.risk_score})</div></div>
   <div class="card"><small class="hint">AFTER IMPROVEMENT</small><div class="big">${i.probability.toFixed(1)}%</div><div>Approval probability · Risk <span class="chip ${i.risk_level}">${i.risk_level}</span> (${i.risk_score})</div></div>
   <div class="card"><small class="hint">CHANGE</small><div class="delta" style="color:${up ? COL.ok : COL.bad}">${up ? "+" : ""}${j.change.toFixed(1)}%</div><div>${c.decision} → ${i.decision}</div></div></div>
   ${warnBox(i)}<div class="grid g21" style="margin-top:16px"><div class="card"><h3>Comparison</h3><div class="cc"><canvas id="c-sim"></canvas></div></div>
   <div class="card"><h3>What changed</h3>${j.steps.map(s => `<p style="margin-bottom:10px">${s.text}</p>`).join("") || "<p class='hint'>Change at least one value in the improved column.</p>"}<p class="hint">Each effect is measured by changing only that field and re-running the trained model.</p></div></div>`;
  bar("c-sim", ["Approval probability (%)", "Risk score (/100)"], [ds("Current", [c.probability, c.risk_score], COL.mut), ds("Improved", [i.probability, i.risk_score], COL.pri)]);
};

/* ---------- performance ---------- */
function initPerf() {
  const R = S.data.results, t = R.metrics.test, cm = t.confusion, H = R.history, st = n => H[n].filter((_, k) => k % 25 === 0), lab = H.train_loss.filter((_, k) => k % 25 === 0).map((_, k) => k * 25);
  const vc = {OVERFITTING: COL.bad, UNDERFITTING: COL.warn, "WELL FITTED": COL.ok}[R.verdict];
  $("#perf").innerHTML = `<div class="grid g6">${[["Accuracy", pct(t.accuracy)], ["Precision", pct(t.precision)], ["Recall", pct(t.recall)], ["F1 Score", pct(t.f1)], ["ROC-AUC", t.roc_auc.toFixed(4)], ["Test samples", R.split.test]].map(([l, v]) => `<div class="card kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("")}</div>
  <div class="grid g2" style="margin-top:16px"><div class="card"><h3>Confusion matrix (test)</h3><div class="cm"><div class="h"></div><div class="h">Pred. Rejected</div><div class="h">Pred. Approved</div>
   <div class="h">Actual Rejected</div><div style="background:#e8f6ee;color:#14532d">${cm.tn}<small>True Neg</small></div><div style="background:#fceaea;color:#7f1d1d">${cm.fp}<small>False Pos</small></div>
   <div class="h">Actual Approved</div><div style="background:#fceaea;color:#7f1d1d">${cm.fn}<small>False Neg</small></div><div style="background:#e8f6ee;color:#14532d">${cm.tp}<small>True Pos</small></div></div></div>
  <div class="card"><h3>ROC curve (AUC ${t.roc_auc.toFixed(3)})</h3><div class="cc"><canvas id="c-roc"></canvas></div></div>
  <div class="card"><h3>Training loss vs epoch (BCE)</h3><div class="cc"><canvas id="c-loss"></canvas></div></div>
  <div class="card"><h3>Actual vs predicted distribution (test)</h3><div class="cc"><canvas id="c-dist"></canvas></div></div></div>
  <div class="card" style="margin-top:16px"><h2>Overfitting / Underfitting analysis</h2><p class="sub" style="margin-bottom:12px">Verdict: <span class="chip" style="background:${vc}22;color:${vc}">${R.verdict}</span> · train–test accuracy gap ${(R.acc_gap * 100).toFixed(2)} points</p>
   <table><thead><tr><th>Set</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>${["train", "val", "test"].map(k => { const m = R.metrics[k]; return `<tr><td>${{train: "Training", val: "Validation", test: "Test"}[k]}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${m.roc_auc.toFixed(4)}</td></tr>` }).join("")}</tbody></table>
   <div class="grid g2" style="margin-top:16px"><div><h3>L2 regularisation study (λ chosen on validation set)</h3><div class="cc s"><canvas id="c-reg"></canvas></div></div>
   <div><h3>Reading the results</h3><p style="margin-bottom:8px">Training and test accuracy are almost identical, so the model is <b>not overfitting</b>. Training accuracy is high (${pct(R.metrics.train.accuracy)}), so it is <b>not underfitting</b>.</p>
   <p style="margin-bottom:8px">Selected λ = <b>${R.config.l2}</b>. L2 adds λ‖W‖² to the loss, shrinking weights to reduce variance. With only 11 features and thousands of rows, regularisation changes little here; at λ = 1 it is too strong and the model underfits (validation accuracy ${pct(R.study[R.study.length - 1].val_acc)}).</p>
   <p>Underfitting demo (20 epochs, learning rate 0.001): loss stays at ${R.underfit.train_loss.toFixed(3)} (≈ chance, ln 2 = 0.693) even though accuracy is ${pct(R.underfit.train_acc)}. Accuracy alone can hide an under-trained model.</p></div></div></div>`;
  const pts = R.roc.fpr.map((x, k) => ({x, y: R.roc.tpr[k]}));
  mk("c-roc", {type: "scatter", data: {datasets: [{label: "Model", data: pts, borderColor: COL.pri, showLine: true, pointRadius: 0, fill: true, backgroundColor: "#1d4ed814"}, {label: "Random", data: [{x: 0, y: 0}, {x: 1, y: 1}], borderColor: COL.mut, borderDash: [5, 5], showLine: true, pointRadius: 0}]}, options: {scales: {x: {min: 0, max: 1, title: {display: true, text: "False positive rate"}}, y: {min: 0, max: 1, title: {display: true, text: "True positive rate"}, grid: {color: "#eef1f7"}}}}});
  mk("c-loss", {type: "line", data: {labels: lab, datasets: [{label: "Train", data: st("train_loss"), borderColor: COL.pri, pointRadius: 0}, {label: "Validation", data: st("val_loss"), borderColor: COL.warn, pointRadius: 0}]}, options: {scales: {x: {title: {display: true, text: "Epoch"}, ticks: {maxTicksLimit: 8}}, y: {grid: {color: "#eef1f7"}}}}});
  bar("c-dist", R.prob_bins, [ds("Actually approved", R.prob_hist_test_approved, COL.ok), ds("Actually rejected", R.prob_hist_test_rejected, COL.bad)]);
  mk("c-reg", {type: "line", data: {labels: R.study.map(s => "λ=" + s.l2), datasets: [{label: "Train acc", data: R.study.map(s => s.train_acc * 100), borderColor: COL.pri}, {label: "Validation acc", data: R.study.map(s => s.val_acc * 100), borderColor: COL.warn}]}, options: {scales: {y: {title: {display: true, text: "Accuracy %"}, grid: {color: "#eef1f7"}}}}});
}

/* ---------- dataset insights ---------- */
function initIns() {
  const E = S.data.eda, a = E.audit;
  const hm = E.corr, n = hm.labels.length, cell = v => { const x = Math.abs(v); return v >= 0 ? `rgba(29,78,216,${x})` : `rgba(185,28,28,${x})`; };
  $("#ins").innerHTML = `<div class="grid g6">${[["Total records", a.rows.toLocaleString()], ["Features", a.columns - 2], ["Approved", a.approved.toLocaleString()], ["Rejected", a.rejected.toLocaleString()], ["Approval %", a.approval_pct + "%"], ["Missing / duplicates", a.missing + " / " + a.duplicates]].map(([l, v]) => `<div class="card kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("")}</div>
  <div class="grid g3" style="margin-top:16px">
  <div class="card"><h3>1. Approval distribution</h3><div class="cc s"><canvas id="i1"></canvas></div></div>
  <div class="card"><h3>2. CIBIL distribution</h3><div class="cc s"><canvas id="i2"></canvas></div></div>
  <div class="card"><h3>3. Income vs loan amount (₹ M, 500-row sample)</h3><div class="cc s"><canvas id="i3"></canvas></div></div>
  <div class="card"><h3>4. CIBIL vs approval rate (%)</h3><div class="cc s"><canvas id="i4"></canvas></div></div>
  <div class="card"><h3>5. Education vs approval rate (%)</h3><div class="cc s"><canvas id="i5"></canvas></div></div>
  <div class="card"><h3>6. Self-employment vs approval rate (%)</h3><div class="cc s"><canvas id="i6"></canvas></div></div>
  <div class="card"><h3>7. Average asset value by outcome (₹ M)</h3><div class="cc s"><canvas id="i7"></canvas></div></div></div>
  <div class="card" style="margin-top:16px"><h3>8. Correlation heatmap</h3><div class="heat" style="grid-template-columns:130px repeat(${n},1fr)"><div class="lb"></div>${hm.labels.map(l => `<div class="lb" style="text-align:center;font-size:9.5px">${l.replace(/_/g, " ").slice(0, 9)}</div>`).join("")}
   ${hm.matrix.map((row, i) => `<div class="lb">${hm.labels[i].replace(/_/g, " ")}</div>` + row.map(v => `<div style="background:${cell(v)};color:${Math.abs(v) > .5 ? "#fff" : "#334155"}">${v.toFixed(2)}</div>`).join("")).join("")}</div></div>
  <div class="note"><b>Key findings (computed from the data):</b> CIBIL score has a correlation of ${hm.matrix[6][n - 1]} with approval, far above every other feature. Approval is ~10% for scores up to 550 and ~99% above it (chart 4). Education and employment show almost no relationship with approval. Data quality: ${a.negative_residential} negative residential-asset values were clipped to 0; IQR outliers exist only in asset columns (${Object.entries(a.outliers).map(([k, v]) => k.replace(/_/g, " ") + ": " + v).join(", ")}).</div>`;
  mk("i1", {type: "doughnut", data: {labels: ["Approved", "Rejected"], datasets: [{data: [a.approved, a.rejected], backgroundColor: [COL.ok, COL.bad], borderWidth: 0}]}, options: {cutout: "62%", plugins: {legend: {position: "bottom"}}}});
  bar("i2", E.cibil.labels, [ds("Approved", E.cibil.approved, COL.ok), ds("Rejected", E.cibil.rejected, COL.bad)], {stack: 1});
  mk("i3", {type: "scatter", data: {datasets: [{label: "Approved", data: E.scatter.filter(p => p.s).map(p => ({x: p.x, y: p.y})), backgroundColor: COL.ok + "99", pointRadius: 2.5}, {label: "Rejected", data: E.scatter.filter(p => !p.s).map(p => ({x: p.x, y: p.y})), backgroundColor: COL.bad + "99", pointRadius: 2.5}]}, options: {scales: {x: {title: {display: true, text: "Income"}}, y: {title: {display: true, text: "Loan"}, grid: {color: "#eef1f7"}}}}});
  bar("i4", E.cibil_rate.labels.map(l => l.replace(/[()\]]/g, "").replace(", ", "–")), [ds("Approval %", E.cibil_rate.rate, COL.pri)], {o: {plugins: {legend: {display: false}}}});
  bar("i5", Object.keys(E.education), [ds("Approval %", Object.values(E.education), COL.pri)], {o: {plugins: {legend: {display: false}}}});
  bar("i6", Object.keys(E.self_employed), [ds("Approval %", Object.values(E.self_employed), COL.pri)], {o: {plugins: {legend: {display: false}}}});
  const ak = Object.keys(E.asset_means); bar("i7", ak, [ds("Approved", ak.map(k => E.asset_means[k][0] / 1e6), COL.ok), ds("Rejected", ak.map(k => E.asset_means[k][1] / 1e6), COL.bad)]);
}

/* ---------- about model ---------- */
function initModel() {
  const R = S.data.results, c = R.coefficients, mx = Math.max(...c.map(d => Math.abs(d.weight))), a = R.ablation;
  $("#mod").innerHTML = `<div class="card"><div class="flow">${["Applicant Data", "Preprocessing", "Feature Scaling", "Logistic Regression", "Probability", "Decision", "Risk Assessment"].map(x => `<div>${x}</div>`).join("<span>→</span>")}</div></div>
  <div class="grid g2" style="margin-top:16px"><div class="card"><h3>Learned model coefficients (weights)</h3><div class="cc t"><canvas id="m-w"></canvas></div>
   <div class="note">These are <b>coefficients on standardised features</b>, not a generic “feature importance”. A positive weight raises the log-odds of approval per 1 standard deviation increase; negative lowers it. Magnitude is comparable across features only because they were scaled. Coefficients of correlated features (income and loan amount here, which move together) partly offset each other and should be read together.</div></div>
  <div class="card"><h3>In simple terms</h3><p style="margin-bottom:8px"><b>1.</b> Each input is standardised: <code>z = (x − mean) / std</code>.</p><p style="margin-bottom:8px"><b>2.</b> The model forms a weighted sum <code>z·W + b</code>.</p><p style="margin-bottom:8px"><b>3.</b> The sigmoid <code>1/(1+e^−s)</code> turns it into a probability between 0 and 1.</p><p style="margin-bottom:8px"><b>4.</b> Probability ≥ ${R.config.threshold} → Approved, otherwise Rejected.</p><p style="margin-bottom:8px"><b>5.</b> Training adjusts W and b by gradient descent to minimise binary cross-entropy: <code>W ← W − α·∂L/∂W</code>.</p><p>The risk score then blends this probability with CIBIL, loan burden, assets and income using a fixed, documented formula.</p>
   <table style="margin-top:14px"><tbody><tr><td>Learning rate α</td><td>${R.config.lr}</td></tr><tr><td>Epochs</td><td>${R.config.epochs}</td></tr><tr><td>L2 λ</td><td>${R.config.l2}</td></tr><tr><td>Threshold</td><td>${R.config.threshold}</td></tr><tr><td>Features</td><td>${R.config.n_features}</td></tr><tr><td>Train / val / test rows</td><td>${R.split.train} / ${R.split.val} / ${R.split.test}</td></tr></tbody></table></div></div>
  <div class="card" style="margin-top:16px"><h3>Feature engineering experiment</h3><p>We tested adding two ratio features (loan-to-income, assets-to-loan). Test accuracy: <b>${pct(a.raw_features.test_acc)}</b> with ${a.raw_features.n} raw features vs <b>${pct(a.with_ratios.test_acc)}</b> with ${a.with_ratios.n}; ROC-AUC ${a.raw_features.test_auc.toFixed(4)} vs ${a.with_ratios.test_auc.toFixed(4)}. No benefit, so the simpler model is used and ratios are used for explanations and risk scoring.</p></div>`;
  mk("m-w", {type: "bar", data: {labels: c.map(d => d.feature), datasets: [{data: c.map(d => d.weight), backgroundColor: c.map(d => d.weight >= 0 ? COL.ok : COL.bad), borderRadius: 4}]}, options: {indexAxis: "y", plugins: {legend: {display: false}}, scales: {x: {grid: {color: "#eef1f7"}}, y: {grid: {display: false}}}}});
}
