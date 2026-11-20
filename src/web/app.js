// GymLeader page: no framework, no build step, no outside requests. Model text is only ever set with textContent.
"use strict";

const $ = (sel) => document.querySelector(sel);
const form = $("#ask"), input = $("#request"), go = $("#go"), out = $("#out"), statusLine = $("#status");
let busy = false, healthTimer = null;

// Display only: 4 upstream catalog names carry a mis-encoded degree sign. /workout returns them unchanged.
const fixDegree = (s) => String(s).replaceAll("в°", "°");

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c instanceof Node ? c : String(c));
  return node;
}

// ---- status line: poll /health until the model is ready ----
async function checkHealth() {
  clearTimeout(healthTimer);
  try {
    const r = await fetch("/health");
    const body = await r.json();
    if (r.ok) {
      statusLine.className = "status ok";
      statusLine.textContent = `● Ready · ${body.model_file} · llama.cpp ${body.server_build ?? "?"}`;
      return;
    }
  } catch (e) { /* server not up yet */ }
  statusLine.className = "status wait";
  statusLine.textContent = "● The model is still starting… checking again in a few seconds";
  healthTimer = setTimeout(checkHealth, 3000);
}

// ---- chips write the canonical sentence ----
function chipSentence() {
  const pick = (name) => form.querySelector(`input[name="${name}"]:checked`)?.value;
  const body = pick("body"), equip = pick("equip"), level = pick("level"), n = pick("n");
  if (!body || !equip || !level) {
    $("#chip-hint").textContent = "Pick a body part, equipment and difficulty to fill in the request.";
    return;
  }
  const text = form.dataset.template.replace("{body}", body).replace("{level}", level)
    .replace("{equip}", equip).replace("{n}", n);
  input.value = text[0].toUpperCase() + text.slice(1);
  $("#chip-hint").textContent = "Request filled in from the chips: press Generate or Enter.";
}

// ---- results ----
function sourceText(r) {
  if (r.source === "guard") return "answered from the catalog";
  // unverified: the best of all the calls, even when that is the greedy one
  if (!r.verified && r.model_calls > 1) return `model, best of ${r.model_calls} answers`;
  if (r.source === "model_greedy") return "model, first answer";
  return `model, rescued at retry ${r.source.replace("model_retry_", "")}`;
}

const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;

function badge(text, cls = "") { return el("span", { class: `badge ${cls}` }, text); }

function table(head, rows) {
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { scope: "col" }, h)))));
  const tb = el("tbody");
  for (const row of rows) tb.append(el("tr", {}, ...row.map((c) => el("td", {}, c))));
  t.append(tb);
  return t;
}

function showResult(r) {
  out.replaceChildren();
  out.append(el("p", { class: "badges" },
    badge(r.verified ? "✓ verified" : "✗ not verified", r.verified ? "good" : "bad"),
    badge(sourceText(r)),
    badge(`${r.model_calls} model call${r.model_calls === 1 ? "" : "s"}`),
    badge(secs(r.timings_ms.total))));

  if (r.interpreted_as.length) {
    out.append(el("ul", { class: "notes" }, ...r.interpreted_as.map((n) => el("li", {}, `ℹ ${n}`))));
  }

  if (Array.isArray(r.workout)) {
    const list = el("ol", { class: "cards" });
    for (const ex of r.workout) {
      const name = ex && typeof ex === "object" ? ex.name : JSON.stringify(ex);
      const dose = ex && typeof ex === "object" ? `${ex.sets} × ${ex.reps}` : "";
      list.append(el("li", { class: "card" }, el("span", { class: "name" }, fixDegree(name)),
        el("span", { class: "dose" }, dose)));
    }
    out.append(list);
  } else if (r.error) {
    out.append(el("div", { class: "box error-box" }, el("p", {}, r.error),
      el("p", { class: "hint" }, r.source === "guard"
        ? "Answered from the catalog: the model wasn't called."
        : "The model answered that this request can't be done.")));
  } else {
    out.append(el("div", { class: "box error-box" }, "The model's answer could not be read. See the raw output below."));
  }

  const details = el("details", {}, el("summary", {}, "How this was checked"));
  if (r.model_input && r.model_input !== r.request_text) {
    details.append(el("p", {}, "Sent to the model: ", el("code", {}, r.model_input)));
  }
  const checks = Object.entries(r.checks).filter(([, v]) => typeof v !== "object" || v === null);
  details.append(el("h3", {}, "Checks"), table(["check", "result"], checks.map(([k, v]) =>
    [k, v === true ? "✓ pass" : v === false ? "✗ fail" : String(v)])));
  if (r.attempts.length) {
    details.append(el("h3", {}, "Attempts"), table(["#", "temperature", "seed", "reward", "first failed check", "ms"],
      r.attempts.map((a, i) => [i + 1, a.temperature, a.seed ?? "none", a.reward.toFixed(3), a.first_failed, a.ms])));
  }
  details.append(el("h3", {}, "Raw output"), el("pre", {}, r.raw_output));
  out.append(details);
}

function showUnreadable(detail) {
  out.replaceChildren();
  const box = el("div", { class: "box error-box" }, el("p", {}, detail.message ?? String(detail)));
  if (detail.allowed) {
    const dl = el("dl", { class: "allowed" });
    for (const [part, values] of Object.entries(detail.allowed)) {
      dl.append(el("dt", {}, part.replaceAll("_", " ")));
      dl.append(el("dd", {}, typeof values === "string" ? values
        : Object.entries(values).map(([v, words]) => `${v}: ${words.join(", ")}`).join(" · ")));
    }
    box.append(el("p", {}, "Words the request can use:"), dl);
  }
  box.append(el("p", { class: "hint" }, "Tip: use the chips above, they always write a valid request."));
  out.append(box);
}

function showProblem(text) {
  out.replaceChildren(el("div", { class: "box error-box" }, el("p", {}, text)));
}

// ---- submit ----
async function submit(event) {
  event.preventDefault();
  const text = input.value.trim();
  if (busy) return;
  if (!text) { input.focus(); return; }
  busy = true;
  go.disabled = true;
  const started = Date.now();
  const tick = () => {
    out.replaceChildren(el("p", { class: "loading" },
      `⏳ Working… ${Math.round((Date.now() - started) / 1000)} s `,
      el("span", { class: "hint" }, "(about 5–20 s on CPU, longer when the answer needs retries)")));
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const r = await fetch("/workout", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: text }) });
    const body = await r.json().catch(() => ({}));
    clearInterval(timer);
    if (r.ok) showResult({ ...body, request_text: text });
    else if (r.status === 422) showUnreadable(typeof body.detail === "object" && !Array.isArray(body.detail)
      ? body.detail : { message: "The request must be 1–500 characters." });
    else if (r.status === 503) { showProblem("The model is still starting or unavailable. Try again in a few seconds."); checkHealth(); }
    else showProblem(`Something went wrong (HTTP ${r.status}).`);
  } catch (e) {
    clearInterval(timer);
    showProblem("Can't reach the server. Is the container still running?");
    checkHealth();
  } finally {
    busy = false;
    go.disabled = false;
  }
}

form.addEventListener("submit", submit);
form.addEventListener("change", (e) => { if (e.target.type === "radio") chipSentence(); });
for (const b of document.querySelectorAll(".example")) {
  b.addEventListener("click", () => { input.value = b.textContent; form.requestSubmit(); });
}
checkHealth();
