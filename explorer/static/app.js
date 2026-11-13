/* YarnBall Local Workbench Frontend (ADR-00018 — Zero-CDN XSS-Safe DOM) */
(function () {
  "use strict";

  const state = {
    activeTab: "audit",
    inputMode: "fixture",
    showRawDiff: false,
    catalog: null,
    lastEvaluation: null,
    rawInputText: "",
  };

  const els = {
    modeSelect: document.getElementById("modeSelect"),
    themeSelect: document.getElementById("themeSelect"),
    densitySelect: document.getElementById("densitySelect"),
    ollamaDot: document.getElementById("ollamaDot"),
    ollamaText: document.getElementById("ollamaText"),
    srcFixtureBtn: document.getElementById("srcFixtureBtn"),
    srcInlineBtn: document.getElementById("srcInlineBtn"),
    fixtureControls: document.getElementById("fixtureControls"),
    inlineControls: document.getElementById("inlineControls"),
    fixtureSelect: document.getElementById("fixtureSelect"),
    fixtureMetaCard: document.getElementById("fixtureMetaCard"),
    inTicker: document.getElementById("inTicker"),
    inDate: document.getElementById("inDate"),
    inCompany: document.getElementById("inCompany"),
    inItems: document.getElementById("inItems"),
    inHeadline: document.getElementById("inHeadline"),
    inContent: document.getElementById("inContent"),
    inMetadata: document.getElementById("inMetadata"),
    topologySelect: document.getElementById("topologySelect"),
    targetTickerInput: document.getElementById("targetTickerInput"),
    evaluateBtn: document.getElementById("evaluateBtn"),
    evalErrorBox: document.getElementById("evalErrorBox"),
    toggleDiffBtn: document.getElementById("toggleDiffBtn"),
    gatingBanner: document.getElementById("gatingBanner"),
    gatingKpis: document.getElementById("gatingKpis"),
    strippedCountPill: document.getElementById("strippedCountPill"),
    sanitizedHeadlineOut: document.getElementById("sanitizedHeadlineOut"),
    sanitizedContentOut: document.getElementById("sanitizedContentOut"),
    rawPanel: document.getElementById("rawPanel"),
    rawContentOut: document.getElementById("rawContentOut"),
    cascadeSubtitle: document.getElementById("cascadeSubtitle"),
    dagHealthBadges: document.getElementById("dagHealthBadges"),
    ganttSvgMount: document.getElementById("ganttSvgMount"),
    impactsTbody: document.getElementById("impactsTbody"),
    citationPill: document.getElementById("citationPill"),
    citationViolationsBox: document.getElementById("citationViolationsBox"),
    reportIdTag: document.getElementById("reportIdTag"),
    synthesisOut: document.getElementById("synthesisOut"),
    thinkTraceOut: document.getElementById("thinkTraceOut"),
    telemetryPills: document.getElementById("telemetryPills"),
    oneNfSummaryTitle: document.getElementById("oneNfSummaryTitle"),
    oneNfTbody: document.getElementById("oneNfTbody"),
    copyJsonlBtn: document.getElementById("copyJsonlBtn"),
    downloadJsonlBtn: document.getElementById("downloadJsonlBtn"),
    jsonlStatusText: document.getElementById("jsonlStatusText"),
  };

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function makeEl(tag, className, textValue) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (textValue !== undefined && textValue !== null) {
      node.textContent = String(textValue);
    }
    return node;
  }

  function renderTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === state.activeTab);
    });
    document.getElementById("panel-audit").classList.toggle("hidden", state.activeTab !== "audit");
    document.getElementById("panel-kg").classList.toggle("hidden", state.activeTab !== "kg");
    document.getElementById("panel-cohort").classList.toggle("hidden", state.activeTab !== "cohort");
  }

  function renderInputMode() {
    const isFix = state.inputMode === "fixture";
    els.srcFixtureBtn.classList.toggle("active", isFix);
    els.srcInlineBtn.classList.toggle("active", !isFix);
    els.fixtureControls.classList.toggle("hidden", !isFix);
    els.inlineControls.classList.toggle("hidden", isFix);
  }

  function renderFixtureMeta() {
    clearNode(els.fixtureMetaCard);
    const filings = (state.catalog && state.catalog.filings) || [];
    const selected = filings.find((f) => f.path === els.fixtureSelect.value);
    if (!selected) {
      els.fixtureMetaCard.appendChild(
        makeEl("div", "muted-text", "No repository fixtures discovered; switch to Inline mode.")
      );
      return;
    }
    const rows = [
      ["Company", selected.company_name || "Unknown"],
      ["Ticker", selected.ticker || "—"],
      ["Filing Date", selected.filing_date || "—"],
      ["Items", (selected.items || []).join(", ") || "—"],
      ["Headline", selected.headline || "—"],
    ];
    rows.forEach(([k, v]) => {
      const row = makeEl("div", "kv-row");
      row.appendChild(makeEl("span", "kv-key", k));
      row.appendChild(makeEl("span", "kv-val mono", v));
      els.fixtureMetaCard.appendChild(row);
    });
  }

  function renderOllamaStatus(ollama) {
    const healthy = Boolean(ollama && ollama.healthy);
    els.ollamaDot.style.backgroundColor = healthy ? "var(--success)" : "var(--warning)";
    els.ollamaText.textContent = healthy
      ? "Ollama Healthy (" + (ollama.detail || "ready") + ")"
      : "Ollama Offline (Mock Safe)";
  }

  function populateCatalog(catalog) {
    state.catalog = catalog;
    renderOllamaStatus(catalog.ollama);

    clearNode(els.fixtureSelect);
    (catalog.filings || []).forEach((f) => {
      const opt = makeEl("option", "", (f.ticker || "?") + " — " + f.name);
      opt.value = f.path;
      els.fixtureSelect.appendChild(opt);
    });
    renderFixtureMeta();

    clearNode(els.topologySelect);
    const defaultOpt = makeEl("option", "", "Built-in Canonical Supply Net (CLI parity)");
    defaultOpt.value = "canonical";
    els.topologySelect.appendChild(defaultOpt);

    (catalog.snapshots || []).forEach((s) => {
      const opt = makeEl(
        "option",
        "",
        "Snapshot: " + s.snapshot_id + " (" + s.edge_count + " edges)"
      );
      opt.value = "snapshot:" + s.snapshot_id;
      els.topologySelect.appendChild(opt);
    });
    (catalog.edge_manifests || []).forEach((m) => {
      const opt = makeEl(
        "option",
        "",
        "Manifest: " + m.name + " (" + m.edge_count + " edges)"
      );
      opt.value = "edges:" + m.path;
      els.topologySelect.appendChild(opt);
    });
  }

  function renderGanttSvg(report) {
    clearNode(els.ganttSvgMount);
    const impacts = report.network_impacts || [];
    if (!impacts.length) {
      els.ganttSvgMount.appendChild(
        makeEl("p", "muted-text", "No downstream network nodes returned for this disclosure.")
      );
      return;
    }

    const compact = els.densitySelect.value === "compact";
    const rowH = compact ? 26 : 32;
    const leftW = 104;
    const rightW = 500;
    const labelRightW = 175;
    const padTop = 22;
    const totalW = leftW + rightW + labelRightW;
    const totalH = padTop + impacts.length * rowH + 12;

    const maxBuf = Math.max(0, ...impacts.map((i) => Number(i.inventory_buffer_days || 0)));
    const maxEnd = Math.max(
      Number(report.shock_duration_days || 0),
      maxBuf,
      ...impacts.map((i) => Number(i.shock_onset_day || 0) + Number(i.unbuffered_downtime_days || 0)),
      60
    );

    const svgNs = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("viewBox", "0 0 " + totalW + " " + totalH);
    svg.style.width = "100%";
    svg.style.height = "auto";
    svg.style.display = "block";

    const tickStep = maxEnd <= 90 ? 30 : Math.ceil(maxEnd / 4 / 10) * 10;
    for (let t = 0; t <= maxEnd; t += tickStep) {
      const x = leftW + (t / maxEnd) * rightW;
      const line = document.createElementNS(svgNs, "line");
      line.setAttribute("x1", String(x));
      line.setAttribute("x2", String(x));
      line.setAttribute("y1", String(padTop - 4));
      line.setAttribute("y2", String(totalH - 6));
      line.setAttribute("stroke", "var(--border-subtle)");
      if (t > 0) line.setAttribute("stroke-dasharray", "3 3");
      svg.appendChild(line);

      const label = document.createElementNS(svgNs, "text");
      label.setAttribute("x", String(x));
      label.setAttribute("y", "13");
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "11");
      label.setAttribute("fill", "var(--muted)");
      label.textContent = "D" + t;
      svg.appendChild(label);
    }

    impacts.forEach((imp, idx) => {
      const y = padTop + idx * rowH;
      const barY = y + 4;
      const barH = rowH - 11;

      const nodeLbl = document.createElementNS(svgNs, "text");
      nodeLbl.setAttribute("x", "2");
      nodeLbl.setAttribute("y", String(barY + barH / 2 + 4));
      nodeLbl.setAttribute("font-size", "12");
      nodeLbl.setAttribute("font-weight", "600");
      nodeLbl.setAttribute("fill", "var(--fg)");
      nodeLbl.textContent = imp.entity_ticker + " (H" + imp.hop_distance + ")";
      svg.appendChild(nodeLbl);

      const bufDays = Number(imp.inventory_buffer_days || 0);
      const shockDays = Number(report.shock_duration_days || 0);
      const graceDays = shockDays > 0 ? Math.min(shockDays, bufDays) : bufDays;
      const bufW = Math.max(4, (graceDays / maxEnd) * rightW);

      const bufRect = document.createElementNS(svgNs, "rect");
      bufRect.setAttribute("x", String(leftW));
      bufRect.setAttribute("y", String(barY));
      bufRect.setAttribute("width", String(bufW));
      bufRect.setAttribute("height", String(barH));
      bufRect.setAttribute("rx", "3");
      bufRect.setAttribute("fill", "var(--buffer)");
      bufRect.setAttribute("fill-opacity", "0.3");
      bufRect.setAttribute("stroke", "var(--buffer)");
      svg.appendChild(bufRect);

      const downtime = Number(imp.unbuffered_downtime_days || 0);
      if (downtime > 0 && imp.shock_onset_day !== null && imp.shock_onset_day !== undefined) {
        const outX = leftW + (Number(imp.shock_onset_day) / maxEnd) * rightW;
        const outW = Math.max(5, (downtime / maxEnd) * rightW);
        const outRect = document.createElementNS(svgNs, "rect");
        outRect.setAttribute("x", String(outX));
        outRect.setAttribute("y", String(barY));
        outRect.setAttribute("width", String(outW));
        outRect.setAttribute("height", String(barH));
        outRect.setAttribute("rx", "3");
        outRect.setAttribute("fill", "var(--error)");
        svg.appendChild(outRect);
      }

      const note = document.createElementNS(svgNs, "text");
      note.setAttribute("x", String(leftW + rightW + 10));
      note.setAttribute("y", String(barY + barH / 2 + 4));
      note.setAttribute("font-size", "11");
      note.setAttribute(
        "fill",
        downtime > 0 ? "var(--error)" : imp.verdict === "BUFFER_ABSORBED" ? "var(--warning)" : "var(--success)"
      );
      note.textContent =
        downtime > 0
          ? downtime.toFixed(1) + "d outage (-" + Number(imp.capacity_contraction_pct).toFixed(1) + "%)"
          : imp.verdict + " (" + bufDays.toFixed(0) + "d buf)";
      svg.appendChild(note);
    });

    els.ganttSvgMount.appendChild(svg);
  }

  function renderEvaluation(payload) {
    state.lastEvaluation = payload;
    const assessment = payload.assessment;
    const report = payload.report;
    const violations = payload.citation_violations || [];
    const tsRows = payload.timeseries_rows || [];

    // Card 1: Gating Banner & KPIs
    clearNode(els.gatingBanner);
    const isDisruption = Boolean(assessment.is_material_operational_disruption);
    els.gatingBanner.style.borderColor = isDisruption ? "var(--warning)" : "var(--success)";
    els.gatingBanner.appendChild(
      makeEl(
        "strong",
        "",
        isDisruption
          ? "MATERIAL OPERATIONAL DISRUPTION CONFIRMED (" + assessment.disruption_category + ")"
          : "ZERO-SHOCK NEGATIVE CONTROL GUARDRAIL ACTIVE — Non-disruptive disclosure"
      )
    );
    els.gatingBanner.appendChild(makeEl("span", "mono muted-text", assessment.gating_reason));

    clearNode(els.gatingKpis);
    [
      ["SEC Item / Tier", assessment.primary_sec_item + " (Tier " + assessment.item_priority_tier + ")"],
      ["Initial Shock", Number(assessment.shock_magnitude_pct).toFixed(1) + "%"],
      ["Shock Duration", Number(assessment.shock_duration_days).toFixed(0) + " days"],
      ["Stripped Paragraphs", String(assessment.stripped_paragraph_count)],
    ].forEach(([lbl, val]) => {
      const box = makeEl("div", "kpi-card");
      box.appendChild(makeEl("div", "kpi-label", lbl));
      box.appendChild(makeEl("div", "kpi-val mono", val));
      els.gatingKpis.appendChild(box);
    });

    els.strippedCountPill.textContent =
      assessment.stripped_paragraph_count + " paragraph(s) stripped";
    els.sanitizedHeadlineOut.textContent = assessment.sanitized_headline || "(No headline)";
    els.sanitizedContentOut.textContent =
      assessment.sanitized_content || "(All boilerplate paragraphs stripped)";
    els.rawContentOut.textContent =
      state.rawInputText || assessment.sanitized_content || "(Raw text unavailable)";
    els.rawPanel.classList.toggle("hidden", !state.showRawDiff);
    els.toggleDiffBtn.textContent = state.showRawDiff ? "Hide Raw vs. Stripped" : "Show Raw vs. Stripped";

    // Card 2: Cascade Summary, Gantt & Table
    els.cascadeSubtitle.textContent =
      "Origin: " +
      report.disrupted_entity_ticker +
      " (" +
      report.disrupted_entity_name +
      ") · Category: " +
      report.disruption_category +
      " · Mode: " +
      report.model_mode;

    clearNode(els.dagHealthBadges);
    [
      "is_acyclic_dag=" + String(report.is_acyclic_dag),
      "spectral_radius=" + Number(report.spectral_radius || 0).toFixed(3),
    ].forEach((txt) => els.dagHealthBadges.appendChild(makeEl("span", "pill", txt)));

    renderGanttSvg(report);

    clearNode(els.impactsTbody);
    const sortedImpacts = [...(report.network_impacts || [])].sort(
      (a, b) => a.hop_distance - b.hop_distance || a.entity_ticker.localeCompare(b.entity_ticker)
    );
    sortedImpacts.forEach((imp) => {
      const tr = document.createElement("tr");
      const cells = [
        { val: imp.entity_ticker, cls: "" },
        { val: String(imp.hop_distance), cls: "" },
        { val: imp.input_group, cls: "" },
        { val: Number(imp.procurement_share_pct).toFixed(1) + "%", cls: "num" },
        { val: Number(imp.inventory_buffer_days).toFixed(0) + "d", cls: "num" },
        {
          val:
            imp.shock_onset_day !== null && imp.shock_onset_day !== undefined
              ? "Day " + Number(imp.shock_onset_day).toFixed(1)
              : "—",
          cls: "num",
        },
        { val: Number(imp.unbuffered_downtime_days).toFixed(1) + "d", cls: "num" },
        { val: Number(imp.capacity_contraction_pct).toFixed(1) + "%", cls: "num" },
        { val: imp.verdict, cls: "verdict" },
      ];
      cells.forEach((c, idx) => {
        const td = makeEl("td", c.cls === "num" ? "num" : "", c.val);
        if (idx === 0) td.style.fontWeight = "600";
        if (c.cls === "verdict") {
          td.style.fontWeight = "600";
          td.style.color =
            c.val === "UNBUFFERED_DISRUPTION"
              ? "var(--error)"
              : c.val === "BUFFER_ABSORBED"
                ? "var(--warning)"
                : "var(--success)";
        }
        tr.appendChild(td);
      });
      els.impactsTbody.appendChild(tr);
    });

    // Card 3: Citation Violations, Synthesis & 1NF Rows
    const cleanCit = violations.length === 0;
    els.citationPill.textContent = cleanCit
      ? "Verbatim Citation Check: 0 Violations"
      : "Verbatim Citation Violations: " + violations.length;
    els.citationPill.style.borderColor = cleanCit ? "var(--success)" : "var(--error)";
    els.citationPill.style.color = cleanCit ? "var(--success)" : "var(--error)";

    clearNode(els.citationViolationsBox);
    els.citationViolationsBox.classList.toggle("hidden", cleanCit);
    if (!cleanCit) {
      violations.forEach((v) => {
        els.citationViolationsBox.appendChild(makeEl("div", "", v));
      });
    }

    els.reportIdTag.textContent = report.report_id + " · " + report.event_id;
    els.synthesisOut.textContent = report.operational_synthesis;
    els.thinkTraceOut.textContent = report.contagion_think_trace || "(No think trace recorded)";

    clearNode(els.telemetryPills);
    const tel = report.telemetry || {};
    Object.entries(tel).forEach(([k, v]) => {
      els.telemetryPills.appendChild(makeEl("span", "pill", k + ": " + v + "ms"));
    });

    els.oneNfSummaryTitle.textContent =
      "1NF Time-Series Records (" + tsRows.length + " row" + (tsRows.length === 1 ? "" : "s") + ")";
    clearNode(els.oneNfTbody);
    tsRows.forEach((row) => {
      const tr = document.createElement("tr");
      [
        row.event_id,
        row.disrupted_entity_ticker,
        row.target_entity_ticker,
        row.sec_item,
        row.calendar_onset_date || "null",
        row.calendar_recovery_date || "null",
        row.verdict,
        row.model_mode,
      ].forEach((val) => {
        tr.appendChild(makeEl("td", "", val));
      });
      els.oneNfTbody.appendChild(tr);
    });
    els.jsonlStatusText.textContent = "";
  }

  function showEvalError(msg) {
    els.evalErrorBox.textContent = msg;
    els.evalErrorBox.classList.toggle("hidden", !msg);
  }

  async function runEvaluation() {
    showEvalError("");
    const body = { use_mock: els.modeSelect.value !== "LIVE_LLM" };

    const topoVal = els.topologySelect.value;
    if (topoVal.startsWith("snapshot:")) {
      body.snapshot_id = topoVal.slice("snapshot:".length);
    } else if (topoVal.startsWith("edges:")) {
      body.edges_file = topoVal.slice("edges:".length);
    }

    const target = (els.targetTickerInput.value || "").trim().toUpperCase();
    if (target) body.target_ticker = target;

    if (state.inputMode === "fixture") {
      if (!els.fixtureSelect.value) {
        showEvalError("No fixture selected. Switch to Inline mode to submit custom 8-K text.");
        return;
      }
      body.filing_path = els.fixtureSelect.value;
      const filings = (state.catalog && state.catalog.filings) || [];
      const chosen = filings.find((f) => f.path === body.filing_path);
      state.rawInputText = chosen ? chosen.headline || body.filing_path : body.filing_path;
    } else {
      let metadataObj = {};
      const rawMeta = (els.inMetadata.value || "").trim();
      if (rawMeta) {
        try {
          metadataObj = JSON.parse(rawMeta);
        } catch (err) {
          showEvalError("Optional Metadata JSON is invalid: " + err.message);
          return;
        }
      }
      const itemsList = (els.inItems.value || "Item 8.01 Other Events")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const rawContent = els.inContent.value || "";
      state.rawInputText = rawContent;
      body.inline_filing = {
        ticker: (els.inTicker.value || "CUSTOM").trim().toUpperCase(),
        company_name: (els.inCompany.value || "Custom Corporate Entity").trim(),
        filing_date: (els.inDate.value || "2024-10-28").trim(),
        items: itemsList.length ? itemsList : ["Item 8.01 Other Events"],
        headline: els.inHeadline.value || "",
        content: rawContent,
        metadata: metadataObj,
      };
    }

    els.evaluateBtn.disabled = true;
    els.evaluateBtn.textContent = "Evaluating...";
    try {
      const resp = await fetch("/api/evaluate-8k", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showEvalError("HTTP " + resp.status + ": " + (data.detail || "Evaluation failed."));
        return;
      }
      renderEvaluation(data);
    } catch (err) {
      showEvalError("Request error: " + err.message);
    } finally {
      els.evaluateBtn.disabled = false;
      els.evaluateBtn.textContent = "Run Event Audit";
    }
  }

  function buildJsonlText() {
    const rows = (state.lastEvaluation && state.lastEvaluation.timeseries_rows) || [];
    return rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : "");
  }

  async function init() {
    document.querySelectorAll(".tab-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        state.activeTab = btn.dataset.tab;
        renderTabs();
      })
    );
    els.srcFixtureBtn.addEventListener("click", () => {
      state.inputMode = "fixture";
      renderInputMode();
    });
    els.srcInlineBtn.addEventListener("click", () => {
      state.inputMode = "inline";
      renderInputMode();
    });
    els.fixtureSelect.addEventListener("change", renderFixtureMeta);
    els.evaluateBtn.addEventListener("click", runEvaluation);
    els.toggleDiffBtn.addEventListener("click", () => {
      state.showRawDiff = !state.showRawDiff;
      els.rawPanel.classList.toggle("hidden", !state.showRawDiff);
      els.toggleDiffBtn.textContent = state.showRawDiff ? "Hide Raw vs. Stripped" : "Show Raw vs. Stripped";
    });
    els.themeSelect.addEventListener("change", () => {
      document.documentElement.setAttribute("data-theme", els.themeSelect.value);
    });
    els.densitySelect.addEventListener("change", () => {
      document.body.setAttribute("data-density", els.densitySelect.value);
      if (state.lastEvaluation) renderGanttSvg(state.lastEvaluation.report);
    });
    els.copyJsonlBtn.addEventListener("click", async () => {
      const txt = buildJsonlText();
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
        els.jsonlStatusText.textContent = "Copied JSONL to clipboard";
      } catch (_err) {
        els.jsonlStatusText.textContent = "Clipboard unavailable in browser context";
      }
    });
    els.downloadJsonlBtn.addEventListener("click", () => {
      const txt = buildJsonlText();
      if (!txt) return;
      const blob = new Blob([txt], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "contagion_timeseries_rows.jsonl";
      a.click();
      URL.revokeObjectURL(url);
      els.jsonlStatusText.textContent = "Downloaded JSONL";
    });

    renderTabs();
    renderInputMode();

    try {
      const resp = await fetch("/api/catalog");
      if (!resp.ok) {
        showEvalError("Failed to load workspace catalog (HTTP " + resp.status + ").");
        return;
      }
      const catalog = await resp.json();
      populateCatalog(catalog);
      if ((catalog.filings || []).length > 0) {
        await runEvaluation();
      }
    } catch (err) {
      showEvalError("Failed to connect to local workbench API: " + err.message);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
