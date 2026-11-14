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
    kgDirection: "both",
    lastSubgraph: null,
    selectedEdgeKey: null,
    lastCohort: null,
    cohortOffset: 0,
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
    kgSourceSelect: document.getElementById("kgSourceSelect"),
    kgTickerSelect: document.getElementById("kgTickerSelect"),
    kgHopsSelect: document.getElementById("kgHopsSelect"),
    kgDateInput: document.getElementById("kgDateInput"),
    kgClearDateBtn: document.getElementById("kgClearDateBtn"),
    kgRelSelect: document.getElementById("kgRelSelect"),
    kgStatusSelect: document.getElementById("kgStatusSelect"),
    kgQueryBtn: document.getElementById("kgQueryBtn"),
    kgErrorBox: document.getElementById("kgErrorBox"),
    kgProvenancePanel: document.getElementById("kgProvenancePanel"),
    kgCanvasTitle: document.getElementById("kgCanvasTitle"),
    kgSvgMount: document.getElementById("kgSvgMount"),
    kgEdgeCountSpan: document.getElementById("kgEdgeCountSpan"),
    kgSourceMetaSpan: document.getElementById("kgSourceMetaSpan"),
    kgEdgesTbody: document.getElementById("kgEdgesTbody"),
    cohortSelect: document.getElementById("cohortSelect"),
    cohortCustomPathInput: document.getElementById("cohortCustomPathInput"),
    cVerdictFilter: document.getElementById("cVerdictFilter"),
    cCategoryFilter: document.getElementById("cCategoryFilter"),
    cItemFilter: document.getElementById("cItemFilter"),
    cTickerFilter: document.getElementById("cTickerFilter"),
    cLimitSelect: document.getElementById("cLimitSelect"),
    cohortQueryBtn: document.getElementById("cohortQueryBtn"),
    cohortMatchSummary: document.getElementById("cohortMatchSummary"),
    cohortErrorBox: document.getElementById("cohortErrorBox"),
    cohortSummaryRow: document.getElementById("cohortSummaryRow"),
    cohortTbody: document.getElementById("cohortTbody"),
    cohortPageStatus: document.getElementById("cohortPageStatus"),
    cohortPrevBtn: document.getElementById("cohortPrevBtn"),
    cohortNextBtn: document.getElementById("cohortNextBtn"),
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

  function setKgTickers(tickers, preferredTicker) {
    const unique = [...new Set((tickers || []).filter(Boolean))];
    clearNode(els.kgTickerSelect);
    unique.forEach((t) => {
      const opt = makeEl("option", "", t);
      opt.value = t;
      els.kgTickerSelect.appendChild(opt);
    });
    const allOpt = makeEl("option", "", "ALL (Full Snapshot)");
    allOpt.value = "";
    els.kgTickerSelect.appendChild(allOpt);

    if (preferredTicker !== undefined && preferredTicker !== null) {
      if (preferredTicker && !unique.includes(preferredTicker)) {
        const extra = makeEl("option", "", preferredTicker);
        extra.value = preferredTicker;
        els.kgTickerSelect.insertBefore(extra, allOpt);
      }
      els.kgTickerSelect.value = preferredTicker;
    } else if (unique.includes("TSM")) {
      els.kgTickerSelect.value = "TSM";
    } else if (unique.length > 0) {
      els.kgTickerSelect.value = unique[0];
    } else {
      els.kgTickerSelect.value = "";
    }
  }

  function syncKgSourceTickers(preferredTicker) {
    if (!state.catalog) return;
    const val = els.kgSourceSelect.value;
    if (val.startsWith("snapshot:")) {
      const sid = val.slice("snapshot:".length);
      const snap = (state.catalog.snapshots || []).find((s) => s.snapshot_id === sid);
      setKgTickers(snap ? snap.tickers : [], preferredTicker);
    } else if (val.startsWith("edges:")) {
      const epath = val.slice("edges:".length);
      const manifest = (state.catalog.edge_manifests || []).find((m) => m.path === epath);
      setKgTickers(manifest ? manifest.tickers : [], preferredTicker);
    } else {
      setKgTickers([], "");
    }
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

    clearNode(els.kgSourceSelect);
    (catalog.snapshots || []).forEach((s) => {
      const label = "Snapshot: " + s.snapshot_id + " (" + s.edge_count + " edges)";
      const opt1 = makeEl("option", "", label);
      opt1.value = "snapshot:" + s.snapshot_id;
      els.topologySelect.appendChild(opt1);

      const opt2 = makeEl("option", "", label);
      opt2.value = "snapshot:" + s.snapshot_id;
      els.kgSourceSelect.appendChild(opt2);
    });
    (catalog.edge_manifests || []).forEach((m) => {
      const label = "Manifest: " + m.name + " (" + m.edge_count + " edges)";
      const opt1 = makeEl("option", "", label);
      opt1.value = "edges:" + m.path;
      els.topologySelect.appendChild(opt1);

      const opt2 = makeEl("option", "", label);
      opt2.value = "edges:" + m.path;
      els.kgSourceSelect.appendChild(opt2);
    });

    if (els.kgSourceSelect.querySelector("option[value='snapshot:G_resolved']")) {
      els.kgSourceSelect.value = "snapshot:G_resolved";
    }
    if (!els.kgSourceSelect.options.length) {
      const emptyOpt = makeEl("option", "", "No local snapshots or manifests found");
      emptyOpt.value = "";
      els.kgSourceSelect.appendChild(emptyOpt);
    }
    syncKgSourceTickers();

    clearNode(els.cohortSelect);
    const cohorts = catalog.cohorts || [];
    if (!cohorts.length) {
      const noneOpt = makeEl("option", "", "No .jsonl cohorts auto-discovered (enter path)");
      noneOpt.value = "";
      els.cohortSelect.appendChild(noneOpt);
    } else {
      cohorts.forEach((c) => {
        const opt = makeEl("option", "", c.name + " (" + c.record_count + " rows)");
        opt.value = c.path;
        els.cohortSelect.appendChild(opt);
      });
    }
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

  // ============================================================================
  // ADR-00019: PiT KG Explorer (GET /api/subgraph)
  // ============================================================================

  function makeEdgeKey(e) {
    return [
      e.source || "",
      e.rel_type || "",
      e.target || "",
      e.input_group || e.nature || "",
    ].join("|");
  }

  function showKgError(msg) {
    els.kgErrorBox.textContent = msg;
    els.kgErrorBox.classList.toggle("hidden", !msg);
  }

  function renderKgDirectionButtons() {
    document.querySelectorAll(".dir-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.dir === state.kgDirection);
    });
  }

  function renderKgProvenance(edge) {
    clearNode(els.kgProvenancePanel);
    if (!edge) {
      els.kgProvenancePanel.appendChild(
        makeEl(
          "div",
          "muted-text",
          "Click any arrow in the graph or row in the table below to inspect contract dates, buffers, and source hash."
        )
      );
      return;
    }
    const fields = [
      ["Edge", (edge.source || "?") + " → " + (edge.target || "?")],
      ["Relationship", edge.rel_type || "—"],
      ["Nature / Input", edge.nature || edge.input_group || "—"],
      [
        "Procurement Share",
        edge.share !== null && edge.share !== undefined
          ? (Number(edge.share) * 100).toFixed(1) + "%"
          : "N/A",
      ],
      [
        "Inventory Buffer",
        edge.buffer_days !== null && edge.buffer_days !== undefined
          ? Number(edge.buffer_days).toFixed(0) + " days"
          : "N/A",
      ],
      [
        "Confidence / Evid.",
        edge.confidence !== null && edge.confidence !== undefined
          ? String(edge.confidence) +
            (edge.evidence_count !== undefined ? " (" + edge.evidence_count + " src)" : "")
          : "—",
      ],
      ["Valid From", edge.valid_from || "Unbounded"],
      ["Valid To", edge.valid_to || "Active / Open"],
      ["Status", edge.status || "—"],
      ["Source Hash", edge.source_hash || "—"],
    ];
    fields.forEach(([k, v]) => {
      const row = makeEl("div", "kv-row");
      row.appendChild(makeEl("span", "kv-key", k));
      row.appendChild(makeEl("span", "kv-val mono", v));
      els.kgProvenancePanel.appendChild(row);
    });
  }

  function computeNodeLanes(nodes, edges, rootTicker) {
    const laneById = new Map();
    if (rootTicker) {
      nodes.forEach((n) => {
        if (n.role === "ROOT" || Number(n.hop) === 0) {
          laneById.set(n.id, 0);
        } else if (n.role === "UPSTREAM") {
          laneById.set(n.id, -Math.max(1, Number(n.hop || 1)));
        } else {
          laneById.set(n.id, Math.max(1, Number(n.hop || 1)));
        }
      });
      return laneById;
    }

    nodes.forEach((n) => laneById.set(n.id, 0));
    const maxPasses = Math.min(8, Math.max(2, nodes.length));
    for (let pass = 0; pass < maxPasses; pass++) {
      let changed = false;
      edges.forEach((e) => {
        const sl = laneById.get(e.source) || 0;
        const tl = laneById.get(e.target) || 0;
        if (tl <= sl) {
          laneById.set(e.target, sl + 1);
          changed = true;
        }
      });
      if (!changed) break;
    }
    return laneById;
  }

  function renderKgSubgraph(data) {
    state.lastSubgraph = data;
    const nodes = data.nodes || [];
    const edges = data.edges || [];

    if (edges.length && !edges.some((e) => makeEdgeKey(e) === state.selectedEdgeKey)) {
      state.selectedEdgeKey = makeEdgeKey(edges[0]);
    } else if (!edges.length) {
      state.selectedEdgeKey = null;
    }

    const activeEdge = edges.find((e) => makeEdgeKey(e) === state.selectedEdgeKey) || null;
    renderKgProvenance(activeEdge);

    const dateSuffix = data.as_of_date ? " @ " + data.as_of_date : " @ latest";
    els.kgCanvasTitle.textContent = data.root_ticker
      ? data.root_ticker +
        " Ego Network (" +
        data.direction +
        ", ≤" +
        data.max_hops +
        " hops" +
        dateSuffix +
        ")"
      : "Full Snapshot Topology (" + (data.source_path || "snapshot") + dateSuffix + ")";

    els.kgEdgeCountSpan.textContent = String(edges.length);
    els.kgSourceMetaSpan.textContent =
      (data.source_path || "") + " · " + nodes.length + " nodes";

    // Render SVG
    clearNode(els.kgSvgMount);
    if (!nodes.length) {
      els.kgSvgMount.appendChild(
        makeEl("p", "muted-text", "No nodes or active edges matched the current Point-in-Time filters.")
      );
    } else {
      const laneById = computeNodeLanes(nodes, edges, data.root_ticker);
      const lanes = [...new Set(laneById.values())].sort((a, b) => a - b);
      const maxPerLane = Math.max(
        1,
        ...lanes.map((lane) => nodes.filter((n) => laneById.get(n.id) === lane).length)
      );
      const width = Math.max(660, lanes.length * 190 + 90);
      const height = Math.max(250, maxPerLane * 88 + 64);

      const pos = new Map();
      lanes.forEach((lane, li) => {
        const group = nodes
          .filter((n) => laneById.get(n.id) === lane)
          .sort((a, b) => String(a.ticker).localeCompare(String(b.ticker)));
        const x =
          lanes.length === 1
            ? width / 2
            : 95 + (li / (lanes.length - 1)) * (width - 190);
        group.forEach((n, ni) => {
          const y = 34 + ((height - 44) / (group.length + 1)) * (ni + 1);
          pos.set(n.id, { x, y, node: n });
        });
      });

      const svgNs = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNs, "svg");
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      svg.style.width = "100%";
      svg.style.height = "auto";
      svg.style.display = "block";

      const defs = document.createElementNS(svgNs, "defs");
      [
        ["kg-arr-supply", "var(--buffer)"],
        ["kg-arr-other", "var(--error)"],
        ["kg-arr-sel", "var(--primary)"],
      ].forEach(([id, col]) => {
        const marker = document.createElementNS(svgNs, "marker");
        marker.setAttribute("id", id);
        marker.setAttribute("viewBox", "0 0 10 10");
        marker.setAttribute("refX", "9");
        marker.setAttribute("refY", "5");
        marker.setAttribute("markerWidth", "6");
        marker.setAttribute("markerHeight", "6");
        marker.setAttribute("orient", "auto-start-reverse");
        const arrow = document.createElementNS(svgNs, "path");
        arrow.setAttribute("d", "M 0 1 L 10 5 L 0 9 z");
        arrow.setAttribute("fill", col);
        marker.appendChild(arrow);
        defs.appendChild(marker);
      });
      svg.appendChild(defs);

      lanes.forEach((lane, li) => {
        const x =
          lanes.length === 1
            ? width / 2
            : 95 + (li / (lanes.length - 1)) * (width - 190);
        const hdr = document.createElementNS(svgNs, "text");
        hdr.setAttribute("x", String(x));
        hdr.setAttribute("y", "18");
        hdr.setAttribute("text-anchor", "middle");
        hdr.setAttribute("font-size", "11");
        hdr.setAttribute("fill", "var(--muted)");
        hdr.textContent = !data.root_ticker
          ? "Tier " + lane
          : lane < 0
            ? "Upstream H" + Math.abs(lane)
            : lane === 0
              ? "Root Issuer"
              : "Downstream H" + lane;
        svg.appendChild(hdr);
      });

      edges.forEach((e) => {
        const srcPos = pos.get(e.source);
        const dstPos = pos.get(e.target);
        if (!srcPos || !dstPos) return;

        const key = makeEdgeKey(e);
        const isSel = key === state.selectedEdgeKey;
        const isSupply = e.rel_type === "SUPPLIES_TO";
        const sameCol = Math.abs(srcPos.x - dstPos.x) < 1;
        const x1 = sameCol ? srcPos.x : srcPos.x + 42;
        const y1 = sameCol ? srcPos.y + 20 : srcPos.y;
        const x2 = sameCol ? dstPos.x : dstPos.x - 46;
        const y2 = sameCol ? dstPos.y - 24 : dstPos.y;
        const mx = (x1 + x2) / 2;

        const g = document.createElementNS(svgNs, "g");
        g.style.cursor = "pointer";
        g.addEventListener("click", () => {
          state.selectedEdgeKey = key;
          renderKgSubgraph(data);
        });

        const pathD = sameCol
          ? "M " + x1 + " " + y1 + " C " + (x1 + 50) + " " + y1 + ", " + (x2 + 50) + " " + y2 + ", " + x2 + " " + y2
          : "M " + x1 + " " + y1 + " C " + mx + " " + y1 + ", " + mx + " " + y2 + ", " + x2 + " " + y2;

        const hitPath = document.createElementNS(svgNs, "path");
        hitPath.setAttribute("d", pathD);
        hitPath.setAttribute("fill", "none");
        hitPath.setAttribute("stroke", "transparent");
        hitPath.setAttribute("stroke-width", "12");
        g.appendChild(hitPath);

        const visPath = document.createElementNS(svgNs, "path");
        visPath.setAttribute("d", pathD);
        visPath.setAttribute("fill", "none");
        visPath.setAttribute(
          "stroke",
          isSel ? "var(--primary)" : isSupply ? "var(--buffer)" : "var(--error)"
        );
        visPath.setAttribute("stroke-width", isSel ? "2.6" : "1.8");
        if (!isSupply) visPath.setAttribute("stroke-dasharray", "5 4");
        visPath.setAttribute(
          "marker-end",
          isSel ? "url(#kg-arr-sel)" : isSupply ? "url(#kg-arr-supply)" : "url(#kg-arr-other)"
        );
        g.appendChild(visPath);

        const lbl = document.createElementNS(svgNs, "text");
        lbl.setAttribute("x", String(sameCol ? x1 + 48 : mx));
        lbl.setAttribute("y", String((y1 + y2) / 2 - 6));
        lbl.setAttribute("text-anchor", "middle");
        lbl.setAttribute("font-size", "11");
        lbl.setAttribute("fill", isSel ? "var(--primary)" : "var(--muted)");
        const shareStr =
          e.share !== null && e.share !== undefined
            ? Math.round(Number(e.share) * 100) + "%"
            : e.rel_type;
        const bufStr =
          e.buffer_days !== null && e.buffer_days !== undefined
            ? " · " + Number(e.buffer_days).toFixed(0) + "d"
            : "";
        lbl.textContent = shareStr + bufStr;
        g.appendChild(lbl);

        svg.appendChild(g);
      });

      nodes.forEach((n) => {
        const p = pos.get(n.id);
        if (!p) return;
        const isRoot = n.role === "ROOT" || n.ticker === data.root_ticker;
        const g = document.createElementNS(svgNs, "g");
        g.style.cursor = "pointer";
        g.addEventListener("click", () => {
          setKgTickers(data.available_tickers || [], n.ticker);
          querySubgraph();
        });

        const rect = document.createElementNS(svgNs, "rect");
        rect.setAttribute("x", String(p.x - 42));
        rect.setAttribute("y", String(p.y - 19));
        rect.setAttribute("width", "84");
        rect.setAttribute("height", "38");
        rect.setAttribute("rx", "8");
        rect.setAttribute("fill", isRoot ? "var(--primary)" : "var(--card)");
        rect.setAttribute("stroke", isRoot ? "var(--primary)" : "var(--border)");
        rect.setAttribute("stroke-width", isRoot ? "2" : "1.5");
        g.appendChild(rect);

        const txt = document.createElementNS(svgNs, "text");
        txt.setAttribute("x", String(p.x));
        txt.setAttribute("y", String(p.y + 4));
        txt.setAttribute("text-anchor", "middle");
        txt.setAttribute("font-size", "12");
        txt.setAttribute("font-weight", "700");
        txt.setAttribute("fill", isRoot ? "var(--primary-fg)" : "var(--fg)");
        txt.textContent = n.ticker;
        g.appendChild(txt);

        svg.appendChild(g);
      });

      els.kgSvgMount.appendChild(svg);
    }

    // Render Edge Table
    clearNode(els.kgEdgesTbody);
    edges.forEach((e) => {
      const key = makeEdgeKey(e);
      const tr = makeEl(
        "tr",
        "kg-row-clickable" + (key === state.selectedEdgeKey ? " kg-row-selected" : "")
      );
      tr.addEventListener("click", () => {
        state.selectedEdgeKey = key;
        renderKgSubgraph(data);
      });
      const vals = [
        { v: (e.source || "?") + " → " + (e.target || "?"), num: false },
        { v: e.rel_type || "—", num: false },
        { v: e.nature || e.input_group || "—", num: false },
        {
          v:
            e.share !== null && e.share !== undefined
              ? (Number(e.share) * 100).toFixed(1) + "%"
              : "—",
          num: true,
        },
        {
          v:
            e.buffer_days !== null && e.buffer_days !== undefined
              ? Number(e.buffer_days).toFixed(0) + "d"
              : "—",
          num: true,
        },
        { v: (e.valid_from || "*") + " .. " + (e.valid_to || "open"), num: false },
        { v: e.status || "—", num: false },
      ];
      vals.forEach((cell, idx) => {
        const td = makeEl("td", cell.num ? "num" : "", cell.v);
        if (idx === 0) td.style.fontWeight = "600";
        tr.appendChild(td);
      });
      els.kgEdgesTbody.appendChild(tr);
    });
  }

  async function querySubgraph() {
    showKgError("");
    const srcVal = els.kgSourceSelect.value;
    if (!srcVal) {
      showKgError("No local Parquet snapshot or edge manifest is available.");
      return;
    }
    const params = new URLSearchParams();
    if (srcVal.startsWith("snapshot:")) {
      params.set("snapshot_id", srcVal.slice("snapshot:".length));
    } else if (srcVal.startsWith("edges:")) {
      params.set("edges_file", srcVal.slice("edges:".length));
    }
    const ticker = (els.kgTickerSelect.value || "").trim();
    if (ticker) params.set("ticker", ticker);
    params.set("max_hops", els.kgHopsSelect.value || "2");
    params.set("direction", state.kgDirection);
    if (els.kgDateInput.value) params.set("as_of_date", els.kgDateInput.value);
    if (els.kgRelSelect.value) params.set("rel_types", els.kgRelSelect.value);
    params.set("required_status", els.kgStatusSelect.value || "");

    els.kgQueryBtn.disabled = true;
    els.kgQueryBtn.textContent = "Querying...";
    try {
      const resp = await fetch("/api/subgraph?" + params.toString());
      const data = await resp.json();
      if (!resp.ok) {
        showKgError("HTTP " + resp.status + ": " + (data.detail || "Subgraph query failed."));
        return;
      }
      renderKgSubgraph(data);
    } catch (err) {
      showKgError("Request error: " + err.message);
    } finally {
      els.kgQueryBtn.disabled = false;
      els.kgQueryBtn.textContent = "Query Subgraph";
    }
  }

  // ============================================================================
  // ADR-00019: Cohort Browser (POST /api/cohort)
  // ============================================================================

  function showCohortError(msg) {
    els.cohortErrorBox.textContent = msg;
    els.cohortErrorBox.classList.toggle("hidden", !msg);
  }

  function renderDistributionCard(title, countsMap, total) {
    const card = makeEl("div", "dist-card");
    card.appendChild(makeEl("div", "dist-title", title));
    const entries = Object.entries(countsMap || {});
    if (!entries.length) {
      card.appendChild(makeEl("div", "muted-text", "No matching entries"));
      return card;
    }
    entries.slice(0, 5).forEach(([label, count]) => {
      const pct = total > 0 ? Math.round((Number(count) / total) * 100) : 0;
      const row = makeEl("div", "dist-row");
      const head = makeEl("div", "dist-row-head mono");
      head.appendChild(makeEl("span", "", label));
      head.appendChild(makeEl("span", "muted-text", count + " (" + pct + "%)"));
      const track = makeEl("div", "dist-bar-track");
      const fill = makeEl("div", "dist-bar-fill");
      fill.style.width = pct + "%";
      if (label === "UNBUFFERED_DISRUPTION") fill.style.backgroundColor = "var(--error)";
      else if (label === "BUFFER_ABSORBED") fill.style.backgroundColor = "var(--warning)";
      else if (label === "SAFE") fill.style.backgroundColor = "var(--success)";
      track.appendChild(fill);
      row.appendChild(head);
      row.appendChild(track);
      card.appendChild(row);
    });
    return card;
  }

  function renderCohort(data) {
    state.lastCohort = data;
    const records = data.records || [];
    const limit = Number(els.cLimitSelect.value || 50);
    const startIdx = records.length ? state.cohortOffset + 1 : 0;
    const endIdx = state.cohortOffset + records.length;

    els.cohortMatchSummary.textContent =
      "Showing " +
      startIdx +
      "–" +
      endIdx +
      " of " +
      data.filtered_records +
      " matching (" +
      data.total_records +
      " total)";

    clearNode(els.cohortSummaryRow);
    els.cohortSummaryRow.appendChild(
      renderDistributionCard("Verdict Breakdown", data.verdict_counts, data.filtered_records)
    );
    els.cohortSummaryRow.appendChild(
      renderDistributionCard("Disruption Categories", data.category_counts, data.filtered_records)
    );
    els.cohortSummaryRow.appendChild(
      renderDistributionCard("SEC Form 8-K Items", data.sec_item_counts, data.filtered_records)
    );

    clearNode(els.cohortTbody);
    records.forEach((rec) => {
      const tr = document.createElement("tr");
      const cells = [
        { v: rec.filing_date || "—", num: false },
        { v: (rec.disrupted_entity_ticker || "?") + " → " + (rec.target_entity_ticker || "?"), num: false },
        { v: String(rec.hop_distance), num: true },
        { v: rec.sec_item || "—", num: false },
        { v: rec.disruption_category || "—", num: false },
        { v: Number(rec.capacity_contraction_pct || 0).toFixed(1) + "%", num: true },
        { v: Number(rec.unbuffered_downtime_days || 0).toFixed(1) + "d", num: true },
        {
          v: rec.calendar_onset_date
            ? rec.calendar_onset_date + " → " + (rec.calendar_recovery_date || "open")
            : "—",
          num: false,
        },
        { v: rec.verdict || "—", num: false, isVerdict: true },
        { v: rec.model_mode || "—", num: false },
      ];
      cells.forEach((c) => {
        const td = makeEl("td", c.num ? "num" : "", c.v);
        if (c.isVerdict) {
          td.style.fontWeight = "600";
          td.style.color =
            c.v === "UNBUFFERED_DISRUPTION"
              ? "var(--error)"
              : c.v === "BUFFER_ABSORBED"
                ? "var(--warning)"
                : "var(--success)";
        }
        tr.appendChild(td);
      });

      const actTd = document.createElement("td");
      const inspectBtn = makeEl("button", "link-btn", "Inspect Network");
      inspectBtn.type = "button";
      inspectBtn.addEventListener("click", () => {
        const targetTicker = rec.disrupted_entity_ticker || rec.target_entity_ticker || "";
        const avail =
          (state.lastSubgraph && state.lastSubgraph.available_tickers) || [];
        setKgTickers(avail, targetTicker);
        if (rec.filing_date) els.kgDateInput.value = String(rec.filing_date).slice(0, 10);
        state.activeTab = "kg";
        renderTabs();
        querySubgraph();
      });
      actTd.appendChild(inspectBtn);
      tr.appendChild(actTd);
      els.cohortTbody.appendChild(tr);
    });

    els.cohortPageStatus.textContent =
      "Offset " + state.cohortOffset + " · Page size " + limit;
    els.cohortPrevBtn.disabled = state.cohortOffset <= 0;
    els.cohortNextBtn.disabled = state.cohortOffset + limit >= Number(data.filtered_records || 0);
  }

  async function queryCohort(newOffset) {
    showCohortError("");
    state.cohortOffset = Math.max(0, Number(newOffset || 0));
    const chosenPath =
      (els.cohortCustomPathInput.value || "").trim() || (els.cohortSelect.value || "").trim();
    if (!chosenPath) {
      showCohortError(
        "No cohort .jsonl file selected. Pick a discovered cohort or enter an allowlisted workspace path."
      );
      return;
    }

    const body = {
      cohort_path: chosenPath,
      limit: Number(els.cLimitSelect.value || 50),
      offset: state.cohortOffset,
    };
    if (els.cVerdictFilter.value) body.verdict = els.cVerdictFilter.value;
    if ((els.cCategoryFilter.value || "").trim()) {
      body.disruption_category = els.cCategoryFilter.value.trim();
    }
    if ((els.cItemFilter.value || "").trim()) body.sec_item = els.cItemFilter.value.trim();
    if ((els.cTickerFilter.value || "").trim()) body.ticker = els.cTickerFilter.value.trim();

    els.cohortQueryBtn.disabled = true;
    els.cohortQueryBtn.textContent = "Loading...";
    try {
      const resp = await fetch("/api/cohort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showCohortError("HTTP " + resp.status + ": " + (data.detail || "Cohort request failed."));
        return;
      }
      renderCohort(data);
    } catch (err) {
      showCohortError("Request error: " + err.message);
    } finally {
      els.cohortQueryBtn.disabled = false;
      els.cohortQueryBtn.textContent = "Load / Filter Cohort";
    }
  }

  async function init() {
    document.querySelectorAll(".tab-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        state.activeTab = btn.dataset.tab;
        renderTabs();
        if (state.activeTab === "kg" && !state.lastSubgraph && els.kgSourceSelect.value) {
          querySubgraph();
        } else if (
          state.activeTab === "cohort" &&
          !state.lastCohort &&
          ((els.cohortCustomPathInput.value || "").trim() || (els.cohortSelect.value || "").trim())
        ) {
          queryCohort(0);
        }
      })
    );
    document.querySelectorAll(".dir-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        state.kgDirection = btn.dataset.dir;
        renderKgDirectionButtons();
        querySubgraph();
      })
    );
    els.kgSourceSelect.addEventListener("change", () => {
      syncKgSourceTickers();
      querySubgraph();
    });
    els.kgQueryBtn.addEventListener("click", querySubgraph);
    els.kgClearDateBtn.addEventListener("click", () => {
      els.kgDateInput.value = "";
      querySubgraph();
    });
    [els.kgTickerSelect, els.kgHopsSelect, els.kgDateInput, els.kgRelSelect, els.kgStatusSelect].forEach(
      (ctrl) => ctrl.addEventListener("change", querySubgraph)
    );

    els.cohortQueryBtn.addEventListener("click", () => queryCohort(0));
    els.cohortPrevBtn.addEventListener("click", () => {
      const limit = Number(els.cLimitSelect.value || 50);
      queryCohort(Math.max(0, state.cohortOffset - limit));
    });
    els.cohortNextBtn.addEventListener("click", () => {
      const limit = Number(els.cLimitSelect.value || 50);
      queryCohort(state.cohortOffset + limit);
    });

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
    renderKgDirectionButtons();

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
