(() => {
  let activeReport = null;
  let activeLayer = "heatmap";
  let currentAccount = "mini";
  const scanHistory = [];

  // DOM Elements
  const dropzoneCard = document.getElementById("dropzoneCard");
  const fileInput = document.getElementById("fileInput");
  const chooseFileBtn = document.getElementById("chooseFileBtn");
  const scanOverlay = document.getElementById("scanOverlay");
  const scanStageLabel = document.getElementById("scanStageLabel");
  const scanProgressBar = document.getElementById("scanProgressBar");
  const inspectorCard = document.getElementById("inspectorCard");
  const newScanBtn = document.getElementById("newScanBtn");
  const howItWorksCard = document.getElementById("howItWorksCard");
  const telemetryCard = document.getElementById("telemetryCard");
  const baseImageLayer = document.getElementById("baseImageLayer");
  const overlayImageLayer = document.getElementById("overlayImageLayer");
  const bboxOverlayContainer = document.getElementById("bboxOverlayContainer");
  const heatmapOpacity = document.getElementById("heatmapOpacity");
  const opacityControlWrap = document.getElementById("opacityControlWrap");
  const toggleRegionsCheck = document.getElementById("toggleRegionsCheck");
  const samplesRow = document.getElementById("samplesRow");
  const loginDialog = document.getElementById("loginDialog");

  // --- Modern Web Guidance: Top-layer exit transition & light-dismiss fallbacks ---
  let supportsDisplayTransition;
  function canTransitionDisplay() {
    if (supportsDisplayTransition !== undefined) return supportsDisplayTransition;
    if (!window.CSS?.supports?.("transition-behavior", "allow-discrete") || !document.body) {
      return false;
    }
    const probe = document.createElement("div");
    probe.style.cssText = "transition: display 1s allow-discrete !important; display: block;";
    document.body.appendChild(probe);
    getComputedStyle(probe).display;
    probe.style.display = "none";
    supportsDisplayTransition = getComputedStyle(probe).display === "block";
    probe.remove();
    return supportsDisplayTransition;
  }

  function supportsTopLayerExit() {
    return window.CSS?.supports?.("overlay", "auto") && canTransitionDisplay();
  }

  async function closeTopLayer(element) {
    if (!supportsTopLayerExit()) {
      element.setAttribute("data-closing", "");
      const animations = element.getAnimations({ subtree: true });
      if (animations.length > 0) {
        await Promise.race([
          Promise.allSettled(animations.map((a) => a.finished)),
          new Promise((r) => setTimeout(r, 2000)),
        ]);
      }
      if (!element.hasAttribute("data-closing")) return;
      element.removeAttribute("data-closing");
    }
    element.close();
  }

  if (loginDialog) {
    loginDialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      closeTopLayer(loginDialog);
    });

    if (!("closedBy" in HTMLDialogElement.prototype)) {
      loginDialog.addEventListener("click", (event) => {
        if (event.target !== loginDialog) return;
        const rect = loginDialog.getBoundingClientRect();
        const isDialogContent =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (isDialogContent) return;
        closeTopLayer(loginDialog);
      });
    }
  }

  if (!window.CSS?.supports?.("accent-color", "#6875f5") && heatmapOpacity) {
    heatmapOpacity.addEventListener("input", (e) => {
      e.target.style.setProperty("--progress", `${e.target.value}%`);
    });
  }

  // --- Engine Status (.env) ---
  async function refreshEngineStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const dot = document.getElementById("engineStatusDot");
      const text = document.getElementById("engineStatusText");
      const groqBadge = document.getElementById("groqStatusBadge");
      const openrouterBadge = document.getElementById("openrouterStatusBadge");
      const openaiBadge = document.getElementById("openaiStatusBadge");

      text.textContent = data.active_mode;
      if (data.groq_configured || data.openrouter_configured || data.openai_configured) {
        dot.classList.remove("warn");
      } else {
        dot.classList.add("warn");
      }

      if (groqBadge) {
        groqBadge.textContent = data.groq_configured ? `Ready (${data.groq_model})` : "Not Set in .env";
        groqBadge.className = `status-pill ${data.groq_configured ? "pill-ok" : "pill-warn"}`;
      }
      if (openrouterBadge) {
        openrouterBadge.textContent = data.openrouter_configured ? `Ready (${data.openrouter_model})` : "Not Set in .env";
        openrouterBadge.className = `status-pill ${data.openrouter_configured ? "pill-ok" : "pill-warn"}`;
      }
      if (openaiBadge) {
        openaiBadge.textContent = data.openai_configured ? `Ready (${data.openai_model})` : "Not Set in .env";
        openaiBadge.className = `status-pill ${data.openai_configured ? "pill-ok" : "pill-warn"}`;
      }
    } catch {
      document.getElementById("engineStatusText").textContent = "Local Signal Engine";
    }
  }

  // --- Load Sample Images ---
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const { samples } = await res.json();
      samplesRow.innerHTML = "";
      samples.forEach((s) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "sample-card pressable";
        card.innerHTML = `
          <img src="${s.url}" alt="${s.title}" loading="lazy" />
          <div class="sample-card-overlay">
            <span class="sample-card-title">${s.title}</span>
            <span class="sample-card-tag">${s.tag}</span>
          </div>
        `;
        card.addEventListener("click", () => runSampleAnalysis(s.id));
        samplesRow.appendChild(card);
      });
    } catch (err) {
      console.error("Could not load samples:", err);
    }
  }

  // --- Staged Scan Progress Helper ---
  function startScanProgress() {
    dropzoneCard.classList.remove("hidden");
    inspectorCard.classList.add("hidden");
    scanOverlay.classList.remove("hidden");
    const stages = [
      { pct: 22, text: "Validating container & EXIF quantization tables…" },
      { pct: 52, text: "Computing 2D Fast Fourier Transform (FFT) spectrum…" },
      { pct: 76, text: "Generating ELA & high-pass noise residual heatmap…" },
      { pct: 92, text: "Fusing VLM semantic audit & multi-head attribution…" },
    ];
    let idx = 0;
    scanProgressBar.value = stages[0].pct;
    scanStageLabel.textContent = stages[0].text;
    const timer = setInterval(() => {
      idx += 1;
      if (idx < stages.length) {
        scanProgressBar.value = stages[idx].pct;
        scanStageLabel.textContent = stages[idx].text;
      }
    }, 320);
    return () => {
      clearInterval(timer);
      scanOverlay.classList.add("hidden");
    };
  }

  async function runFileAnalysis(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("VisionForensics-X accepts still images only (JPG, PNG, WEBP). Video files are not supported.");
      return;
    }
    switchWorkbenchTab("detect");
    document.getElementById("workbench").scrollIntoView({ behavior: "smooth", block: "start" });
    const stopProgress = startScanProgress();

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`/api/analyze?account=${encodeURIComponent(currentAccount)}`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Analysis failed" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const report = await res.json();
      stopProgress();
      renderForensicReport(report);
    } catch (err) {
      stopProgress();
      alert(err.message || "Could not analyze image.");
    }
  }

  async function runSampleAnalysis(sampleId) {
    switchWorkbenchTab("detect");
    document.getElementById("workbench").scrollIntoView({ behavior: "smooth", block: "start" });
    const stopProgress = startScanProgress();

    try {
      const res = await fetch(`/api/analyze-sample/${encodeURIComponent(sampleId)}?account=${encodeURIComponent(currentAccount)}`, { method: "POST" });
      if (!res.ok) throw new Error("Sample analysis failed.");
      const report = await res.json();
      stopProgress();
      renderForensicReport(report);
    } catch (err) {
      stopProgress();
      alert(err.message);
    }
  }

  async function loadAccountHistory() {
    try {
      const res = await fetch(`/api/account/${encodeURIComponent(currentAccount)}/history`);
      if (!res.ok) return;
      const data = await res.json();
      scanHistory.length = 0;
      (data.history || []).forEach((item) => scanHistory.push(item));
      updateHistoryUI();
    } catch (err) {
      console.error("Could not load account history:", err);
    }
  }

  // --- Render Report into Workbench ---
  function renderForensicReport(report) {
    activeReport = report;
    if (!scanHistory.some((h) => h.report_id === report.report_id && h.filename === report.filename)) {
      scanHistory.unshift(report);
      updateHistoryUI();
    }

    dropzoneCard.classList.add("hidden");
    inspectorCard.classList.remove("hidden");
    newScanBtn.classList.remove("hidden");
    howItWorksCard.classList.add("hidden");
    telemetryCard.classList.remove("hidden");

    // Set base & overlay layers
    baseImageLayer.src = report.visual_layers.original;
    applyVisualLayer(activeLayer);
    renderBoundingBoxes(report.suspicious_regions || []);

    // Metadata footer
    document.getElementById("resReportId").textContent = report.report_id;
    document.getElementById("resDims").textContent = report.dimensions;
    document.getElementById("resFormat").textContent = `${report.format} (${report.file_size_kb} KB)`;
    document.getElementById("resLatency").textContent = `${report.elapsed_ms} ms`;

    // Verdict Banner
    const banner = document.getElementById("verdictBanner");
    banner.className = "verdict-banner";
    if (report.verdict === "AI-Generated") banner.classList.add("verdict-ai");
    else if (report.verdict.includes("Edited")) banner.classList.add("verdict-edited");
    else banner.classList.add("verdict-real");

    document.getElementById("verdictLabel").textContent = report.verdict;
    document.getElementById("synthProbNum").textContent = `${report.synthetic_probability}%`;
    document.getElementById("manipulationCategory").textContent = `Category: ${report.manipulation_category} · Authenticity: ${report.authenticity_score}%`;

    // Branch Scores
    const bs = report.branch_scores;
    document.getElementById("activeVlmBadge").textContent = bs.vlm_active ? bs.vlm_provider : "Signal Core";
    setBar("Spatial", bs.spatial);
    setBar("Freq", bs.frequency);
    setBar("Vlm", bs.vlm);
    setBar("Meta", bs.metadata);

    // VLM Reasoning & Findings
    document.getElementById("vlmReasoningText").textContent = report.vlm_reasoning;
    const findingsList = document.getElementById("findingsList");
    findingsList.innerHTML = "";
    (report.findings || []).forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      findingsList.appendChild(li);
    });

    // Attribution & Uncertainty
    const uncBadge = document.getElementById("uncertaintyBadge");
    uncBadge.textContent = `${report.uncertainty.level} (${report.uncertainty.reliability_pct}%)`;
    const attrList = document.getElementById("attributionList");
    attrList.innerHTML = "";
    (report.attribution || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "attr-row";
      row.innerHTML = `<span>${item.family}</span><strong class="mono-val">${item.probability}%</strong>`;
      attrList.appendChild(row);
    });

    // Populate 4-Quadrant Spectral Matrix & Radial SVG
    document.getElementById("quadOrig").src = report.visual_layers.original;
    document.getElementById("quadHeat").src = report.visual_layers.heatmap;
    document.getElementById("quadFft").src = report.visual_layers.fft;
    document.getElementById("quadEla").src = report.visual_layers.ela;
    renderRadialChart(report.radial_spectrum || []);

    // Populate JSON report viewer (excluding huge base64 strings for readability)
    const cleanJson = { ...report, visual_layers: "[4 Base64 Forensic Image Layers Embedded]" };
    document.getElementById("jsonReportViewer").textContent = JSON.stringify(cleanJson, null, 2);
  }

  function setBar(suffix, val) {
    const numEl = document.getElementById(`score${suffix}`);
    const fillEl = document.getElementById(`fill${suffix}`);
    if (numEl) numEl.textContent = `${val}%`;
    if (fillEl) fillEl.style.width = `${Math.min(100, Math.max(4, val))}%`;
  }

  function applyVisualLayer(layerName) {
    activeLayer = layerName;
    document.querySelectorAll(".layer-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.layer === layerName);
    });
    if (!activeReport) return;

    if (layerName === "original") {
      overlayImageLayer.style.opacity = "0";
      opacityControlWrap.classList.add("hidden");
    } else {
      overlayImageLayer.src = activeReport.visual_layers[layerName];
      opacityControlWrap.classList.remove("hidden");
      overlayImageLayer.style.opacity = String(Number(heatmapOpacity.value) / 100);
    }
  }

  function renderBoundingBoxes(regions) {
    bboxOverlayContainer.innerHTML = "";
    if (!toggleRegionsCheck.checked) return;
    regions.forEach((r, idx) => {
      const box = document.createElement("div");
      box.className = "Forensic-bbox";
      box.style.left = `${r.x_pct}%`;
      box.style.top = `${r.y_pct}%`;
      box.style.width = `${r.w_pct}%`;
      box.style.height = `${r.h_pct}%`;
      box.textContent = `R${idx + 1}: ${r.intensity}%`;
      box.title = r.reason;
      bboxOverlayContainer.appendChild(box);
    });
  }

  function renderRadialChart(profile) {
    const svg = document.getElementById("radialSpectrumSvg");
    if (!svg || !profile.length) return;
    const w = 600;
    const h = 130;
    const pad = 18;
    const pts = profile.map((v, i) => {
      const x = pad + (i / (profile.length - 1)) * (w - pad * 2);
      const y = h - pad - v * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const areaPts = `${pad},${h - pad} ${pts.join(" ")} ${w - pad},${h - pad}`;
    svg.innerHTML = `
      <defs>
        <linearGradient id="radFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#818cf8" stop-opacity="0.45"/>
          <stop offset="100%" stop-color="#818cf8" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      <polygon points="${areaPts}" fill="url(#radFill)" />
      <polyline fill="none" stroke="#818cf8" stroke-width="2.4" points="${pts.join(" ")}" />
    `;
  }

  function updateHistoryUI() {
    document.getElementById("historyCountBadge").textContent = String(scanHistory.length);
    const container = document.getElementById("historyListContainer");
    if (!scanHistory.length) {
      container.innerHTML = `<div class="empty-state">No images saved in ${currentAccount.toUpperCase()} account yet.</div>`;
      return;
    }
    container.innerHTML = "";
    scanHistory.forEach((item) => {
      const card = document.createElement("div");
      card.className = "history-item pressable";
      const thumb = item.visual_layers?.heatmap || item.visual_layers?.original || "";
      card.innerHTML = `
        ${thumb ? `<img src="${thumb}" alt="${item.filename}" style="width:100%;height:115px;object-fit:cover;border-radius:8px;margin-bottom:0.55rem;border:1px solid rgba(114,134,255,0.18);" />` : ""}
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;gap:0.4rem;">
          <strong style="font-size:0.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${item.filename}</strong>
          <span class="mono-tag">${item.synthetic_probability}% AI</span>
        </div>
        <div style="font-size:0.73rem;color:#94a3b8;">${item.verdict} · ${item.report_id}</div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:0.2rem;">${item.branch_scores?.vlm_provider || "Signal Core"}</div>
      `;
      card.addEventListener("click", () => {
        switchWorkbenchTab("detect");
        renderForensicReport(item);
      });
      container.appendChild(card);
    });
  }

  function switchWorkbenchTab(tabId) {
    document.querySelectorAll(".wb-nav-item").forEach((btn) => {
      const active = btn.dataset.wbTab === tabId;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".wb-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `panel-${tabId}`);
    });
  }

  function downloadJsonReport() {
    if (!activeReport) return;
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeReport.report_id}_${activeReport.filename}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // --- Event Listeners ---
  chooseFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzoneCard.addEventListener("click", () => fileInput.click());
  dropzoneCard.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files?.[0]) runFileAnalysis(e.target.files[0]);
  });

  dropzoneCard.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzoneCard.classList.add("drag-over");
  });
  dropzoneCard.addEventListener("dragleave", () => dropzoneCard.classList.remove("drag-over"));
  dropzoneCard.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzoneCard.classList.remove("drag-over");
    if (e.dataTransfer.files?.[0]) runFileAnalysis(e.dataTransfer.files[0]);
  });

  newScanBtn.addEventListener("click", () => {
    inspectorCard.classList.add("hidden");
    dropzoneCard.classList.remove("hidden");
    newScanBtn.classList.add("hidden");
    telemetryCard.classList.add("hidden");
    howItWorksCard.classList.remove("hidden");
    fileInput.value = "";
  });

  document.querySelectorAll(".layer-tab").forEach((btn) => {
    btn.addEventListener("click", () => applyVisualLayer(btn.dataset.layer));
  });

  heatmapOpacity.addEventListener("input", () => {
    if (activeLayer !== "original") {
      overlayImageLayer.style.opacity = String(Number(heatmapOpacity.value) / 100);
    }
  });

  toggleRegionsCheck.addEventListener("change", () => {
    if (activeReport) renderBoundingBoxes(activeReport.suspicious_regions || []);
  });

  document.querySelectorAll(".wb-nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchWorkbenchTab(btn.dataset.wbTab));
  });

  document.querySelectorAll("[data-scroll]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.querySelector(btn.dataset.scroll);
      if (target) target.scrollIntoView({ behavior: "smooth" });
    });
  });

  document.getElementById("heroUploadBtn").addEventListener("click", () => {
    document.getElementById("workbench").scrollIntoView({ behavior: "smooth" });
    setTimeout(() => fileInput.click(), 350);
  });

  document.getElementById("heroSampleBtn").addEventListener("click", () => {
    runSampleAnalysis("ai_diffusion_synth.png");
  });

  document.querySelectorAll("[data-sample-trigger]").forEach((card) => {
    card.addEventListener("click", () => runSampleAnalysis(card.dataset.sampleTrigger));
  });

  document.getElementById("exportJsonBtn").addEventListener("click", downloadJsonReport);
  document.getElementById("reportTabJsonBtn").addEventListener("click", downloadJsonReport);
  document.getElementById("exportPdfBtn").addEventListener("click", () => window.print());
  document.getElementById("reportTabPrintBtn").addEventListener("click", () => window.print());

  document.getElementById("clearHistoryBtn").addEventListener("click", async () => {
    await fetch(`/api/account/${encodeURIComponent(currentAccount)}/history`, { method: "DELETE" });
    scanHistory.length = 0;
    updateHistoryUI();
  });

  document.getElementById("refreshEnvBtn").addEventListener("click", refreshEngineStatus);

  // Login Modal triggers
  document.getElementById("openLoginBtn").addEventListener("click", () => {
    loginDialog.showModal();
  });
  document.getElementById("closeLoginBtn").addEventListener("click", () => {
    closeTopLayer(loginDialog);
  });
  document.getElementById("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const email = document.getElementById("loginEmail").value.trim() || "mini@visionforensics.ai";
    const namePart = email.split("@")[0].replace(/[^a-zA-Z0-9]/g, "") || "Mini";
    currentAccount = namePart.toLowerCase();
    const formatted = namePart.charAt(0).toUpperCase() + namePart.slice(1, 12);
    document.getElementById("userNameDisplay").textContent = formatted;
    document.getElementById("userAvatarBadge").textContent = formatted.charAt(0).toUpperCase();
    document.getElementById("openLoginBtn").querySelector("span").textContent = formatted;
    loadAccountHistory();
    closeTopLayer(loginDialog);
  });

  // Initialize
  refreshEngineStatus();
  loadSamples();
  loadAccountHistory();
})();
