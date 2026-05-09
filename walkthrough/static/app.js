const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const pageLinks = document.querySelectorAll("[data-page-link]");
const appPages = document.querySelectorAll(".app-page");
const forms = document.querySelectorAll(".analysis-form");
const resultContent = document.querySelector("#resultContent");
const resultMeta = document.querySelector("#resultMeta");
const urgencyBadge = document.querySelector("#urgencyBadge");
const engineStatus = document.querySelector("#engineStatus");
const incidentCount = document.querySelector("#incidentCount");
const highCount = document.querySelector("#highCount");
const incidentList = document.querySelector("#incidentList");
const roleSelect = document.querySelector("#roleSelect");
const languageSelect = document.querySelector("#languageSelect");
const demoScenario = document.querySelector("#demoScenario");
const voiceStatus = document.querySelector("#voiceStatus");
const routeBoard = document.querySelector("#routeBoard");
const medicalNeedCount = document.querySelector("#medicalNeedCount");
const waterNeedCount = document.querySelector("#waterNeedCount");
const blockedRouteCount = document.querySelector("#blockedRouteCount");
const verificationFlagCount = document.querySelector("#verificationFlagCount");
const videoScanForm = document.querySelector("#videoScanForm");
const videoPreview = document.querySelector("#videoPreview");
const videoScanResults = document.querySelector("#videoScanResults");
const videoScanMeta = document.querySelector("#videoScanMeta");
const videoModelSource = document.querySelector("#videoModelSource");
const groundingImage = document.querySelector("#groundingImage");
const groundingPreview = document.querySelector("#groundingPreview");
const discordDestination = document.querySelector("#discordDestination");
const syncQueueList = document.querySelector("#syncQueueList");
const outboxList = document.querySelector("#outboxList");
const offlineProofGrid = document.querySelector("#offlineProofGrid");
let latestGroundingResult = null;

const DEMOS = {
  flooded_school: {
    tab: "shelter",
    location: "Govt School, Ward 7",
    note: "Govt School shelter, Ward 7. 43 people. 6 elderly. 2 insulin patients. Water left for 8 hours. Bridge road blocked near market. Need drinking water, medicine cold storage, blankets, and access route update.",
  },
  washed_road: {
    tab: "damage",
    location: "Creek road washout",
    note: "Road shoulder washed out after flooding. Cones and caution tape are in place. Route unsafe for water delivery vehicles. Need closure, alternate access, and route safety assessment.",
  },
  clinic_meds: {
    tab: "shelter",
    location: "Primary health sub-center shelter",
    note: "Clinic shelter has 18 evacuees, 4 elderly residents, 3 patients needing daily medicine, and cold storage uncertain after power loss. Drinking water is low.",
  },
  evac_route: {
    tab: "damage",
    location: "Bridge route to Ward 9",
    note: "Evacuation route blocked by flood debris near bridge. Residents beyond crossing may be isolated. Need structural inspection, alternate route, and radio update.",
  },
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    panels.forEach((panel) => panel.classList.remove("active"));
    tab.classList.add("active");
    document.querySelector(`#panel-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "log") {
      loadIncidents();
    }
  });
});

forms.forEach((form) => {
  const fileInputs = form.querySelectorAll('input[type="file"]');
  const previews = form.querySelectorAll("[data-file-preview]");
  fileInputs.forEach((fileInput, index) => {
    const preview = previews[index];
    if (preview) {
      fileInput.addEventListener("change", () => renderLocalPreview(fileInput, preview));
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await analyze(form);
  });
});

document.querySelector("#refreshLog")?.addEventListener("click", loadIncidents);
document.querySelector("#refreshDashboard")?.addEventListener("click", loadDashboard);
document.querySelector("#loadScenario")?.addEventListener("click", loadSelectedScenario);
document.querySelector("#voiceButton")?.addEventListener("click", startRadioDictation);
document.querySelector("#exportSync")?.addEventListener("click", exportSyncQueue);
document.querySelector("#markSyncExported")?.addEventListener("click", markSyncExported);
document.querySelector("#refreshQueues")?.addEventListener("click", loadExportQueues);
document.querySelector("#groundingButton")?.addEventListener("click", runGroundingChallenge);
document.querySelector("#startMission")?.addEventListener("click", startGuidedMission);
document.querySelector("#refreshOfflineProof")?.addEventListener("click", loadOfflineProof);
document.querySelectorAll("[data-mission-action]").forEach((button) => {
  button.addEventListener("click", () => runMissionAction(button.dataset.missionAction));
});
groundingImage?.addEventListener("change", (event) => renderLocalPreview(event.target, groundingPreview));
videoScanForm?.addEventListener("submit", runVideoScan);
videoScanForm?.querySelector('input[type="file"]')?.addEventListener("change", (event) => renderLocalPreview(event.target, videoPreview));
window.addEventListener("hashchange", routeFromHash);

resultContent.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-sms]");
  if (!button) {
    return;
  }
  const sms = button.getAttribute("data-copy-sms");
  await navigator.clipboard.writeText(sms);
  const original = button.textContent;
  button.textContent = "Copied";
  window.setTimeout(() => {
    button.textContent = original;
  }, 1200);
});

document.querySelector("#groundingResult")?.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy-discord]");
  const openButton = event.target.closest("[data-open-discord]");
  if (!button && !openButton) {
    return;
  }
  const activeButton = button || openButton;
  const message = activeButton.getAttribute("data-copy-discord") || activeButton.getAttribute("data-open-discord");
  try {
    if (openButton) {
      await openDiscordDestination(message);
    } else {
      await navigator.clipboard.writeText(message);
    }
  } catch (error) {
    const status = document.querySelector("#discordOpenStatus");
    if (status) {
      status.textContent = error.message || "Discord handoff failed before it could be recorded.";
    }
    return;
  }
  const original = activeButton.textContent;
  activeButton.textContent = openButton ? "Queued + opened" : "Copied";
  window.setTimeout(() => {
    activeButton.textContent = original;
  }, 1200);
});

async function analyze(form) {
  const data = new FormData(form);
  data.append("scenario_type", form.dataset.scenario);
  data.append("audience", "district emergency operations center");
  data.append("role", roleSelect ? roleSelect.value : "district operations");
  data.append("sms_language", languageSelect ? languageSelect.value : "English");
  setLoading(true, form);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: data,
    });
    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.status}`);
    }
    const result = await response.json();
    renderResult(result);
    loadIncidents();
  } catch (error) {
    resultContent.className = "empty";
    resultContent.innerHTML = `
      <div class="empty-state">
        <strong>Analysis failed</strong>
        <p>${escapeHtml(error.message)}</p>
      </div>
    `;
  } finally {
    setLoading(false, form);
  }
}

function setLoading(isLoading, activeForm) {
  document.querySelectorAll(".primary").forEach((button) => {
    if (!button.dataset.original) {
      button.dataset.original = button.textContent;
    }
    const isActive = button.closest("form") === activeForm;
    button.disabled = isLoading;
    button.textContent = isLoading && isActive ? "Analyzing locally..." : button.dataset.original;
  });
}

function renderResult(result) {
  engineStatus.textContent = result.engine;
  resultMeta.textContent = `Report #${result.report_id} / ${result.location} / ${result.model}`;
  urgencyBadge.textContent = result.urgency;
  urgencyBadge.className = `urgency ${result.urgency}`;
  resultContent.className = "";
  resultContent.innerHTML = `
    <section class="section">
      <div class="result-meta-grid">
        <span class="meta-chip">Report #${escapeHtml(result.report_id)}</span>
        <span class="meta-chip">${escapeHtml(result.engine)}</span>
        <span class="meta-chip">${Math.round(result.confidence * 100)}% confidence</span>
      </div>
    </section>
    ${renderImage(result.before_image, "Before Image")}
    ${renderImage(result.image)}
    <section class="section">
      <h3>Summary</h3>
      <p>${escapeHtml(result.summary)}</p>
    </section>
    <section class="section">
      <h3>Prioritized Actions</h3>
      <div class="action-list">
        ${result.action_plan.map(renderAction).join("")}
      </div>
    </section>
    <section class="section">
      <div class="sms-top">
        <h3>SMS / Radio Update</h3>
        <button class="copy-button" type="button" data-copy-sms="${escapeHtml(result.sms_update)}">Copy SMS</button>
      </div>
      <div class="sms">${escapeHtml(result.sms_update)}</div>
      ${result.localized_sms ? `<div class="sms localized">${escapeHtml(result.localized_sms)}</div>` : ""}
    </section>
    <section class="section">
      <h3>Function Call Trace</h3>
      <div class="trace-row">${result.tool_trace.map((tool) => `<span>${escapeHtml(tool)}</span>`).join("")}</div>
    </section>
    <section class="section">
      <h3>Priority Timeline</h3>
      <div class="timeline">${result.timeline.map(renderTimelineEvent).join("")}</div>
    </section>
    <section class="section">
      <h3>Handoff Packet</h3>
      <button class="secondary" type="button" onclick="downloadHandoff(${Number(result.report_id)})">Download Packet</button>
    </section>
    <section class="section">
      <h3>Supply Request</h3>
      ${renderSupply(result.supply_request)}
    </section>
    <section class="section">
      <h3>Trust Panel</h3>
      ${renderTrust(result)}
    </section>
  `;
}

function renderImage(image, title = "Uploaded Image") {
  if (!image) {
    return "";
  }
  return `
    <section class="section">
      <h3>${escapeHtml(title)}</h3>
      <img class="upload-preview" src="${escapeHtml(image.url)}" alt="${escapeHtml(image.filename)}">
      <p>${escapeHtml(image.filename)} / ${escapeHtml(image.content_type)} / ${formatBytes(image.size_bytes)}</p>
    </section>
  `;
}

function renderTimelineEvent(event) {
  return `
    <article class="timeline-event">
      <strong>${escapeHtml(event.label)}</strong>
      <p>${escapeHtml(event.detail)}</p>
    </article>
  `;
}

function renderAction(action) {
  return `
    <article class="action ${escapeHtml(action.priority)}">
      <div class="action-top">
        <strong>${escapeHtml(action.title)}</strong>
        <span class="chip ${escapeHtml(action.priority)}">${escapeHtml(action.priority)}</span>
      </div>
      <p>${escapeHtml(action.rationale)}</p>
      <p><strong>Owner:</strong> ${escapeHtml(action.owner)}</p>
      <p><strong>Next:</strong> ${escapeHtml(action.next_step)}</p>
    </article>
  `;
}

function renderSupply(items) {
  if (!items.length) {
    return "<p>No specific supplies requested yet. Verify field conditions.</p>";
  }
  return `
    <div class="supply-grid">
      ${items.map((item) => `
        <article class="supply-card">
          <strong>${escapeHtml(item.item)}</strong>
          <p>${escapeHtml(item.quantity)}</p>
          <p>${escapeHtml(item.reason)}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function renderTrust(result) {
  return `
    <div class="trust-grid">
      <article class="trust-card">
        <h3>Extracted Facts</h3>
        ${renderList(result.extracted_facts)}
      </article>
      <article class="trust-card">
        <h3>Inferred Recommendations</h3>
        ${renderList(result.inferred_recommendations)}
      </article>
      <article class="trust-card">
        <h3>Verification Flags</h3>
        ${renderList(result.verification_flags)}
      </article>
      <article class="trust-card">
        <h3>Citations</h3>
        <div class="citation-list">
          ${result.citations.map((citation) => `
            <article class="citation">
              <strong>${escapeHtml(citation.title)}</strong>
              <p>${escapeHtml(citation.snippet)}</p>
              <p>${escapeHtml(citation.source)}</p>
            </article>
          `).join("")}
        </div>
      </article>
    </div>
  `;
}

function renderList(items) {
  if (!items.length) {
    return "<p>No items reported.</p>";
  }
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

async function loadIncidents() {
  const response = await fetch("/api/incidents");
  const incidents = await response.json();
  if (incidentCount) {
    incidentCount.textContent = String(incidents.length);
  }
  if (highCount) {
    highCount.textContent = String(incidents.filter((incident) => ["high", "critical"].includes(incident.urgency)).length);
  }
  if (!incidents.length) {
    incidentList.innerHTML = `
      <div class="empty-state">
        <strong>No incidents saved yet</strong>
        <p>Run a shelter or damage analysis to create the first local record.</p>
      </div>
    `;
    return;
  }
  incidentList.innerHTML = incidents.map(renderIncident).join("");
  await loadDashboard();
}

function renderIncident(incident) {
  const image = incident.analysis_json && incident.analysis_json.image;
  return `
    <article class="incident">
      <div class="incident-top">
        <strong>#${incident.id} / ${escapeHtml(incident.location)}</strong>
        <span class="chip ${escapeHtml(incident.urgency)}">${escapeHtml(incident.urgency)}</span>
      </div>
      <div class="incident-meta">
        <span>${escapeHtml(incident.scenario_type)}</span>
        <span>${escapeHtml(incident.status)}</span>
        <span>${escapeHtml(incident.created_at)}</span>
      </div>
      ${image ? `<img class="incident-thumb" src="${escapeHtml(image.url)}" alt="${escapeHtml(image.filename)}">` : ""}
      <p>${escapeHtml(incident.summary)}</p>
      ${renderIncidentTimeline(incident)}
      <div class="incident-actions">
        ${["new", "verified", "dispatched", "resolved"].map((status) => `<button class="secondary mini" type="button" onclick="setIncidentStatus(${incident.id}, '${status}')">${status}</button>`).join("")}
        <button class="secondary mini" type="button" onclick="downloadHandoff(${incident.id})">packet</button>
      </div>
      <p><strong>SMS:</strong> ${escapeHtml(incident.sms_update)}</p>
    </article>
  `;
}

function renderIncidentTimeline(incident) {
  const timeline = incident.analysis_json && incident.analysis_json.timeline;
  if (!timeline || !timeline.length) {
    return "";
  }
  return `<div class="timeline compact">${timeline.map(renderTimelineEvent).join("")}</div>`;
}

async function setIncidentStatus(id, status) {
  await fetch(`/api/incidents/${id}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  await loadIncidents();
}

async function loadDashboard() {
  const response = await fetch("/api/dashboard");
  const dashboard = await response.json();
  const counts = dashboard.counts || {};
  if (medicalNeedCount) medicalNeedCount.textContent = String(counts.urgent_medical || 0);
  if (waterNeedCount) waterNeedCount.textContent = String(counts.water_shortage || 0);
  if (blockedRouteCount) blockedRouteCount.textContent = String(counts.blocked_routes || 0);
  if (verificationFlagCount) verificationFlagCount.textContent = String(counts.verification_flags || 0);
  const routes = dashboard.routes || [];
  if (!routeBoard) {
    return;
  }
  routeBoard.innerHTML = routes.length
    ? routes.map((route) => `
      <article class="route-node ${escapeHtml(route.urgency)}">
        <strong>#${route.id} / ${escapeHtml(route.location)}</strong>
        <span>${escapeHtml(route.urgency)} / ${escapeHtml(route.status)}</span>
      </article>
    `).join("")
    : "<p>No blocked-route incidents yet.</p>";
}

function loadSelectedScenario() {
  const demo = DEMOS[demoScenario.value];
  if (!demo) {
    return;
  }
  activatePage("intake");
  activateTab(demo.tab);
  const form = document.querySelector(`.analysis-form[data-scenario="${demo.tab}"]`);
  form.querySelector('[name="location"]').value = demo.location;
  form.querySelector('[name="note_text"]').value = demo.note;
}

function startGuidedMission() {
  loadSelectedScenario();
  runMissionAction("intake");
}

function runMissionAction(action) {
  if (action === "intake") {
    activatePage("intake");
    activateTab("shelter");
    return;
  }
  activatePage(action);
}

function activateTab(name) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  document.querySelectorAll("#page-intake .panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));
}

function routeFromHash() {
  const page = (window.location.hash || "#intake").replace("#", "");
  activatePage(page);
}

function activatePage(page) {
  const safePage = document.querySelector(`[data-page="${page}"]`) ? page : "intake";
  appPages.forEach((item) => item.classList.toggle("active", item.dataset.page === safePage));
  pageLinks.forEach((link) => link.classList.toggle("active", link.dataset.pageLink === safePage));
  if (window.location.hash !== `#${safePage}`) {
    history.replaceState(null, "", `#${safePage}`);
  }
  if (safePage === "dashboard") {
    loadDashboard();
  }
  if (safePage === "incidents") {
    loadIncidents();
  }
  if (safePage === "exports") {
    loadExportQueues();
  }
  if (safePage === "offline") {
    loadOfflineProof();
  }
}

async function runVideoScan(event) {
  event.preventDefault();
  const data = new FormData(videoScanForm);
  const button = videoScanForm.querySelector(".primary");
  if (!data.get("video") || !data.get("video").name) {
    renderVideoError("Upload a video before running the scan.");
    return;
  }
  button.disabled = true;
  button.textContent = "Scanning video...";
  videoScanResults.className = "empty";
  videoScanResults.innerHTML = `
    <div class="empty-state">
      <strong>Sampling frames</strong>
      <p>FieldAid is running the local video scan and creating an incident.</p>
    </div>
  `;
  try {
    const response = await fetch("/api/video/scan", { method: "POST", body: data });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Video scan failed: ${response.status}`);
    }
    const result = await response.json();
    renderVideoScan(result);
    await loadIncidents();
  } catch (error) {
    renderVideoError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run Video Scan";
  }
}

function renderVideoScan(result) {
  const disaster = result.disaster_analysis || { disaster_type: "unknown", engine: "not-run" };
  videoScanMeta.textContent = `${result.sampled_frames} sampled frames / likely ${disaster.disaster_type} / incident #${result.incident.report_id}`;
  videoModelSource.textContent = result.model_source;
  videoModelSource.className = `urgency ${result.model_source === "custom" ? "low" : "medium"}`;
  videoScanResults.className = "";
  videoScanResults.innerHTML = `
    <section class="section">
      <div class="result-meta-grid">
        <span class="meta-chip">${escapeHtml(result.model_source)}</span>
        <span class="meta-chip">${escapeHtml(result.model_name)}</span>
        <span class="meta-chip">Incident #${escapeHtml(result.incident.report_id)}</span>
      </div>
    </section>
    <section class="section">
      <h3>Disaster Identification</h3>
      <div class="trust-card">
        <strong>${escapeHtml(disaster.disaster_type)}</strong>
        <p>${escapeHtml(disaster.summary || "No disaster summary available.")}</p>
        <p><strong>Analyzer:</strong> ${escapeHtml(disaster.engine)} / <strong>Confidence:</strong> ${Math.round(Number(disaster.confidence || 0) * 100)}%</p>
        <h3>Visual Evidence</h3>
        ${renderList(disaster.visual_evidence || [])}
        <h3>Response Focus</h3>
        ${renderList(disaster.recommended_focus || [])}
      </div>
    </section>
    ${renderVideoClassifierAggregation(result.classifier_aggregation)}
    <section class="section">
      <h3>Observations</h3>
      ${renderList(result.observations)}
    </section>
    <section class="section">
      <h3>Annotated Frames</h3>
      <div class="frame-gallery">
        ${result.annotated_frames.map((frame) => `
          <figure>
            <img src="${escapeHtml(frame.url)}" alt="${escapeHtml(frame.filename)}">
            <figcaption>${escapeHtml(frame.filename)} / ${formatBytes(frame.size_bytes)}</figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
    <section class="section">
      <h3>Supporting YOLO Detection Timeline</h3>
      ${renderDetectionTable(result.detections)}
    </section>
    <section class="section">
      <h3>Created Incident</h3>
      <p>${escapeHtml(result.incident.summary)}</p>
      <div class="control-actions">
        <a class="link-button" href="#incidents">Open Incident Log</a>
        <button class="secondary" type="button" onclick="downloadHandoff(${Number(result.incident.report_id)})">Download Packet</button>
      </div>
    </section>
    <section class="section">
      <h3>Tool Trace</h3>
      <div class="trace-row">${result.tool_trace.map((tool) => `<span>${escapeHtml(tool)}</span>`).join("")}</div>
    </section>
  `;
}

function renderVideoClassifierAggregation(aggregation) {
  if (!aggregation) {
    return "";
  }
  if (!aggregation.available) {
    return `
      <section class="section">
        <h3>Local Frame Classifier</h3>
        <div class="classifier-panel unavailable">
          <strong>Unavailable</strong>
          <span>${escapeHtml(aggregation.reason || "No local classifier result was available.")}</span>
          ${renderList(aggregation.observations || [])}
        </div>
      </section>
    `;
  }
  const counts = Object.entries(aggregation.label_counts || {}).map(([label, count]) => `
    <tr><td>${escapeHtml(label).replaceAll("_", " ")}</td><td>${escapeHtml(count)}</td></tr>
  `).join("");
  const frames = (aggregation.frame_results || []).map((frame) => `
    <tr>
      <td>${escapeHtml(frame.timestamp_seconds)}s</td>
      <td>${escapeHtml(frame.top_label || "unknown").replaceAll("_", " ")}</td>
      <td>${Math.round(Number(frame.top_confidence || 0) * 100)}%</td>
    </tr>
  `).join("");
  return `
    <section class="section">
      <h3>Local Frame Classifier</h3>
      <div class="classifier-panel">
        <div>
          <strong>${escapeHtml(aggregation.confidence_band)} ${escapeHtml(aggregation.top_label || "unknown").replaceAll("_", " ")}</strong>
          <span>${escapeHtml(aggregation.model)} · ${escapeHtml(aggregation.frames_classified)} frames classified</span>
        </div>
        ${renderList(aggregation.observations || [])}
        <div class="split-tables">
          <table class="mini-table">
            <thead><tr><th>Class</th><th>Frames</th></tr></thead>
            <tbody>${counts}</tbody>
          </table>
          <table class="mini-table">
            <thead><tr><th>Time</th><th>Top label</th><th>Score</th></tr></thead>
            <tbody>${frames}</tbody>
          </table>
        </div>
      </div>
    </section>
  `;
}

function renderDetectionTable(detections) {
  if (!detections.length) {
    return "<p>No object detections met threshold. Manual review is still required.</p>";
  }
  return `
    <div class="detection-table">
      <div class="table-head">Time</div>
      <div class="table-head">Class</div>
      <div class="table-head">Confidence</div>
      ${detections.map((detection) => `
        <div>${escapeHtml(detection.timestamp_seconds)}s</div>
        <div>${escapeHtml(detection.class_name)}</div>
        <div>${Math.round(Number(detection.confidence) * 100)}%</div>
      `).join("")}
    </div>
  `;
}

function renderVideoError(message) {
  videoModelSource.textContent = "Needs attention";
  videoModelSource.className = "urgency high";
  videoScanResults.className = "empty";
  videoScanResults.innerHTML = `
    <div class="empty-state">
      <strong>Video scan unavailable</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function startRadioDictation() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const activePanel = document.querySelector(".panel.active");
  const textarea = activePanel && activePanel.querySelector("textarea");
  if (!SpeechRecognition || !textarea) {
    voiceStatus.textContent = "Browser dictation is unavailable here. Paste a radio transcript into the note field.";
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.onresult = (event) => {
    textarea.value = `${textarea.value}\n${event.results[0][0].transcript}`.trim();
    voiceStatus.textContent = "Radio transcript captured locally.";
  };
  recognition.onerror = () => {
    voiceStatus.textContent = "Dictation stopped. You can still paste the radio transcript.";
  };
  recognition.start();
  voiceStatus.textContent = "Listening for radio-style intake...";
}

async function exportSyncQueue() {
  const response = await fetch("/api/sync/export");
  const payload = await response.json();
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  downloadBlob(blob, "fieldaid-sync-queue.json");
  await loadSyncQueue();
}

async function markSyncExported() {
  const proceed = window.confirm("Mark queued sync records as exported only after you have transferred or archived the JSON packet.");
  if (!proceed) {
    return;
  }
  await fetch("/api/sync/mark-exported", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: null }),
  });
  await loadExportQueues();
}

async function loadExportQueues() {
  await Promise.all([loadSyncQueue(), loadOutbox()]);
}

async function loadOfflineProof() {
  if (!offlineProofGrid) {
    return;
  }
  const [dashboardResponse, queueResponse, outboxResponse] = await Promise.all([
    fetch("/api/dashboard"),
    fetch("/api/sync/queue"),
    fetch("/api/outbox"),
  ]);
  const dashboard = await dashboardResponse.json();
  const queue = await queueResponse.json();
  const outbox = await outboxResponse.json();
  offlineProofGrid.innerHTML = `
    <article class="proof-card">
      <span>Gemma Runtime</span>
      <strong>Ollama local</strong>
      <p>Default: gemma4:e4b. Optional: fieldaid-gemma4:e4b and gemma4:26b.</p>
    </article>
    <article class="proof-card">
      <span>Visual Model</span>
      <strong>MEDIC classifier</strong>
      <p>Local checkpoint: models/fieldaid-medic-image-classifier-final/best.pt.</p>
    </article>
    <article class="proof-card">
      <span>Local Records</span>
      <strong>${escapeHtml(dashboard.total || 0)} incidents</strong>
      <p>${escapeHtml(queue.length)} sync records waiting locally and ${escapeHtml(outbox.length)} Discord/radio handoff drafts.</p>
    </article>
    <article class="proof-card">
      <span>Grounding</span>
      <strong>Bundled RAG</strong>
      <p>Guidance retrieval, citations, and safety flags run from local project assets.</p>
    </article>
  `;
}

async function loadSyncQueue() {
  if (!syncQueueList) {
    return;
  }
  syncQueueList.innerHTML = `<p class="inline-note">Loading sync queue...</p>`;
  const response = await fetch("/api/sync/queue");
  const items = await response.json();
  if (!items.length) {
    syncQueueList.innerHTML = `<p class="inline-note">No local records are waiting for sync.</p>`;
    return;
  }
  syncQueueList.innerHTML = items.map((item) => {
    const payload = item.payload_json || {};
    return `
      <article class="queue-item">
        <div>
          <strong>${escapeHtml(item.entity_type)} #${escapeHtml(item.entity_id)}</strong>
          <span>${escapeHtml(payload.location || payload.summary || "FieldAid record")}</span>
        </div>
        <span class="status-pill ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span>
      </article>
    `;
  }).join("");
}

async function loadOutbox() {
  if (!outboxList) {
    return;
  }
  outboxList.innerHTML = `<p class="inline-note">Loading handoff outbox...</p>`;
  const response = await fetch("/api/outbox");
  const items = await response.json();
  if (!items.length) {
    outboxList.innerHTML = `<p class="inline-note">No Discord handoffs have been drafted yet.</p>`;
    return;
  }
  outboxList.innerHTML = items.map((item) => `
    <article class="queue-item outbox-item">
      <div>
        <strong>${escapeHtml(item.channel)} ${item.incident_id ? `incident #${escapeHtml(item.incident_id)}` : "handoff"}</strong>
        <span>${escapeHtml(item.destination)}</span>
        <p>${escapeHtml(item.message)}</p>
      </div>
      <span class="status-pill ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span>
    </article>
  `).join("");
}

async function downloadHandoff(reportId) {
  const response = await fetch(`/api/incidents/${reportId}/handoff`);
  downloadBlob(new Blob([await response.text()], { type: "text/plain" }), `fieldaid-handoff-${reportId}.txt`);
}

async function runGroundingChallenge() {
  const question = document.querySelector("#groundingQuestion").value;
  const data = new FormData();
  data.append("question", question);
  data.append("scenario_type", "damage");
  data.append("model", document.querySelector("#groundingModel").value);
  if (groundingImage?.files?.[0]) {
    data.append("image", groundingImage.files[0]);
  }
  const response = await fetch("/api/grounding", { method: "POST", body: data });
  const result = await response.json();
  latestGroundingResult = result;
  document.querySelector("#groundingResult").innerHTML = `
    <div class="trust-card">
      <div class="result-meta-grid">
        <span class="meta-chip">${escapeHtml(result.status)}</span>
        <span class="meta-chip">${escapeHtml(result.scene_type)}</span>
        <span class="meta-chip">${Math.round(Number(result.confidence || 0) * 100)}% confidence</span>
      </div>
      ${result.image ? `<img class="upload-preview" src="${escapeHtml(result.image.url)}" alt="${escapeHtml(result.image.filename)}">` : ""}
      <h3>Status Answer</h3>
      <p>${escapeHtml(result.answer)}</p>
      ${renderClassifierPanel(result.classifier)}
      <h3>Visual Evidence</h3>
      ${renderList(result.visual_evidence || [])}
      <h3>Recommended Actions</h3>
      ${renderList(result.recommended_actions || [])}
      <h3>Discord Draft</h3>
      <div class="sms">${escapeHtml(result.discord_message)}</div>
      <div class="control-actions discord-actions">
        <button class="copy-button" type="button" data-copy-discord="${escapeHtml(result.discord_message)}">Copy Discord Message</button>
        <button class="copy-button" type="button" data-open-discord="${escapeHtml(result.discord_message)}">Prompt Discord Handoff</button>
      </div>
      <p class="inline-note" id="discordOpenStatus">Use Prompt Discord Handoff to create an outbox record, copy this draft, and open the configured Discord destination.</p>
      <h3>Trace</h3>
      <div class="trace-row">${result.tool_trace.map((tool) => `<span>${escapeHtml(tool)}</span>`).join("")}</div>
      <h3>Verification Flags</h3>
      ${renderList(result.verification_flags || [])}
      <h3>Citations</h3>
      ${result.citations.map((citation) => `<p>${escapeHtml(citation.title)}: ${escapeHtml(citation.snippet)}</p>`).join("")}
    </div>
  `;
}

function renderClassifierPanel(classifier) {
  if (!classifier) {
    return "";
  }
  if (!classifier.available) {
    return `
      <h3>Local Visual Classifier</h3>
      <div class="classifier-panel unavailable">
        <div>
          <strong>Unavailable</strong>
          <span>${escapeHtml(classifier.reason || "Classifier files or dependencies are missing.")}</span>
        </div>
      </div>
    `;
  }
  const predictions = (classifier.predictions || []).map((item) => `
    <tr>
      <td>${escapeHtml(item.label).replaceAll("_", " ")}</td>
      <td>${Math.round(Number(item.confidence || 0) * 100)}%</td>
    </tr>
  `).join("");
  return `
    <h3>Local Visual Classifier</h3>
    <div class="classifier-panel">
      <div>
        <strong>${escapeHtml(classifier.confidence_band || "unknown")} ${escapeHtml(classifier.top_label || "unknown").replaceAll("_", " ")}</strong>
        <span>${escapeHtml(classifier.model || "fieldaid classifier")} · ${Math.round(Number(classifier.validation_accuracy || 0) * 100)}% validation accuracy</span>
      </div>
      <table class="mini-table">
        <thead><tr><th>Label</th><th>Score</th></tr></thead>
        <tbody>${predictions}</tbody>
      </table>
      <p class="inline-note">Classifier output supports triage only. FieldAid never declares roads, bridges, or structures safe from imagery.</p>
    </div>
  `;
}

async function openDiscordDestination(message) {
  const rawDestination = (discordDestination?.value || "").trim();
  if (!rawDestination) {
    const status = document.querySelector("#discordOpenStatus");
    if (status) {
      status.textContent = "Add a Discord destination URL or ID above before creating an auditable handoff.";
    }
    return;
  }
  const proceed = window.confirm(
    "FieldAid will create an outbox record, copy the draft to your clipboard, and open Discord. Review and send it manually."
  );
  if (!proceed) {
    return;
  }
  const url = normalizeDiscordDestination(rawDestination);
  const outboxRecord = await createDiscordOutboxRecord(rawDestination, message);
  await navigator.clipboard.writeText(message);
  await updateOutboxStatus(outboxRecord.id, "copied");
  window.open(url, "_blank", "noopener");
  await updateOutboxStatus(outboxRecord.id, "opened");
  const status = document.querySelector("#discordOpenStatus");
  if (status) {
    status.textContent = `Outbox #${outboxRecord.id} recorded. Message copied; Discord opened: ${url}`;
  }
  await loadOutbox();
}

async function createDiscordOutboxRecord(destination, message) {
  const response = await fetch("/api/outbox", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      channel: "discord",
      destination,
      message,
      metadata_json: {
        source: "grounding_challenge",
        status: latestGroundingResult?.status || "unknown",
        scene_type: latestGroundingResult?.scene_type || "unknown",
        confidence: latestGroundingResult?.confidence || 0,
        human_send_required: true,
      },
    }),
  });
  if (!response.ok) {
    throw new Error("Discord outbox record could not be created.");
  }
  return response.json();
}

async function updateOutboxStatus(id, status) {
  await fetch(`/api/outbox/${id}/status`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

function normalizeDiscordDestination(destination) {
  if (destination.startsWith("https://") || destination.startsWith("http://") || destination.startsWith("discord://")) {
    return destination;
  }
  if (/^\d+$/.test(destination)) {
    return `https://discord.com/users/${destination}`;
  }
  return `https://discord.com/channels/@me`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function renderLocalPreview(input, preview) {
  const file = input.files && input.files[0];
  if (!file) {
    preview.classList.add("hidden");
    preview.innerHTML = "";
    return;
  }
  const url = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  preview.innerHTML = `
    ${file.type.startsWith("video/")
      ? `<video src="${url}" controls muted></video>`
      : `<img src="${url}" alt="${escapeHtml(file.name)}">`}
    <div>
      <strong>${escapeHtml(file.name)}</strong>
      <span>${escapeHtml(file.type || "image file")} / ${formatBytes(file.size)}</span>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

routeFromHash();
loadIncidents();

/* ===== Walkthrough Controller ===== */

const WT = {
  steps: [],
  current: -1,
  loading: false,
  started: false,

  els: {
    start: null,
    nav: null,
    prev: null,
    next: null,
    counter: null,
    end: null,
  },

  init() {
    this.els.start = document.getElementById("walkthroughStart");
    this.els.nav = document.getElementById("walkthroughNav");
    this.els.prev = document.getElementById("walkthroughPrev");
    this.els.next = document.getElementById("walkthroughNext");
    this.els.counter = document.getElementById("walkthroughCounter");
    this.els.end = document.getElementById("walkthroughEnd");

    if (this.els.start) {
      this.els.start.addEventListener("click", (e) => {
        e.preventDefault();
        this.start();
      });
    }
    if (this.els.next) {
      this.els.next.addEventListener("click", () => this.next());
    }
    if (this.els.prev) {
      this.els.prev.addEventListener("click", () => this.prev());
    }
    if (this.els.end) {
      this.els.end.addEventListener("click", () => this.end());
    }
  },

  async start() {
    try {
      const res = await fetch("/api/walkthrough/steps");
      this.steps = await res.json();
    } catch {
      alert("Could not load walkthrough steps.");
      return;
    }

    this.started = true;
    this.current = -1;
    document.body.classList.add("walkthrough-active");
    this.els.start.classList.add("hidden");
    this.els.nav.classList.remove("hidden");
    this.els.end.classList.remove("hidden");
    this.updateUI();
  },

  end() {
    this.started = false;
    this.current = -1;
    document.body.classList.remove("walkthrough-active");
    this.els.start.classList.remove("hidden");
    this.els.nav.classList.add("hidden");
    this.els.end.classList.add("hidden");
    this.els.next.disabled = false;
    this.els.next.classList.remove("loading", "done");
    this.els.next.textContent = "Next \u25B6";
  },

  next() {
    if (this.loading) return;
    if (this.steps.length && this.current < this.steps.length - 1) {
      this.current++;
      this.executeStep();
    } else {
      this.end();
    }
  },

  prev() {
    if (this.loading || this.current <= 0) return;
    this.current--;
    this.navigateAndPrefill();
  },

  async executeStep() {
    const step = this.steps[this.current];
    if (!step) return;

    this.navigateAndPrefill();
    this.els.next.disabled = true;
    this.els.next.classList.remove("done");
    this.els.next.classList.add("loading");
    this.els.next.textContent = "Running...";

    await this.wait(1200);

    try {
      if (step.button.includes("analyzeShelter") || step.button.includes("assessDamage")) {
        await this.submitAnalyze(step);
      } else if (step.button.includes("runVideoScan")) {
        await this.submitVideo(step);
      } else if (step.button.includes("groundingButton")) {
        await this.submitGrounding(step);
      } else if (step.button.includes("refreshDashboard")) {
        await this.apiClick("#refreshDashboard");
        await loadDashboard();
        await this.wait(1500);
      } else if (step.button.includes("refreshLog")) {
        await this.apiClick("#refreshLog");
        await loadIncidents();
        await this.wait(1500);
      } else if (step.button.includes("exportSync")) {
        await this.apiClick("#exportSync");
        await this.wait(1500);
      } else if (step.button.includes("markSyncExported")) {
        await this.markSyncExportedQuietly();
        await this.wait(1500);
      }

      this.els.next.classList.remove("loading");
      this.els.next.classList.add("done");
      this.els.next.textContent = "\u2714 Done";
      this.els.next.disabled = false;
    } catch (err) {
      this.els.next.classList.remove("loading");
      this.els.next.disabled = false;
      this.els.next.textContent = "Next \u25B6";
      console.error("Walkthrough step failed:", err);
    }

    this.updateUI();
  },

  navigateAndPrefill() {
    const step = this.steps[this.current];
    if (!step) return;

    window.location.hash = step.page;

    if (step.tab) {
      const tab = document.querySelector(`.tab[data-tab="${step.tab}"]`);
      if (tab) tab.click();
    }

    const analysisForm = step.tab ? document.querySelector(`#panel-${step.tab} .analysis-form`) : null;

    if (step.note) {
      const textarea = analysisForm?.querySelector('[name="note_text"]');
      if (textarea) {
        textarea.value = step.note;
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }

    if (step.location) {
      const locInput = analysisForm?.querySelector('[name="location"]');
      if (locInput) {
        locInput.value = step.location;
        locInput.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }

    if (step.model) {
      const modelSelect = analysisForm?.querySelector('[name="model"]');
      if (modelSelect) {
        modelSelect.value = step.model;
        modelSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }

    if (step.sms_language) {
      const langSelect = document.getElementById("languageSelect");
      if (langSelect) {
        langSelect.value = step.sms_language;
        langSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }

    if (step.role) {
      const roleSelect = document.getElementById("roleSelect");
      if (roleSelect) {
        roleSelect.value = step.role;
        roleSelect.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }

    if (step.question) {
      const qInput = document.getElementById("groundingQuestion");
      if (qInput) {
        qInput.value = step.question;
        setTimeout(() => qInput.dispatchEvent(new Event("input", { bubbles: true })), 100);
      }
    }

    if (step.video_type && step.location) {
      const vLoc = document.querySelector('#videoScanForm [name="location"]');
      if (vLoc) {
        vLoc.value = step.location;
        vLoc.dispatchEvent(new Event("input", { bubbles: true }));
      }
      const vModel = document.querySelector('#videoScanForm [name="model"]');
      if (vModel) {
        vModel.value = step.model || "gemma4:e4b";
        vModel.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }

    this.loadMedia(step);
    this.updateUI();
  },

  loadMedia(step) {
    if (!step.media) return;

    let fileInput = null;
    if (step.video_type) {
      fileInput = document.querySelector('#videoScanForm input[type="file"]');
    } else if (step.question) {
      fileInput = document.querySelector('#groundingImage');
    } else if (step.tab) {
      fileInput = document.querySelector(`#panel-${step.tab} .analysis-form input[type="file"]`);
    }

    if (!fileInput) return;

    fetch(`/wt-media/${step.media}`)
      .then((r) => {
        if (!r.ok) {
          throw new Error(`Media ${step.media} failed to load (${r.status})`);
        }
        return r.blob();
      })
      .then((blob) => {
        const file = new File([blob], step.media, { type: blob.type });
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        fileInput.dispatchEvent(new Event("change", { bubbles: true }));
      })
      .catch((e) => console.error("Media load failed:", e));
  },

  async submitAnalyze(step) {
    const fd = new FormData();
    fd.append("scenario_type", step.tab || "shelter");
    fd.append("note_text", step.note || "");
    fd.append("location", step.location || "Unknown");
    fd.append("model", step.model || "gemma4:e4b");
    fd.append("sms_language", step.sms_language || "English");
    fd.append("role", step.role || "district operations");
    fd.append("audience", "district emergency operations center");

    if (step.media && step.media.endsWith(".jpg")) {
      const blob = await this.fetchMediaBlob(step.media);
      fd.append("image", new File([blob], step.media), step.media);
    }

    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const result = await res.json();
    renderResult(result);
    loadIncidents();
  },

  async submitVideo(step) {
    const fd = new FormData();
    if (step.media) {
      const blob = await this.fetchMediaBlob(step.media);
      fd.append("video", new File([blob], step.media), step.media);
    }
    fd.append("location", step.location || "Unknown");
    fd.append("sample_interval_seconds", "2");
    fd.append("model", step.model || "gemma4:e4b");

    const res = await fetch("/api/video/scan", { method: "POST", body: fd });
    const result = await res.json();
    renderVideoScan(result);
    loadIncidents();
  },

  async submitGrounding(step) {
    const fd = new FormData();
    fd.append("question", step.question || "");
    fd.append("scenario_type", "damage");
    fd.append("model", step.model || "gemma4:e4b");

    if (step.media) {
      const blob = await this.fetchMediaBlob(step.media);
      fd.append("image", new File([blob], step.media), step.media);
    }

    const res = await fetch("/api/grounding", { method: "POST", body: fd });
    const result = await res.json();
    latestGroundingResult = result;
    const groundingResult = document.querySelector("#groundingResult");
    if (groundingResult) {
      groundingResult.innerHTML = `
        <div class="trust-card">
          <div class="result-meta-grid">
            <span class="meta-chip">${escapeHtml(result.status)}</span>
            <span class="meta-chip">${escapeHtml(result.scene_type)}</span>
            <span class="meta-chip">${Math.round(Number(result.confidence || 0) * 100)}% confidence</span>
          </div>
          ${result.image ? `<img class="upload-preview" src="${escapeHtml(result.image.url)}" alt="${escapeHtml(result.image.filename)}">` : ""}
          <h3>Status Answer</h3>
          <p>${escapeHtml(result.answer)}</p>
          ${renderClassifierPanel(result.classifier)}
          <h3>Visual Evidence</h3>
          ${renderList(result.visual_evidence || [])}
          <h3>Recommended Actions</h3>
          ${renderList(result.recommended_actions || [])}
          <h3>Discord Draft</h3>
          <div class="sms">${escapeHtml(result.discord_message)}</div>
          <h3>Trace</h3>
          <div class="trace-row">${result.tool_trace.map((tool) => `<span>${escapeHtml(tool)}</span>`).join("")}</div>
          <h3>Verification Flags</h3>
          ${renderList(result.verification_flags || [])}
          <h3>Citations</h3>
          ${result.citations.map((citation) => `<p>${escapeHtml(citation.title)}: ${escapeHtml(citation.snippet)}</p>`).join("")}
        </div>
      `;
    }
  },

  async apiClick(selector) {
    const el = document.querySelector(selector);
    if (el) el.click();
    await this.wait(500);
  },

  async fetchMediaBlob(filename) {
    const res = await fetch(`/wt-media/${filename}`);
    if (!res.ok) {
      throw new Error(`Media ${filename} failed to load (${res.status})`);
    }
    return res.blob();
  },

  async markSyncExportedQuietly() {
    await fetch("/api/sync/mark-exported", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: null }),
    });
    await loadExportQueues();
  },

  updateUI() {
    if (!this.steps.length) return;
    if (this.current < 0) {
      this.els.counter.textContent = `Ready / ${this.steps.length} steps`;
      this.els.prev.disabled = true;
      this.els.next.disabled = false;
      this.els.next.classList.remove("loading", "done");
      this.els.next.textContent = "Next \u25B6";
      return;
    }
    this.els.counter.textContent = `Step ${this.current + 1} / ${this.steps.length}: ${this.steps[this.current].title}`;
    this.els.prev.disabled = this.current <= 0;

    if (this.current >= this.steps.length - 1) {
      this.els.next.textContent = "Finish";
    } else if (!this.els.next.classList.contains("done") && !this.els.next.classList.contains("loading")) {
      this.els.next.textContent = "Next \u25B6";
    }
  },

  wait(ms) {
    return new Promise((r) => setTimeout(r, ms));
  },
};

WT.init();
