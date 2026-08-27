/* ============================================================
   VISIONGUARD AI - MODERN FRONTEND
   Keeps the existing Flask /analyze API and Firebase collection.
   ============================================================ */
"use strict";

const $ = (selector) => document.querySelector(selector);

const fileInput = $("#file");
const preview = $("#preview");
const previewInfo = $("#previewInfo");
const dropArea = $("#dropArea");
const analyzeBtn = $("#analyze");
const clearBtn = $("#clearBtn");
const reportBtn = $("#reportBtn");

const patientName = $("#personName");
const patientAge = $("#personAge");
const patientGender = $("#gender");
const patientIdInput = $("#patientId");
const screeningIdInput = $("#screeningId");
const screeningBadge = $("#screeningBadge");

const qscore = $("#qscore");
const qmsg = $("#qmsg");
const qualityRing = $("#qualityRing");
const mode = $("#mode");

const resultImg = $("#resultImg");
const empty = $("#empty");
const heat = $("#heat");
const pred = $("#pred");
const conf = $("#conf");
const prog = $("#prog");
const risk = $("#risk");
const explain = $("#explain");
const lesions = $("#lesions");
const lang = $("#lang");
const plain = $("#plain");

const currentPatient = $("#currentPatient");
const currentAge = $("#currentAge");
const currentGender = $("#currentGender");
const currentPatientId = $("#currentPatientId");
const currentScreeningId = $("#currentScreeningId");
const currentResult = $("#currentResult");
const currentConfidence = $("#currentConfidence");
const currentDate = $("#currentDate");
const historySummary = $("#history");

const pages = {
  new: $("#newPage"),
  history: $("#historyPage"),
  reports: $("#reportsPage"),
  settings: $("#settingsPage")
};

const navButtons = {
  new: $("#newScreeningNav"),
  history: $("#historyNav"),
  reports: $("#reportsNav"),
  settings: $("#settingsNav")
};

const historyCards = $("#historyCards");
const reportCards = $("#reportCards");
const historySearch = $("#historySearch");
const historyFrom = $("#historyFrom");
const historyTo = $("#historyTo");
const historyCount = $("#historyCount");
const recordModal = $("#recordModal");
const modalTitle = $("#modalTitle");
const modalBody = $("#modalBody");
const modalDownload = $("#modalDownload");
const modalPrint = $("#modalPrint");

let selectedFile = null;
let lastResult = null;
let allRecords = [];
let activeRecord = null;

function textValue(value, fallback = "Not provided") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayClassName(value) {
  return String(value || "Unknown").replaceAll("_", " ");
}

function absoluteURL(url) {
  if (!url || url === "Not available") return "";
  try {
    return new URL(url, window.location.origin).href;
  } catch {
    return url;
  }
}

function makePatientId() {
  let number = Number(localStorage.getItem("visionguardPatientCounter") || "0");
  number += 1;
  localStorage.setItem("visionguardPatientCounter", String(number));
  return "P-" + String(number).padStart(4, "0");
}

function setScreeningId(value) {
  const id = String(value || "").trim();
  if (screeningIdInput) screeningIdInput.value = id || "Generated after analysis";
  if (screeningBadge) screeningBadge.textContent = id ? id : "SCR —";
  if (currentScreeningId && id) currentScreeningId.textContent = id;
}

function resetScreeningId() {
  setScreeningId("");
}

function setButtonLoading(loading) {
  if (!analyzeBtn) return;
  if (loading) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing with CNN...";
  } else {
    analyzeBtn.disabled = !selectedFile;
    analyzeBtn.textContent = "Analyze retinal image →";
  }
}

function setImage(url, element) {
  if (!element) return;
  if (!url) {
    element.removeAttribute("src");
    element.style.display = "none";
    return;
  }
  element.src = url;
  element.style.display = "block";
}

function parseRecordDate(record) {
  const created = record?.createdAt;
  const timestamp = record?.timestamp;

  if (created) {
    const d = new Date(created);
    if (!Number.isNaN(d.getTime())) return d;
  }

  if (timestamp) {
    const d = new Date(timestamp);
    if (!Number.isNaN(d.getTime())) return d;
  }

  return null;
}

function formatDate(record) {
  const d = parseRecordDate(record);
  if (!d) return textValue(record?.timestamp, "Date unavailable");
  return d.toLocaleString([], {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function formatDateOnly(date) {
  if (!date) return "—";
  return date.toLocaleDateString([], {
    day: "2-digit",
    month: "short",
    year: "numeric"
  });
}

function resultInitial(name) {
  const clean = String(name || "P").trim();
  return clean.charAt(0).toUpperCase() || "P";
}

function riskClass(record) {
  const riskText = String(record?.risk || "").toLowerCase();
  if (riskText.includes("low")) return "low";
  if (riskText.includes("severe") || riskText.includes("advanced")) return "high";
  return "medium";
}

async function waitForFirebase(timeoutMs = 8000) {
  const started = Date.now();

  while (!window.firebaseReady) {
    if (Date.now() - started > timeoutMs) {
      throw new Error("Firebase is not ready. Please refresh the page and try again.");
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  if (!window.firebaseDB || !window.firebaseCollection ||
      !window.firebaseAddDoc || !window.firebaseGetDocs) {
    throw new Error("Firebase Firestore functions are unavailable.");
  }
}

/* ---------------- NAVIGATION ---------------- */

function activatePage(name) {
  Object.entries(pages).forEach(([key, page]) => {
    if (page) page.classList.toggle("active-page", key === name);
  });

  Object.entries(navButtons).forEach(([key, button]) => {
    if (button) button.classList.toggle("active", key === name);
  });

  window.scrollTo({ top: 0, behavior: "smooth" });

  if (name === "history") {
    refreshHistoryTable();
  }

  if (name === "reports") {
    renderReports();
  }

  if (name === "settings") {
    checkSystemStatus();
  }
}

if (navButtons.new) navButtons.new.addEventListener("click", () => activatePage("new"));
if (navButtons.history) navButtons.history.addEventListener("click", () => activatePage("history"));
if (navButtons.reports) navButtons.reports.addEventListener("click", () => activatePage("reports"));
if (navButtons.settings) navButtons.settings.addEventListener("click", () => activatePage("settings"));

/* ---------------- IMAGE INPUT ---------------- */

function handleSelectedFile(file) {
  if (!file) return;

  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    alert("Please select a JPG, PNG or WEBP retinal image.");
    return;
  }

  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > 10) {
    alert("Image is too large. Maximum size is 10 MB.");
    return;
  }

  selectedFile = file;

  const objectURL = URL.createObjectURL(file);
  setImage(objectURL, preview);

  if (previewInfo) {
    previewInfo.textContent =
      `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  }

  if (analyzeBtn) analyzeBtn.disabled = false;

  if (qscore) qscore.textContent = "—";
  if (qmsg) qmsg.textContent = "Image selected. Ready for AI screening.";
  if (mode) mode.textContent = "Ready for analysis";
  if (qualityRing) qualityRing.style.background =
    "conic-gradient(var(--teal) 0deg,#dceae8 0deg)";
}

if (fileInput) {
  fileInput.addEventListener("change", () => handleSelectedFile(fileInput.files?.[0]));
}

if (dropArea) {
  ["dragenter", "dragover"].forEach(eventName => {
    dropArea.addEventListener(eventName, event => {
      event.preventDefault();
      dropArea.classList.add("drag");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, event => {
      event.preventDefault();
      dropArea.classList.remove("drag");
    });
  });

  dropArea.addEventListener("drop", event => {
    handleSelectedFile(event.dataTransfer.files?.[0]);
  });
}

/* ---------------- FIREBASE SAVE ---------------- */

async function saveScreeningToFirebase(data) {
  try {
    await waitForFirebase();

    const patient = data.patient || {};
    const quality = data.quality || {};
    const gradcam = data.gradcam || {};

    const record = {
      screeningId: textValue(data.screeningId),
      patientId: textValue(patient.id),
      patientName: textValue(patient.name),
      age: textValue(patient.age),
      gender: textValue(patient.gender, ""),
      result: textValue(data.class_, "Unknown"),
      confidence: Number(data.confidence || 0),
      risk: textValue(data.risk),
      explanation: textValue(data.explanation),
      findings: Array.isArray(data.lesions) ? data.lesions : [],
      quality: Number(quality.score || 0),
      qualityStatus: textValue(quality.status, "unknown"),
      qualityMessage: textValue(quality.message),
      imageUrl: textValue(data.image_url, "Not available"),
      gradcamUrl: textValue(data.gradcam_url, "Not available"),
      gradcamStatus: textValue(gradcam.status, "unavailable"),
      gradcamMessage: textValue(gradcam.message),
      mode: textValue(data.mode, "AI MODEL"),
      timestamp: textValue(data.timestamp, new Date().toLocaleString()),
      createdAt: new Date().toISOString()
    };

    const ref = await window.firebaseAddDoc(
      window.firebaseCollection(window.firebaseDB, "screenings"),
      record
    );

    console.log("Screening saved to Firebase:", ref.id);
    return ref.id;
  } catch (error) {
    console.error("Could not save screening to Firebase:", error);
    return null;
  }
}

/* ---------------- HISTORY LOAD + FILTER ---------------- */

async function loadAllHistory() {
  await waitForFirebase();

  const snapshot = await window.firebaseGetDocs(
    window.firebaseCollection(window.firebaseDB, "screenings")
  );

  const records = [];
  snapshot.forEach(docSnap => {
    records.push({
      firestoreId: docSnap.id,
      ...docSnap.data()
    });
  });

  records.sort((a, b) => {
    const aTime = parseRecordDate(a)?.getTime() || 0;
    const bTime = parseRecordDate(b)?.getTime() || 0;
    return bTime - aTime;
  });

  allRecords = records;
  return records;
}

function filteredRecords() {
  const query = String(historySearch?.value || "").trim().toLowerCase();
  const from = historyFrom?.value ? new Date(historyFrom.value + "T00:00:00") : null;
  const to = historyTo?.value ? new Date(historyTo.value + "T23:59:59.999") : null;

  return allRecords.filter(record => {
    const searchable = [
      record.patientName,
      record.patientId,
      record.screeningId,
      record.gender,
      record.result
    ].map(value => String(value || "").toLowerCase()).join(" ");

    if (query && !searchable.includes(query)) return false;

    const date = parseRecordDate(record);
    if (from && (!date || date < from)) return false;
    if (to && (!date || date > to)) return false;

    return true;
  });
}

function renderHistoryCards() {
  if (!historyCards) return;

  const records = filteredRecords();

  if (historyCount) {
    historyCount.textContent =
      `${records.length} screening${records.length === 1 ? "" : "s"}`;
  }

  if (!records.length) {
    historyCards.innerHTML = `
      <div class="empty-card">
        <strong>No screenings found</strong>
        <div style="margin-top:7px">Try a different patient name, ID or date range.</div>
      </div>`;
    return;
  }

  historyCards.innerHTML = records.map((record, index) => {
    const patient = textValue(record.patientName || record.name, "Patient");
    const patientId = textValue(record.patientId, "No patient ID");
    const screeningId = textValue(record.screeningId, "No screening ID");
    const result = displayClassName(record.result || record.class_);
    const confidence = Number(record.confidence || 0).toFixed(1) + "%";
    const quality = Number(record.quality || 0) + "%";
    const risk = textValue(record.risk);
    const riskType = riskClass(record);

    return `
      <article class="history-card">
        <div class="card-top">
          <div style="display:flex;gap:11px;align-items:center">
            <div class="avatar">${escapeHTML(resultInitial(patient))}</div>
            <div>
              <div class="patient-name">${escapeHTML(patient)}</div>
              <div class="patient-id">${escapeHTML(patientId)}</div>
            </div>
          </div>
          <span class="result-pill">${escapeHTML(result)}</span>
        </div>

        <div class="history-details">
          <div><small>Confidence</small><strong>${escapeHTML(confidence)}</strong></div>
          <div><small>Quality</small><strong>${escapeHTML(quality)}</strong></div>
          <div><small>Risk</small><strong class="${riskType}">${escapeHTML(risk)}</strong></div>
        </div>

        <div class="card-date">${escapeHTML(formatDate(record))}</div>
        <div class="patient-id" style="margin-bottom:12px">Screening ID: ${escapeHTML(screeningId)}</div>

        <div class="card-actions">
          <button class="secondary" data-view-record="${escapeHTML(record.firestoreId)}">View details</button>
          <button class="secondary" data-download-record="${escapeHTML(record.firestoreId)}">Report</button>
        </div>
      </article>`;
  }).join("");
}

async function refreshHistoryTable() {
  if (historyCards) {
    historyCards.innerHTML = `<div class="empty-card">Loading screening history...</div>`;
  }

  try {
    await loadAllHistory();
    renderHistoryCards();
    renderReports();
  } catch (error) {
    console.error("History load error:", error);
    if (historyCards) {
      historyCards.innerHTML = `
        <div class="empty-card">
          <strong>Could not load screening history.</strong>
          <div style="margin-top:7px">Check Firebase/Firestore permissions and connection.</div>
        </div>`;
    }
    if (reportCards) {
      reportCards.innerHTML = `<div class="empty-card">Reports unavailable until Firebase is connected.</div>`;
    }
  }
}

[historySearch, historyFrom, historyTo].forEach(input => {
  if (input) input.addEventListener("input", renderHistoryCards);
});

$("#clearFilters")?.addEventListener("click", () => {
  if (historySearch) historySearch.value = "";
  if (historyFrom) historyFrom.value = "";
  if (historyTo) historyTo.value = "";
  renderHistoryCards();
});

$("#refreshHistory")?.addEventListener("click", refreshHistoryTable);

/* ---------------- REPORTS ---------------- */

function reportCard(record) {
  const patient = textValue(record.patientName || record.name, "Patient");
  const result = displayClassName(record.result || record.class_);
  const confidence = Number(record.confidence || 0).toFixed(1) + "%";

  return `
    <article class="report-card">
      <div class="card-top">
        <div>
          <h3>${escapeHTML(patient)}</h3>
          <div class="report-sub">${escapeHTML(textValue(record.screeningId))}</div>
        </div>
        <span class="result-pill">${escapeHTML(result)}</span>
      </div>

      <div class="report-preview">
        <div>Patient ID<strong>${escapeHTML(textValue(record.patientId))}</strong></div>
        <div>Confidence<strong>${escapeHTML(confidence)}</strong></div>
        <div>Quality<strong>${escapeHTML(Number(record.quality || 0) + "%")}</strong></div>
        <div>Date<strong>${escapeHTML(formatDate(record))}</strong></div>
      </div>

      <div class="card-date" style="margin-top:12px">${escapeHTML(textValue(record.risk))}</div>

      <div class="card-actions">
        <button class="secondary" data-view-report="${escapeHTML(record.firestoreId)}">View report</button>
        <button class="secondary" data-download-report="${escapeHTML(record.firestoreId)}">Download</button>
      </div>
    </article>`;
}

function renderReports() {
  if (!reportCards) return;

  if (!allRecords.length) {
    reportCards.innerHTML = `
      <div class="empty-card">
        <strong>No saved reports yet.</strong>
        <div style="margin-top:7px">Complete a screening and the report will appear here.</div>
      </div>`;
    return;
  }

  reportCards.innerHTML = allRecords.map(reportCard).join("");
}

$("#refreshReports")?.addEventListener("click", refreshHistoryTable);

/* ---------------- MODAL / DETAILS ---------------- */

function findRecord(id) {
  return allRecords.find(record => record.firestoreId === id);
}

function recordToResult(record) {
  return {
    screeningId: record.screeningId,
    patient: {
      id: record.patientId || "",
      name: record.patientName || record.name || "",
      age: record.age || "",
      gender: record.gender || ""
    },
    class_: record.result || record.class_,
    confidence: Number(record.confidence || 0),
    risk: record.risk,
    explanation: record.explanation,
    lesions: Array.isArray(record.findings) ? record.findings : [],
    quality: {
      score: Number(record.quality || 0),
      status: record.qualityStatus || "",
      message: record.qualityMessage || ""
    },
    image_url: record.imageUrl,
    gradcam_url: record.gradcamUrl,
    gradcam: {
      status: record.gradcamStatus,
      message: record.gradcamMessage
    },
    mode: record.mode || "AI MODEL",
    timestamp: record.timestamp || record.createdAt
  };
}

function showRecordModal(record) {
  if (!record || !recordModal) return;

  activeRecord = record;

  const patient = recordToResult(record);
  const name = textValue(patient.patient.name, "Patient");
  const result = displayClassName(patient.class_);
  const findings = patient.lesions.length
    ? patient.lesions.map(item => `<span class="finding">${escapeHTML(item)}</span>`).join("")
    : `<span style="color:var(--muted);font-size:12px">No specific findings recorded.</span>`;

  const imageURL = absoluteURL(patient.image_url);
  const gradcamURL = absoluteURL(patient.gradcam_url);

  modalTitle.textContent = `${name} • ${textValue(record.screeningId)}`;

  modalBody.innerHTML = `
    <div class="detail-grid">
      <div class="detail-box"><small>Patient ID</small><strong>${escapeHTML(patient.patient.id)}</strong></div>
      <div class="detail-box"><small>Age</small><strong>${escapeHTML(patient.patient.age)}</strong></div>
      <div class="detail-box"><small>Gender</small><strong>${escapeHTML(patient.patient.gender)}</strong></div>
      <div class="detail-box"><small>Date</small><strong>${escapeHTML(formatDate(record))}</strong></div>
      <div class="detail-box"><small>Category</small><strong>${escapeHTML(result)}</strong></div>
      <div class="detail-box"><small>Confidence</small><strong>${escapeHTML(patient.confidence + "%")}</strong></div>
      <div class="detail-box"><small>Image Quality</small><strong>${escapeHTML(patient.quality.score + "%")}</strong></div>
      <div class="detail-box"><small>Mode</small><strong>${escapeHTML(patient.mode)}</strong></div>
    </div>

    <div class="modal-section">
      <h3>Recommended action</h3>
      <p><strong>${escapeHTML(textValue(patient.risk))}</strong></p>
    </div>

    <div class="modal-section">
      <h3>AI explanation</h3>
      <p>${escapeHTML(textValue(patient.explanation))}</p>
    </div>

    <div class="modal-section">
      <h3>Potential visual findings</h3>
      <div>${findings}</div>
    </div>

    ${(imageURL || gradcamURL) ? `
      <div class="modal-section">
        <h3>Explainability images</h3>
        <div class="modal-images">
          ${imageURL ? `<img src="${escapeHTML(imageURL)}" alt="Original retinal image">` : ""}
          ${gradcamURL ? `<img src="${escapeHTML(gradcamURL)}" alt="Grad-CAM attention map">` : ""}
        </div>
      </div>` : ""}

    <div class="notice">
      <strong>Important:</strong> This is an AI-based screening result and is not a medical diagnosis. The result should be reviewed by a qualified eye-care professional.
    </div>
  `;

  recordModal.classList.add("open");
  recordModal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  if (!recordModal) return;
  recordModal.classList.remove("open");
  recordModal.setAttribute("aria-hidden", "true");
  activeRecord = null;
}

document.addEventListener("click", event => {
  const close = event.target.closest("[data-close-modal]");
  if (close) {
    closeModal();
    return;
  }

  const viewId = event.target.closest("[data-view-record]")?.dataset.viewRecord;
  if (viewId) {
    const record = findRecord(viewId);
    if (record) showRecordModal(record);
    return;
  }

  const downloadHistoryId = event.target.closest("[data-download-record]")?.dataset.downloadRecord;
  if (downloadHistoryId) {
    const record = findRecord(downloadHistoryId);
    if (record) downloadReport(record);
    return;
  }

  const viewReportId = event.target.closest("[data-view-report]")?.dataset.viewReport;
  if (viewReportId) {
    const record = findRecord(viewReportId);
    if (record) showRecordModal(record);
    return;
  }

  const downloadReportId = event.target.closest("[data-download-report]")?.dataset.downloadReport;
  if (downloadReportId) {
    const record = findRecord(downloadReportId);
    if (record) downloadReport(record);
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeModal();
});

modalDownload?.addEventListener("click", () => {
  if (activeRecord) downloadReport(activeRecord);
});

modalPrint?.addEventListener("click", () => {
  if (!activeRecord) return;
  const html = buildReportHTML(activeRecord);
  const printWindow = window.open("", "_blank", "width=1000,height=800");
  if (!printWindow) {
    alert("Please allow pop-ups to print the report.");
    return;
  }
  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => printWindow.print(), 300);
});

/* ---------------- REPORT GENERATION ---------------- */

function buildReportHTML(record) {
  const data = recordToResult(record);
  const patient = data.patient || {};
  const name = textValue(patient.name);
  const patientId = textValue(patient.id);
  const screeningId = textValue(data.screeningId);
  const category = displayClassName(data.class_);
  const confidence = Number(data.confidence || 0);
  const quality = Number(data.quality?.score || 0);
  const riskText = textValue(data.risk);
  const explanation = textValue(data.explanation);
  const findings = Array.isArray(data.lesions) && data.lesions.length
    ? data.lesions.map(item => `<li>${escapeHTML(item)}</li>`).join("")
    : "<li>No specific findings recorded.</li>";
  const imageURL = absoluteURL(data.image_url);
  const gradcamURL = absoluteURL(data.gradcam_url);
  const timestamp = textValue(data.timestamp);

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisionGuard AI - Screening Report</title>
<style>
body{margin:0;padding:30px;background:#f4f7f7;color:#1e2b2b;font-family:Arial,sans-serif}
.container{max-width:900px;margin:auto;background:#fff;padding:32px;border-radius:18px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
.header{display:flex;gap:18px;align-items:center;border-bottom:2px solid #dceeed;padding-bottom:18px;margin-bottom:22px}
.logo{width:76px;height:76px;border-radius:16px;background:#fff;display:grid;place-items:center;overflow:hidden;flex-shrink:0}.logo img{width:100%;height:100%;object-fit:contain;display:block}
h1{margin:0;color:#166f6b}h2{color:#166f6b;margin-bottom:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{border:1px solid #e1e7e7;padding:18px;border-radius:12px;margin-top:18px}
.label{font-size:11px;color:#738080;text-transform:uppercase}.value{font-size:16px;font-weight:bold;margin-top:5px}
img{max-width:100%;border-radius:12px;margin-top:10px}
.note{background:#fff7df;padding:15px;border-radius:12px;margin-top:22px;font-size:13px;line-height:1.5}
.footer{text-align:center;margin-top:30px;color:#7a8585;font-size:11px}
li{margin:6px 0}
@media print{body{background:#fff;padding:0}.container{box-shadow:none}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<div class="logo"><img src="${window.location.origin}/static/visionguard-logo.jpeg" alt="VisionGuard AI Logo"></div>
<div><h1>VISIONGUARD AI</h1><div>Explainable Retinal Screening</div><p>Screening ID: <strong>${escapeHTML(screeningId)}</strong></p></div>
</div>

<div class="card">
<h2>Patient Information</h2>
<div class="grid">
<div><div class="label">Patient ID</div><div class="value">${escapeHTML(patientId)}</div></div>
<div><div class="label">Patient Name</div><div class="value">${escapeHTML(name)}</div></div>
<div><div class="label">Age</div><div class="value">${escapeHTML(textValue(patient.age))}</div></div>
<div><div class="label">Gender</div><div class="value">${escapeHTML(textValue(patient.gender))}</div></div>
</div>
</div>

<div class="card">
<h2>AI Screening Result</h2>
<div class="grid">
<div><div class="label">Category</div><div class="value">${escapeHTML(category)}</div></div>
<div><div class="label">Confidence</div><div class="value">${confidence}%</div></div>
<div><div class="label">Image Quality</div><div class="value">${quality}%</div></div>
<div><div class="label">Recommended Action</div><div class="value">${escapeHTML(riskText)}</div></div>
</div>
<h3>Explanation</h3><p>${escapeHTML(explanation)}</p>
<h3>Potential Visual Findings</h3><ul>${findings}</ul>
<h3>Date & Time</h3><p>${escapeHTML(timestamp)}</p>
</div>

${imageURL ? `<div class="card"><h2>Original Retinal Image</h2><img src="${escapeHTML(imageURL)}" alt="Original retinal image"></div>` : ""}
${gradcamURL ? `<div class="card"><h2>Grad-CAM Explainability</h2><p>The Grad-CAM image is an explainability aid and not proof of disease.</p><img src="${escapeHTML(gradcamURL)}" alt="Grad-CAM attention map"></div>` : ""}

<div class="note"><strong>Important:</strong><br>This is an AI-based screening result and is not a medical diagnosis. The result should be reviewed by a qualified eye-care professional.</div>
<div class="footer">VisionGuard AI • Explainable AI • Retinal Screening</div>
</div>
</body>
</html>`;
}

function downloadReport(record) {
  const html = buildReportHTML(record);
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const safeId = String(record.screeningId || "screening").replace(/[^a-zA-Z0-9_-]/g, "_");

  const link = document.createElement("a");
  link.href = url;
  link.download = `VisionGuard_${safeId}_Report.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

if (reportBtn) {
  reportBtn.addEventListener("click", () => {
    if (!lastResult) {
      alert("Please analyze a retinal image first.");
      return;
    }
    const tempRecord = {
      firestoreId: "current",
      screeningId: lastResult.screeningId,
      patientId: lastResult.patient?.id,
      patientName: lastResult.patient?.name,
      age: lastResult.patient?.age,
      gender: lastResult.patient?.gender,
      result: lastResult.class_,
      confidence: lastResult.confidence,
      risk: lastResult.risk,
      explanation: lastResult.explanation,
      findings: lastResult.lesions,
      quality: lastResult.quality?.score,
      qualityStatus: lastResult.quality?.status,
      qualityMessage: lastResult.quality?.message,
      imageUrl: lastResult.image_url,
      gradcamUrl: lastResult.gradcam_url,
      gradcamStatus: lastResult.gradcam?.status,
      gradcamMessage: lastResult.gradcam?.message,
      mode: lastResult.mode,
      timestamp: lastResult.timestamp,
      createdAt: new Date().toISOString()
    };
    downloadReport(tempRecord);
  });
}

/* ---------------- ANALYSIS ---------------- */

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("Please select a retinal image first.");
      return;
    }

    const name = String(patientName?.value || "").trim();
    const age = String(patientAge?.value || "").trim();
    const gender = String(patientGender?.value || "").trim();

    if (!name) {
      alert("Please enter the patient's name.");
      patientName?.focus();
      return;
    }

    if (!age) {
      alert("Please enter the patient's age.");
      patientAge?.focus();
      return;
    }

    if (!window.__visionguardCurrentPatientId) {
      window.__visionguardCurrentPatientId = makePatientId();
    }

    const patientId = window.__visionguardCurrentPatientId;

    if (patientIdInput) patientIdInput.value = patientId;
    if (currentPatientId) currentPatientId.textContent = patientId;

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("patientName", name);
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("patientId", patientId);

    setButtonLoading(true);
    if (mode) mode.textContent = "Running CNN + Grad-CAM";
    if (qmsg) qmsg.textContent = "Checking image quality and analyzing retinal features...";

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData
      });

      let data = {};
      try {
        data = await response.json();
      } catch {
        throw new Error(`Server returned an invalid response (${response.status}).`);
      }

      if (!response.ok) {
        throw new Error(data.error || data.details || `Analysis failed (${response.status}).`);
      }

      lastResult = data;

      setScreeningId(data.screeningId);
      if (patientIdInput) patientIdInput.value = data.patient?.id || patientId;

      if (pred) pred.textContent = displayClassName(data.class_);
      const confidence = Number(data.confidence || 0);
      if (conf) conf.textContent = confidence + "%";
      if (prog) prog.style.width = Math.min(100, Math.max(0, confidence)) + "%";

      if (risk) risk.textContent = data.risk || "Further evaluation recommended.";
      if (explain) explain.textContent = data.explanation || "AI analysis completed.";

      if (qscore) qscore.textContent = Number(data.quality?.score || 0) + "%";
      if (qmsg) qmsg.textContent = data.quality?.message || "Quality check completed.";

      const q = Number(data.quality?.score || 0);
      if (qualityRing) {
        qualityRing.style.background =
          `conic-gradient(var(--teal) ${q * 3.6}deg,#dceae8 ${q * 3.6}deg)`;
      }

      if (mode) mode.textContent = data.mode || "AI MODEL";

      if (currentPatient) currentPatient.textContent = textValue(data.patient?.name);
      if (currentAge) currentAge.textContent = textValue(data.patient?.age);
      if (currentGender) currentGender.textContent = textValue(data.patient?.gender);
      if (currentPatientId) currentPatientId.textContent = textValue(data.patient?.id);
      if (currentScreeningId) currentScreeningId.textContent = textValue(data.screeningId);
      if (currentResult) currentResult.textContent = displayClassName(data.class_);
      if (currentConfidence) currentConfidence.textContent = confidence + "%";
      if (currentDate) currentDate.textContent = textValue(data.timestamp);

      if (historySummary) {
        historySummary.textContent =
          `Latest screening: ${textValue(data.patient?.name, "Patient")} — ${displayClassName(data.class_)} — ${textValue(data.timestamp)}`;
      }

      if (resultImg) {
        setImage(absoluteURL(data.image_url), resultImg);
      }
      if (empty) empty.style.display = "none";

      if (heat) {
        heat.innerHTML = "";
        if (data.gradcam_url) {
          const wrapper = document.createElement("div");
          wrapper.innerHTML = `
            <strong style="display:block;color:#d9eeeb;margin-bottom:8px">Grad-CAM attention map</strong>
            <img src="${escapeHTML(absoluteURL(data.gradcam_url))}" alt="Grad-CAM attention map">
          `;
          heat.appendChild(wrapper);
        }
      }

      if (lesions) {
        const findings = Array.isArray(data.lesions) ? data.lesions : [];
        lesions.innerHTML = findings.length
          ? findings.map(item => `<span class="finding">${escapeHTML(item)}</span>`).join("")
          : `<span style="color:var(--muted);font-size:12px">No specific findings recorded.</span>`;
      }

      updateLanguage(data);

      const firebaseId = await saveScreeningToFirebase(data);
      if (firebaseId) {
        await refreshHistoryTable();
      } else {
        console.warn("Screening completed, but Firebase save failed.");
      }

      alert(
        "Screening completed successfully.\n\nScreening ID: " +
        textValue(data.screeningId, "Not available")
      );

    } catch (error) {
      console.error("Analysis error:", error);
      alert(error.message || "Something went wrong during analysis.");
    } finally {
      setButtonLoading(false);
    }
  });
}

/* ---------------- LANGUAGE ---------------- */

function updateLanguage(data) {
  if (!plain) return;

  const result = displayClassName(data?.class_);

  if (!lang) {
    plain.textContent =
      `Screening result: ${result}. Further evaluation by an eye-care professional is recommended.`;
    return;
  }

  const selected = String(lang.value || "");

  if (selected.includes("Kannada")) {
    plain.textContent =
      `ಸ್ಕ್ರೀನಿಂಗ್ ಫಲಿತಾಂಶ: ${result}. ಹೆಚ್ಚಿನ ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ನೇತ್ರ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಲು ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ.`;
  } else if (selected.includes("Hindi")) {
    plain.textContent =
      `स्क्रीनिंग परिणाम: ${result}. आगे के मूल्यांकन के लिए नेत्र विशेषज्ञ से सलाह लेने की अनुशंसा की जाती है।`;
  } else {
    plain.textContent =
      `Screening result: ${result}. Further evaluation by an eye-care professional is recommended.`;
  }
}

lang?.addEventListener("change", () => {
  if (lastResult) updateLanguage(lastResult);
});

/* ---------------- NEW SCREENING ---------------- */

function startNewScreening() {
  if (fileInput) fileInput.value = "";
  selectedFile = null;
  lastResult = null;
  window.__visionguardCurrentPatientId = "";

  if (patientName) patientName.value = "";
  if (patientAge) patientAge.value = "";
  if (patientGender) patientGender.value = "";
  if (patientIdInput) patientIdInput.value = "Generated on analysis";

  resetScreeningId();
  setImage("", preview);

  if (previewInfo) previewInfo.textContent = "No image selected";
  if (analyzeBtn) analyzeBtn.disabled = true;

  if (qscore) qscore.textContent = "—";
  if (qmsg) qmsg.textContent = "Upload an image to run the quality gate.";
  if (mode) mode.textContent = "Waiting for image";
  if (qualityRing) qualityRing.style.background =
    "conic-gradient(var(--teal) 0deg,#dceae8 0deg)";

  if (pred) pred.textContent = "—";
  if (conf) conf.textContent = "—";
  if (prog) prog.style.width = "0%";
  if (risk) risk.textContent = "Awaiting analysis";
  if (explain) explain.textContent = "Upload a fundus image to begin.";
  if (lesions) lesions.textContent = "—";
  if (resultImg) setImage("", resultImg);
  if (empty) empty.style.display = "block";
  if (heat) heat.innerHTML = "";
  if (plain) plain.textContent = "The system will explain the result in simple language after analysis.";

  if (currentPatient) currentPatient.textContent = "—";
  if (currentAge) currentAge.textContent = "—";
  if (currentGender) currentGender.textContent = "—";
  if (currentPatientId) currentPatientId.textContent = "—";
  if (currentScreeningId) currentScreeningId.textContent = "—";
  if (currentResult) currentResult.textContent = "—";
  if (currentConfidence) currentConfidence.textContent = "—";
  if (currentDate) currentDate.textContent = "—";
  if (historySummary) historySummary.textContent = "No screening yet.";

  activatePage("new");
}

clearBtn?.addEventListener("click", startNewScreening);

/* ---------------- SETTINGS ---------------- */

async function checkSystemStatus() {
  const firebaseStatus = $("#firebaseStatus");
  const backendStatus = $("#backendStatus");

  if (firebaseStatus) {
    firebaseStatus.textContent = window.firebaseReady ? "CONNECTED" : "OFFLINE";
  }

  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    if (backendStatus) {
      backendStatus.textContent =
        data.status === "online" ? "ONLINE" : "MODEL ERROR";
    }
  } catch {
    if (backendStatus) backendStatus.textContent = "OFFLINE";
  }
}

$("#resetPatientCounter")?.addEventListener("click", () => {
  if (!confirm("Reset the local patient ID counter? The next generated ID will start from P-0001.")) return;
  localStorage.removeItem("visionguardPatientCounter");
  alert("Patient ID counter reset.");
});

/* ---------------- STARTUP ---------------- */

function initializeUI() {
  resetScreeningId();
  if (patientIdInput) patientIdInput.value = "Generated on analysis";
  if (analyzeBtn) analyzeBtn.disabled = true;
  if (preview) preview.style.display = "none";
  if (resultImg) resultImg.style.display = "none";
  if (empty) empty.style.display = "block";

  console.log("VisionGuard AI modern frontend loaded.");
  refreshHistoryTable();
  checkSystemStatus();
}

initializeUI();
