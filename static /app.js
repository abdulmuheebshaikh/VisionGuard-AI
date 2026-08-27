// ============================================================
// VISIONGUARD AI - APP.JS
// Complete frontend for the current index.html + app.py
//
// IMPORTANT FIX:
// - NO screening ID is generated when the page loads.
// - Refreshing the page does NOT create/increment a screening.
// - A screening ID comes from Flask only after /analyze succeeds.
// - Firebase is written only after a real screening is completed.
// ============================================================

"use strict";

// ------------------------------------------------------------
// MAIN ELEMENTS
// ------------------------------------------------------------
const fileInput = document.querySelector("#file");
const preview = document.querySelector("#preview");
const analyzeBtn = document.querySelector("#analyze");
const clearBtn = document.querySelector("#clearBtn");
const reportBtn = document.querySelector("#reportBtn");

const qscore = document.querySelector("#qscore");
const qmsg = document.querySelector("#qmsg");
const mode = document.querySelector("#mode");

const resultImg = document.querySelector("#resultImg");
const empty = document.querySelector("#empty");
const heat = document.querySelector("#heat");

const pred = document.querySelector("#pred");
const conf = document.querySelector("#conf");
const prog = document.querySelector("#prog");

const risk = document.querySelector("#risk");
const explain = document.querySelector("#explain");
const lesions = document.querySelector("#lesions");
const historySummary = document.querySelector("#history");

const lang = document.querySelector("#lang");
const plain = document.querySelector("#plain");
const previewInfo = document.querySelector("#previewInfo");

// ------------------------------------------------------------
// CURRENT SCREENING
// ------------------------------------------------------------
const currentPatient = document.querySelector("#currentPatient");
const currentAge = document.querySelector("#currentAge");
const currentGender = document.querySelector("#currentGender");
const currentPatientIdEl = document.querySelector("#currentPatientId");
const currentResult = document.querySelector("#currentResult");
const currentConfidence = document.querySelector("#currentConfidence");
const currentDate = document.querySelector("#currentDate");

// ------------------------------------------------------------
// PATIENT
// ------------------------------------------------------------
const patientName = document.querySelector("#personName");
const patientAge = document.querySelector("#personAge");
const patientGender = document.querySelector("#gender");

// Patient ID and Screening ID are displayed separately.
// Both are generated/filled only when an analysis is completed.
// ------------------------------------------------------------
const patientIdDisplay = document.querySelector("#PatientId");
const screeningIdDisplay = document.querySelector("#ScreeningId");
const screeningBadge = document.querySelector("#screeningBadge");

// ------------------------------------------------------------
// NAVIGATION
// ------------------------------------------------------------
const newScreeningNav = document.querySelector("#newScreeningNav");
const historyNav = document.querySelector("#historyNav");
const newScreeningSection = document.querySelector("#newScreeningSection");
const screeningHistorySection = document.querySelector("#screeningHistorySection");
const refreshHistory = document.querySelector("#refreshHistory");
const historyTable = document.querySelector("#historyTable");

// ------------------------------------------------------------
// STATE
// ------------------------------------------------------------
let selectedFile = null;
let lastResult = null;
let currentPatientId = "";
let currentScreeningId = "";
let firebaseSavePromise = null;

// ------------------------------------------------------------
// HELPERS
// ------------------------------------------------------------
function safeValue(element) {
    if (!element) return "";
    return String(element.value || "").trim();
}

function textValue(value, fallback = "Not provided") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

function displayClassName(value) {
    return String(value || "Unknown").replaceAll("_", " ");
}

function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function makePatientId() {
    // Patient IDs are generated ONLY when an actual screening is analyzed.
    // Refreshing the page does not consume an ID.
    let number = Number(
        localStorage.getItem("visionguardPatientCounter") || "0"
    );

    number += 1;

    localStorage.setItem(
        "visionguardPatientCounter",
        String(number)
    );

    return "P-" + String(number).padStart(4, "0");
}

function setPatientId(value) {
    currentPatientId = String(value || "").trim();

    if (patientIdDisplay) {
        patientIdDisplay.value =
            currentPatientId || "";
        patientIdDisplay.placeholder =
            currentPatientId ? "" : "Generated after analysis";
    }
}

function setScreeningId(value) {
    currentScreeningId = String(value || "").trim();

    if (screeningIdDisplay) {
        screeningIdDisplay.value = currentScreeningId;
        screeningIdDisplay.placeholder =
            currentScreeningId ? "" : "Generated after analysis";
    }

    if (screeningBadge) {
        screeningBadge.textContent =
            currentScreeningId || "SCR —";
    }
}

function resetScreeningId() {
    currentScreeningId = "";

    if (screeningIdDisplay) {
        screeningIdDisplay.value = "";
        screeningIdDisplay.placeholder = "Generated after analysis";
    }

    if (screeningBadge) {
        screeningBadge.textContent = "SCR —";
    }
}

function resetPatientId() {
    currentPatientId = "";

    if (patientIdDisplay) {
        patientIdDisplay.value = "";
        patientIdDisplay.placeholder = "Generated after analysis";
    }
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

function setImage(url, imageElement) {
    if (!imageElement) return;

    if (!url) {
        imageElement.removeAttribute("src");
        imageElement.style.display = "none";
        return;
    }

    imageElement.src = url;
    imageElement.style.display = "block";
}

function absoluteURL(url) {
    if (!url) return "";
    try {
        return new URL(url, window.location.origin).href;
    } catch {
        return url;
    }
}

// ------------------------------------------------------------
// FIREBASE WAIT
// ------------------------------------------------------------
async function waitForFirebase(timeoutMs = 8000) {
    const started = Date.now();

    while (!window.firebaseReady) {
        if (Date.now() - started > timeoutMs) {
            throw new Error(
                "Firebase is not ready. Please refresh the page and try again."
            );
        }

        await new Promise(resolve => setTimeout(resolve, 100));
    }

    if (!window.firebaseDB ||
        !window.firebaseCollection ||
        !window.firebaseAddDoc ||
        !window.firebaseGetDocs) {
        throw new Error(
            "Firebase Firestore functions are unavailable."
        );
    }
}

// ------------------------------------------------------------
// FIREBASE: SAVE SCREENING
// ------------------------------------------------------------
async function saveScreeningToFirebase(data) {
    try {
        await waitForFirebase();

        const db = window.firebaseDB;
        const collection = window.firebaseCollection;
        const addDoc = window.firebaseAddDoc;

        const patient = data.patient || {};
        const quality = data.quality || {};
        const gradcam = data.gradcam || {};

        const record = {
            screeningId: textValue(data.screeningId, "Not provided"),
            patientId: textValue(patient.id, "Not provided"),
            patientName: textValue(patient.name, "Not provided"),
            age: textValue(patient.age, "Not provided"),
            gender: textValue(patient.gender, "Not provided"),

            result: textValue(data.class_, "Unknown"),
            confidence: Number(data.confidence || 0),

            risk: textValue(data.risk, "Not provided"),
            explanation: textValue(data.explanation, "Not provided"),

            findings: Array.isArray(data.lesions)
                ? data.lesions
                : [],

            quality: Number(quality.score || 0),
            qualityStatus: textValue(
                quality.status,
                "unknown"
            ),
            qualityMessage: textValue(
                quality.message,
                "Not provided"
            ),

            imageUrl: textValue(
                data.image_url,
                "Not available"
            ),

            gradcamUrl: gradcam.generated
                ? textValue(
                    data.gradcam_url,
                    "Not available"
                )
                : textValue(
                    data.gradcam_url,
                    "Not available"
                ),

            gradcamStatus: textValue(
                gradcam.status,
                "unavailable"
            ),

            gradcamMessage: textValue(
                gradcam.message,
                "Not available"
            ),

            mode: textValue(
                data.mode,
                "AI MODEL"
            ),

            timestamp: textValue(
                data.timestamp,
                new Date().toLocaleString()
            ),

            // ISO value makes client-side sorting reliable.
            createdAt: new Date().toISOString()
        };

        const ref = await addDoc(
            collection(db, "screenings"),
            record
        );

        console.log(
            "✅ Screening saved to Firebase:",
            ref.id
        );

        return ref.id;

    } catch (error) {
        console.error(
            "❌ Could not save screening to Firebase:",
            error
        );

        // Do not hide the AI result if Firebase is temporarily unavailable.
        return null;
    }
}

// ------------------------------------------------------------
// FIREBASE: LOAD HISTORY
// ------------------------------------------------------------
async function loadScreeningHistory() {
    if (!historyTable) return;

    historyTable.innerHTML = `
        <tr>
            <td colspan="6">Loading screening history...</td>
        </tr>
    `;

    try {
        await waitForFirebase();

        const db = window.firebaseDB;
        const collection = window.firebaseCollection;
        const getDocs = window.firebaseGetDocs;

        const snapshot = await getDocs(
            collection(db, "screenings")
        );

        const records = [];

        snapshot.forEach(docSnap => {
            records.push({
                firestoreId: docSnap.id,
                ...docSnap.data()
            });
        });

        // Newest first. This works even with old records.
        records.sort((a, b) => {
            const aTime = Date.parse(
                a.createdAt || a.timestamp || ""
            ) || 0;

            const bTime = Date.parse(
                b.createdAt || b.timestamp || ""
            ) || 0;

            return bTime - aTime;
        });

        if (records.length === 0) {
            historyTable.innerHTML = `
                <tr>
                    <td colspan="6">No previous screenings.</td>
                </tr>
            `;
            return;
        }

        historyTable.innerHTML = records.map(
            record => createHistoryRow(record)
        ).join("");

        // Make each row open its full screening details.
        historyTable
            .querySelectorAll("[data-history-index]")
            .forEach(row => {
                row.addEventListener("click", () => {
                    const index = Number(
                        row.dataset.historyIndex
                    );

                    if (records[index]) {
                        showHistoryRecord(records[index]);
                    }
                });
            });

    } catch (error) {
        console.error(
            "History load error:",
            error
        );

        historyTable.innerHTML = `
            <tr>
                <td colspan="6">
                    Could not load Firebase history.
                    Check Firestore permissions.
                </td>
            </tr>
        `;
    }
}

// ------------------------------------------------------------
// HISTORY ROW
// ------------------------------------------------------------
function createHistoryRow(record) {
    const patient = textValue(
        record.patientName || record.name,
        "Not provided"
    );

    const age = textValue(
        record.age,
        "Not provided"
    );

    const result = displayClassName(
        record.result || record.class_
    );

    const confidence =
        Number(record.confidence || 0).toFixed(2) + "%";

    const risk = textValue(
        record.risk,
        "Not provided"
    );

    const date = textValue(
        record.timestamp || record.createdAt,
        "Not available"
    );

    const screeningId = textValue(
        record.screeningId,
        "Not provided"
    );

    return `
        <tr
            data-history-index="${createHistoryRow.indexCounter ?? 0}"
            style="cursor:pointer"
            title="Click to view this screening"
        >
            <td>
                <strong>${escapeHTML(patient)}</strong>
                <small style="
                    display:block;
                    margin-top:4px;
                    opacity:.65;
                    font-size:10px;
                ">
                    ${escapeHTML(screeningId)}
                </small>
            </td>
            <td>${escapeHTML(age)}</td>
            <td>
                <strong>${escapeHTML(result)}</strong>
            </td>
            <td>${escapeHTML(confidence)}</td>
            <td>${escapeHTML(risk)}</td>
            <td>${escapeHTML(date)}</td>
        </tr>
    `;
}

// ------------------------------------------------------------
// HISTORY ROW INDEX FIX
// ------------------------------------------------------------
// Rebuild rows with a reliable index without changing HTML.
function renderHistory(records) {
    if (!historyTable) return;

    historyTable.innerHTML = records.map(
        (record, index) => {
            const patient = textValue(
                record.patientName || record.name,
                "Not provided"
            );

            const age = textValue(
                record.age,
                "Not provided"
            );

            const result = displayClassName(
                record.result || record.class_
            );

            const confidence =
                Number(record.confidence || 0).toFixed(2) + "%";

            const risk = textValue(
                record.risk,
                "Not provided"
            );

            const date = textValue(
                record.timestamp || record.createdAt,
                "Not available"
            );

            const screeningId = textValue(
                record.screeningId,
                "Not provided"
            );

            return `
                <tr
                    data-history-index="${index}"
                    style="cursor:pointer"
                    title="Click to view this screening"
                >
                    <td>
                        <strong>${escapeHTML(patient)}</strong>
                        <small style="
                            display:block;
                            margin-top:4px;
                            opacity:.65;
                            font-size:10px;
                        ">
                            ${escapeHTML(screeningId)}
                        </small>
                    </td>
                    <td>${escapeHTML(age)}</td>
                    <td><strong>${escapeHTML(result)}</strong></td>
                    <td>${escapeHTML(confidence)}</td>
                    <td>${escapeHTML(risk)}</td>
                    <td>${escapeHTML(date)}</td>
                </tr>
            `;
        }
    ).join("");

    historyTable
        .querySelectorAll("[data-history-index]")
        .forEach(row => {
            row.addEventListener("click", () => {
                const index = Number(
                    row.dataset.historyIndex
                );

                if (records[index]) {
                    showHistoryRecord(records[index]);
                }
            });
        });
}

// Replace the earlier simple loader with the indexed renderer.
async function refreshHistoryTable() {
    if (!historyTable) return;

    historyTable.innerHTML = `
        <tr>
            <td colspan="6">Loading screening history...</td>
        </tr>
    `;

    try {
        await waitForFirebase();

        const snapshot = await window.firebaseGetDocs(
            window.firebaseCollection(
                window.firebaseDB,
                "screenings"
            )
        );

        const records = [];

        snapshot.forEach(docSnap => {
            records.push({
                firestoreId: docSnap.id,
                ...docSnap.data()
            });
        });

        records.sort((a, b) => {
            const aTime =
                Date.parse(
                    a.createdAt || a.timestamp || ""
                ) || 0;

            const bTime =
                Date.parse(
                    b.createdAt || b.timestamp || ""
                ) || 0;

            return bTime - aTime;
        });

        if (!records.length) {
            historyTable.innerHTML = `
                <tr>
                    <td colspan="6">No previous screenings.</td>
                </tr>
            `;
            return;
        }

        renderHistory(records);

    } catch (error) {
        console.error("History error:", error);

        historyTable.innerHTML = `
            <tr>
                <td colspan="6">
                    Unable to load history.
                    Check Firebase/Firestore permissions.
                </td>
            </tr>
        `;
    }
}

// ------------------------------------------------------------
// SHOW HISTORY RECORD
// ------------------------------------------------------------
function showHistoryRecord(record) {
    const patient = {
        id: record.patientId || "",
        name: record.patientName || record.name || "",
        age: record.age || "",
        gender: record.gender || ""
    };

    currentScreeningId =
        record.screeningId || "";

    setPatientId(
        patient.id || ""
    );

    setScreeningId(
        record.screeningId || ""
    );

    if (currentPatient) {
        currentPatient.textContent =
            textValue(patient.name);
    }

    if (currentAge) {
        currentAge.textContent =
            textValue(patient.age);
    }

    if (currentGender) {
        currentGender.textContent =
            textValue(patient.gender);
    }

    if (currentPatientIdEl) {
        currentPatientIdEl.textContent =
            textValue(patient.id);
    }

    if (currentResult) {
        currentResult.textContent =
            displayClassName(record.result);
    }

    if (currentConfidence) {
        currentConfidence.textContent =
            Number(record.confidence || 0) + "%";
    }

    if (currentDate) {
        currentDate.textContent =
            textValue(
                record.timestamp ||
                record.createdAt
            );
    }

    if (pred) {
        pred.textContent =
            displayClassName(record.result);
    }

    if (conf) {
        conf.textContent =
            Number(record.confidence || 0) + "%";
    }

    if (prog) {
        prog.style.width =
            Math.min(
                100,
                Math.max(
                    0,
                    Number(record.confidence || 0)
                )
            ) + "%";
    }

    if (risk) {
        risk.textContent =
            textValue(record.risk);
    }

    if (explain) {
        explain.textContent =
            textValue(record.explanation);
    }

    if (qscore) {
        qscore.textContent =
            Number(record.quality || 0) + "%";
    }

    if (qmsg) {
        qmsg.textContent =
            textValue(record.qualityMessage);
    }

    if (mode) {
        mode.textContent =
            textValue(record.mode, "AI MODEL");
    }

    if (lesions) {
        lesions.innerHTML = "";

        const findings =
            Array.isArray(record.findings)
                ? record.findings
                : [];

        if (findings.length) {
            findings.forEach(item => {
                const span =
                    document.createElement("span");

                span.textContent = item;
                span.style.display = "inline-block";
                span.style.padding = "6px 8px";
                span.style.margin = "7px 5px 0 0";
                span.style.borderRadius = "7px";
                span.style.fontSize = "10px";
                span.style.background = "#edf7f5";
                span.style.color = "#166f6b";

                lesions.appendChild(span);
            });
        } else {
            lesions.textContent =
                "No specific findings recorded.";
        }
    }

    if (resultImg) {
        setImage(
            absoluteURL(record.imageUrl),
            resultImg
        );
    }

    if (empty) {
        empty.style.display =
            record.imageUrl ? "none" : "block";
    }

    // Show Grad-CAM if the current HTML has a heat container.
    if (heat) {
        heat.innerHTML = "";

        if (record.gradcamUrl) {
            const img =
                document.createElement("img");

            img.src =
                absoluteURL(record.gradcamUrl);

            img.alt =
                "Grad-CAM attention map";

            img.style.width = "100%";
            img.style.maxWidth = "100%";
            img.style.borderRadius = "12px";
            img.style.marginTop = "12px";

            heat.appendChild(img);
        }
    }

    if (historySummary) {
        historySummary.textContent =
            "Viewing saved screening " +
            (record.screeningId || "—") +
            " for " +
            textValue(patient.name, "Patient") +
            ".";
    }

    if (screeningHistorySection) {
        screeningHistorySection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    lastResult = {
        screeningId: record.screeningId,
        patient,
        class_: record.result,
        confidence: Number(record.confidence || 0),
        risk: record.risk,
        explanation: record.explanation,
        lesions: record.findings || [],
        quality: {
            score: record.quality || 0,
            message: record.qualityMessage || "",
            status: record.qualityStatus || ""
        },
        image_url: record.imageUrl,
        gradcam_url: record.gradcamUrl,
        gradcam: {
            status: record.gradcamStatus,
            message: record.gradcamMessage
        },
        timestamp: record.timestamp
    };

    updateLanguage(lastResult);
}

// ------------------------------------------------------------
// IMAGE SELECTION
// ------------------------------------------------------------
if (fileInput) {
    fileInput.addEventListener("change", () => {
        if (!fileInput.files ||
            fileInput.files.length === 0) {

            selectedFile = null;

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }

            setImage("", preview);

            if (previewInfo) {
                previewInfo.textContent =
                    "No image selected";
            }

            return;
        }

        const file =
            fileInput.files[0];

        const sizeMB =
            file.size / (1024 * 1024);

        if (sizeMB > 10) {
            alert(
                "Image is too large. Maximum size is 10 MB."
            );

            fileInput.value = "";
            selectedFile = null;

            setImage("", preview);

            if (previewInfo) {
                previewInfo.textContent =
                    "No image selected";
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }

            return;
        }

        const allowed = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ];

        if (!allowed.includes(file.type)) {
            alert(
                "Please select a JPG, PNG or WEBP image."
            );

            fileInput.value = "";
            selectedFile = null;

            setImage("", preview);

            if (previewInfo) {
                previewInfo.textContent =
                    "No image selected";
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }

            return;
        }

        selectedFile = file;

        if (preview) {
            const oldURL =
                preview.dataset.objectUrl;

            if (oldURL) {
                URL.revokeObjectURL(oldURL);
            }

            const objectURL =
                URL.createObjectURL(file);

            preview.dataset.objectUrl =
                objectURL;

            preview.src = objectURL;
            preview.style.display = "block";
        }

        if (previewInfo) {
            previewInfo.textContent =
                "✔ Image selected: " +
                file.name;
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }

        if (qscore) {
            qscore.textContent = "—";
        }

        if (qmsg) {
            qmsg.textContent =
                "Image selected. Ready for AI screening.";
        }

        if (mode) {
            mode.textContent =
                "Ready for analysis";
        }

        if (empty) {
            empty.style.display = "block";
        }

        console.log(
            "VisionGuard selected image:",
            file.name
        );
    });
}

// ------------------------------------------------------------
// DRAG + DROP SUPPORT
// ------------------------------------------------------------
const dropArea = document.querySelector("#dropArea");

function processSelectedFile(file) {
    if (!file) return;

    const sizeMB = file.size / (1024 * 1024);
    const allowed = ["image/jpeg", "image/png", "image/webp"];

    if (sizeMB > 10) {
        alert("Image is too large. Maximum size is 10 MB.");
        return;
    }

    // Some browsers may not populate MIME type reliably, so also check extension.
    const extension = String(file.name || "").toLowerCase().split(".").pop();
    if (!allowed.includes(file.type) && !["jpg", "jpeg", "png", "webp"].includes(extension)) {
        alert("Please select a JPG, PNG or WEBP image.");
        return;
    }

    selectedFile = file;

    if (preview) {
        const oldURL = preview.dataset.objectUrl;
        if (oldURL) URL.revokeObjectURL(oldURL);

        const objectURL = URL.createObjectURL(file);
        preview.dataset.objectUrl = objectURL;
        preview.src = objectURL;
        preview.style.display = "block";
    }

    if (previewInfo) {
        previewInfo.textContent = "✔ Image selected: " + file.name;
    }

    if (analyzeBtn) analyzeBtn.disabled = false;
    if (qscore) qscore.textContent = "—";
    if (qmsg) qmsg.textContent = "Image selected. Ready for AI screening.";
    if (mode) mode.textContent = "Ready for analysis";
    if (empty) empty.style.display = "block";
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
        const file = event.dataTransfer?.files?.[0];
        if (!file) return;
        processSelectedFile(file);
    });
}

// ------------------------------------------------------------
// ANALYZE
// ------------------------------------------------------------
if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {

        if (!selectedFile) {
            alert(
                "Please select a retinal image first."
            );
            return;
        }

        const name =
            safeValue(patientName);

        const age =
            safeValue(patientAge);

        const gender =
            safeValue(patientGender);

        if (!name) {
            alert(
                "Please enter the patient's name."
            );

            patientName?.focus();
            return;
        }

        if (!age) {
            alert(
                "Please enter the patient's age."
            );

            patientAge?.focus();
            return;
        }

        // Generate patient ID ONLY now.
        // It is NOT generated on page refresh.
        if (!window.__visionguardCurrentPatientId) {
            window.__visionguardCurrentPatientId =
                makePatientId();
        }

        const patientId =
            window.__visionguardCurrentPatientId;

        const formData =
            new FormData();

        formData.append(
            "image",
            selectedFile
        );

        formData.append(
            "patientName",
            name
        );

        formData.append(
            "age",
            age
        );

        formData.append(
            "gender",
            gender
        );

        formData.append(
            "patientId",
            patientId
        );

        setButtonLoading(true);

        if (mode) {
            mode.textContent =
                "Running CNN + Grad-CAM";
        }

        if (qmsg) {
            qmsg.textContent =
                "Checking image quality and analyzing retinal features...";
        }

        try {
            const response =
                await fetch(
                    "/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const rawText =
                await response.text();

            let data;

            try {
                data =
                    JSON.parse(rawText);
            } catch {
                throw new Error(
                    "Server returned an invalid response. Check the Flask terminal."
                );
            }

            console.log(
                "VisionGuard server response:",
                data
            );

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Analysis failed."
                );
            }

            lastResult = data;

            // ----------------------------------------------------
            // IMPORTANT:
            // Screening ID comes ONLY from Flask after analysis.
            // ----------------------------------------------------
            setPatientId(
                data.patient?.id || patientId
            );

            setScreeningId(
                data.screeningId
            );

            // ----------------------------------------------------
            // QUALITY
            // ----------------------------------------------------
            if (qscore && data.quality) {
                qscore.textContent =
                    Number(
                        data.quality.score || 0
                    ) + "%";
            }

            if (qmsg && data.quality) {
                qmsg.textContent =
                    data.quality.message ||
                    "Quality check completed.";
            }

            // ----------------------------------------------------
            // MODE
            // ----------------------------------------------------
            if (mode) {
                mode.textContent =
                    data.mode ||
                    "AI MODEL";
            }

            // ----------------------------------------------------
            // ORIGINAL IMAGE
            // ----------------------------------------------------
            if (resultImg && data.image_url) {
                setImage(
                    absoluteURL(data.image_url),
                    resultImg
                );
            }

            if (empty) {
                empty.style.display = "none";
            }

            // ----------------------------------------------------
            // GRAD-CAM
            // ----------------------------------------------------
            if (heat) {
                heat.innerHTML = "";

                if (data.gradcam_url) {
                    const wrapper =
                        document.createElement("div");

                    wrapper.style.marginTop =
                        "14px";

                    const label =
                        document.createElement("strong");

                    label.textContent =
                        "Grad-CAM attention map";

                    label.style.display =
                        "block";

                    label.style.marginBottom =
                        "8px";

                    const gradImg =
                        document.createElement("img");

                    gradImg.src =
                        absoluteURL(
                            data.gradcam_url
                        );

                    gradImg.alt =
                        "Grad-CAM attention map";

                    gradImg.style.width =
                        "100%";

                    gradImg.style.borderRadius =
                        "12px";

                    gradImg.style.display =
                        "block";

                    wrapper.appendChild(label);
                    wrapper.appendChild(gradImg);
                    heat.appendChild(wrapper);
                }
            }

            // ----------------------------------------------------
            // RESULT
            // ----------------------------------------------------
            if (pred) {
                pred.textContent =
                    displayClassName(
                        data.class_
                    );
            }

            // ----------------------------------------------------
            // CONFIDENCE
            // ----------------------------------------------------
            const confidence =
                Number(
                    data.confidence || 0
                );

            if (conf) {
                conf.textContent =
                    confidence + "%";
            }

            if (prog) {
                prog.style.width =
                    Math.min(
                        100,
                        Math.max(
                            0,
                            confidence
                        )
                    ) + "%";
            }

            // ----------------------------------------------------
            // RISK
            // ----------------------------------------------------
            if (risk) {
                risk.textContent =
                    data.risk ||
                    "Further evaluation recommended.";
            }

            // ----------------------------------------------------
            // EXPLANATION
            // ----------------------------------------------------
            if (explain) {
                explain.textContent =
                    data.explanation ||
                    "AI analysis completed.";
            }

            // ----------------------------------------------------
            // FINDINGS
            // ----------------------------------------------------
            if (lesions) {
                lesions.innerHTML = "";

                const findings =
                    Array.isArray(data.lesions)
                        ? data.lesions
                        : [];

                if (findings.length) {
                    findings.forEach(item => {
                        const span =
                            document.createElement(
                                "span"
                            );

                        span.textContent = item;
                        span.style.display =
                            "inline-block";
                        span.style.padding =
                            "6px 8px";
                        span.style.margin =
                            "7px 5px 0 0";
                        span.style.borderRadius =
                            "7px";
                        span.style.fontSize =
                            "10px";
                        span.style.background =
                            "#edf7f5";
                        span.style.color =
                            "#166f6b";

                        lesions.appendChild(
                            span
                        );
                    });
                } else {
                    lesions.textContent =
                        "No specific findings recorded.";
                }
            }

            // ----------------------------------------------------
            // CURRENT SCREENING
            // ----------------------------------------------------
            if (data.patient) {
                if (currentPatient) {
                    currentPatient.textContent =
                        textValue(
                            data.patient.name
                        );
                }

                if (currentAge) {
                    currentAge.textContent =
                        textValue(
                            data.patient.age
                        );
                }

                if (currentGender) {
                    currentGender.textContent =
                        textValue(
                            data.patient.gender
                        );
                }

                if (currentPatientIdEl) {
                    currentPatientIdEl.textContent =
                        textValue(
                            data.patient.id
                        );
                }
            }

            if (currentResult) {
                currentResult.textContent =
                    displayClassName(
                        data.class_
                    );
            }

            if (currentConfidence) {
                currentConfidence.textContent =
                    confidence + "%";
            }

            if (currentDate) {
                currentDate.textContent =
                    textValue(
                        data.timestamp,
                        "Not available"
                    );
            }

            // ----------------------------------------------------
            // HISTORY SUMMARY
            // ----------------------------------------------------
            if (historySummary) {
                historySummary.textContent =
                    "Latest screening: " +
                    textValue(
                        data.patient?.name,
                        "Patient"
                    ) +
                    " — " +
                    displayClassName(
                        data.class_
                    ) +
                    " — " +
                    textValue(
                        data.timestamp,
                        "Not available"
                    );
            }

            // ----------------------------------------------------
            // LANGUAGE
            // ----------------------------------------------------
            updateLanguage(data);

            console.log(
                "✅ AI analysis completed.",
                {
                    screeningId:
                        data.screeningId,
                    patientId:
                        data.patient?.id
                }
            );

            // ----------------------------------------------------
            // FIREBASE
            // ----------------------------------------------------
            firebaseSavePromise =
                saveScreeningToFirebase(data);

            const firebaseId =
                await firebaseSavePromise;

            if (firebaseId) {
                console.log(
                    "✅ Screening saved to Firebase:",
                    firebaseId
                );

                // Refresh history after saving.
                await refreshHistoryTable();
            } else {
                console.warn(
                    "Screening result is available, but Firebase save failed."
                );
            }

            alert(
                "Screening completed successfully.\n\n" +
                "Screening ID: " +
                (data.screeningId || "Not available")
            );

        } catch (error) {
            console.error(
                "❌ Analysis error:",
                error
            );

            alert(
                error.message ||
                "Something went wrong during analysis."
            );

        } finally {
            setButtonLoading(false);
        }
    });
}

// ------------------------------------------------------------
// CLEAR / NEW SCREENING
// ------------------------------------------------------------
function startNewScreening() {
    if (fileInput) {
        fileInput.value = "";
    }

    selectedFile = null;
    lastResult = null;

    // Do NOT generate an ID here.
    // The backend creates the screening ID only after analysis.
    setScreeningId("");
    resetPatientId();

    // A new patient ID is generated only when Analyze is clicked.
    window.__visionguardCurrentPatientId = "";

    if (patientName) {
        patientName.value = "";
    }

    if (patientAge) {
        patientAge.value = "";
    }

    if (patientGender) {
        patientGender.value = "";
    }

    setImage("", preview);

    if (preview) {
        const objectURL =
            preview.dataset.objectUrl;

        if (objectURL) {
            URL.revokeObjectURL(objectURL);
        }

        delete preview.dataset.objectUrl;
    }

    if (previewInfo) {
        previewInfo.textContent =
            "No image selected";
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent =
            "Analyze retinal image →";
    }

    if (qscore) {
        qscore.textContent = "—";
    }

    if (qmsg) {
        qmsg.textContent =
            "Upload an image to run the quality gate.";
    }

    if (mode) {
        mode.textContent =
            "Waiting for image";
    }

    if (resultImg) {
        setImage("", resultImg);
    }

    if (empty) {
        empty.style.display = "block";
    }

    if (heat) {
        heat.innerHTML = "";
    }

    if (pred) {
        pred.textContent = "—";
    }

    if (conf) {
        conf.textContent = "—";
    }

    if (prog) {
        prog.style.width = "0%";
    }

    if (risk) {
        risk.textContent =
            "Awaiting analysis";
    }

    if (explain) {
        explain.textContent =
            "Upload a fundus image to begin.";
    }

    if (lesions) {
        lesions.textContent = "—";
    }

    if (currentPatient) {
        currentPatient.textContent = "—";
    }

    if (currentAge) {
        currentAge.textContent = "—";
    }

    if (currentGender) {
        currentGender.textContent = "—";
    }

    if (currentPatientIdEl) {
        currentPatientIdEl.textContent = "—";
    }

    if (currentResult) {
        currentResult.textContent = "—";
    }

    if (currentConfidence) {
        currentConfidence.textContent = "—";
    }

    if (currentDate) {
        currentDate.textContent = "—";
    }

    if (historySummary) {
        historySummary.textContent =
            "No screening yet.";
    }

    if (plain) {
        plain.textContent =
            "The system will explain the result in simple language after analysis.";
    }

    if (newScreeningSection) {
        newScreeningSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    console.log(
        "➕ New screening started. No screening ID created yet."
    );
}

function setActiveNav(activeElement) {
    [newScreeningNav, historyNav].forEach(element => {
        if (element) element.classList.remove("active");
    });
    if (activeElement) activeElement.classList.add("active");
}

if (clearBtn) {
    clearBtn.addEventListener(
        "click",
        startNewScreening
    );
}

if (newScreeningNav) {
    newScreeningNav.addEventListener(
        "click",
        () => {
            setActiveNav(newScreeningNav);
            startNewScreening();
        }
    );
}

// ------------------------------------------------------------
// HISTORY NAVIGATION
// ------------------------------------------------------------
if (historyNav) {
    historyNav.addEventListener(
        "click",
        async () => {
            setActiveNav(historyNav);

            if (screeningHistorySection) {
                screeningHistorySection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

            await refreshHistoryTable();
        }
    );
}

if (refreshHistory) {
    refreshHistory.addEventListener(
        "click",
        refreshHistoryTable
    );
}

// ------------------------------------------------------------
// LANGUAGE
// ------------------------------------------------------------
function updateLanguage(data) {
    if (!plain) return;

    const result =
        displayClassName(
            data?.class_
        );

    if (!lang) {
        plain.textContent =
            "Screening result: " +
            result +
            ". Further evaluation by an eye-care professional is recommended.";
        return;
    }

    const selectedLanguage =
        String(lang.value || "");

    if (selectedLanguage.includes("Kannada")) {
        plain.textContent =
            "ಸ್ಕ್ರೀನಿಂಗ್ ಫಲಿತಾಂಶ: " +
            result +
            ". ಹೆಚ್ಚಿನ ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ನೇತ್ರ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಲು ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ.";
    } else if (
        selectedLanguage.includes("Hindi")
    ) {
        plain.textContent =
            "स्क्रीनिंग परिणाम: " +
            result +
            ". आगे के मूल्यांकन के लिए नेत्र विशेषज्ञ से सलाह लेने की अनुशंसा की जाती है।";
    } else {
        plain.textContent =
            "Screening result: " +
            result +
            ". Further evaluation by an eye-care professional is recommended.";
    }
}

if (lang) {
    lang.addEventListener(
        "change",
        () => {
            if (lastResult) {
                updateLanguage(lastResult);
            } else if (plain) {
                plain.textContent =
                    "Run an analysis to generate the patient-friendly explanation.";
            }
        }
    );
}

// ------------------------------------------------------------
// REPORT
// ------------------------------------------------------------
if (reportBtn) {
    reportBtn.addEventListener(
        "click",
        () => {
            if (!lastResult) {
                alert(
                    "Please analyze a retinal image first."
                );
                return;
            }

            generateReport(lastResult);
        }
    );
}

function generateReport(data) {
    const patient =
        data.patient || {};

    const name =
        textValue(
            patient.name,
            "Not provided"
        );

    const age =
        textValue(
            patient.age,
            "Not provided"
        );

    const gender =
        textValue(
            patient.gender,
            "Not provided"
        );

    const patientId =
        textValue(
            patient.id,
            "Not provided"
        );

    const screeningId =
        textValue(
            data.screeningId,
            "Not provided"
        );

    const category =
        displayClassName(
            data.class_
        );

    const confidence =
        Number(
            data.confidence || 0
        );

    const quality =
        Number(
            data.quality?.score || 0
        );

    const riskText =
        textValue(
            data.risk,
            "Further evaluation recommended."
        );

    const explanation =
        textValue(
            data.explanation,
            "AI analysis completed."
        );

    const findings =
        Array.isArray(data.lesions)
            ? data.lesions.join(", ")
            : "None recorded";

    const timestamp =
        textValue(
            data.timestamp,
            new Date().toLocaleString()
        );

    const imageURL =
        absoluteURL(
            data.image_url
        );

    const gradcamURL =
        absoluteURL(
            data.gradcam_url
        );

    const reportHTML = `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>VisionGuard AI - Screening Report</title>
<style>
body{
    font-family:Arial,sans-serif;
    margin:0;
    padding:30px;
    background:#f4f7f7;
    color:#1e2b2b;
}
.container{
    max-width:900px;
    margin:auto;
    background:#fff;
    padding:32px;
    border-radius:18px;
    box-shadow:0 8px 30px rgba(0,0,0,.08);
}
h1{margin:0;color:#166f6b}
h2{color:#166f6b}
.header{
    border-bottom:2px solid #dceeed;
    padding-bottom:18px;
    margin-bottom:22px;
}
.grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
}
.card{
    border:1px solid #e1e7e7;
    padding:18px;
    border-radius:12px;
    margin-top:18px;
}
.label{
    font-size:11px;
    color:#738080;
    text-transform:uppercase;
}
.value{
    font-size:16px;
    font-weight:bold;
    margin-top:5px;
}
img{
    max-width:100%;
    border-radius:12px;
    margin-top:10px;
}
.note{
    background:#fff7df;
    padding:15px;
    border-radius:12px;
    margin-top:22px;
    font-size:13px;
}
.footer{
    text-align:center;
    margin-top:30px;
    color:#7a8585;
    font-size:11px;
}
@media print{
    body{background:#fff;padding:0}
    .container{box-shadow:none}
}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>VISIONGUARD AI</h1>
    <div>Explainable Retinal Screening</div>
    <p>
        Screening ID:
        <strong>${escapeHTML(screeningId)}</strong>
    </p>
</div>

<div class="card">
<h2>Patient Information</h2>
<div class="grid">
    <div>
        <div class="label">Patient ID</div>
        <div class="value">${escapeHTML(patientId)}</div>
    </div>
    <div>
        <div class="label">Patient Name</div>
        <div class="value">${escapeHTML(name)}</div>
    </div>
    <div>
        <div class="label">Age</div>
        <div class="value">${escapeHTML(age)}</div>
    </div>
    <div>
        <div class="label">Gender</div>
        <div class="value">${escapeHTML(gender)}</div>
    </div>
</div>
</div>

<div class="card">
<h2>AI Screening Result</h2>
<div class="grid">
    <div>
        <div class="label">Category</div>
        <div class="value">${escapeHTML(category)}</div>
    </div>
    <div>
        <div class="label">Confidence</div>
        <div class="value">${confidence}%</div>
    </div>
    <div>
        <div class="label">Image Quality</div>
        <div class="value">${quality}%</div>
    </div>
    <div>
        <div class="label">Risk</div>
        <div class="value">${escapeHTML(riskText)}</div>
    </div>
</div>

<h3>Explanation</h3>
<p>${escapeHTML(explanation)}</p>

<h3>Potential Visual Findings</h3>
<p>${escapeHTML(findings)}</p>

<h3>Date & Time</h3>
<p>${escapeHTML(timestamp)}</p>
</div>

${
    imageURL
    ? `
<div class="card">
<h2>Original Retinal Image</h2>
<img src="${escapeHTML(imageURL)}" alt="Original retinal image">
</div>
`
    : ""
}

${
    gradcamURL
    ? `
<div class="card">
<h2>Grad-CAM Explainability</h2>
<p>
The Grad-CAM image shows regions that contributed to
the CNN prediction. It is an explainability aid and
not proof of disease.
</p>
<img src="${escapeHTML(gradcamURL)}" alt="Grad-CAM attention map">
</div>
`
    : ""
}

<div class="note">
<strong>Important:</strong><br><br>
This is an AI-based screening result and is not a medical diagnosis.
The result should be reviewed by a qualified eye-care professional.
</div>

<div class="footer">
VisionGuard AI • Explainable AI • Retinal Screening
</div>

</div>
</body>
</html>
`;

    const blob =
        new Blob(
            [reportHTML],
            {type:"text/html"}
        );

    const url =
        URL.createObjectURL(blob);

    const link =
        document.createElement("a");

    link.href = url;

    link.download =
        "VisionGuard_" +
        screeningId.replace(
            /[^a-zA-Z0-9_-]/g,
            "_"
        ) +
        "_Report.html";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
}

// ------------------------------------------------------------
// INITIAL UI
// ------------------------------------------------------------
// IMPORTANT: This function does NOT generate a screening ID.
function initializeUI() {
    // The existing HTML has six history columns. Keep that layout,
    // but make the first heading explicitly show that the screening
    // ID is displayed underneath the patient name.
    const firstHistoryHeader =
        document.querySelector("#screeningHistorySection thead th");

    if (firstHistoryHeader) {
        firstHistoryHeader.textContent =
            "Patient / Screening ID";
    }

    resetScreeningId();

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
    }

    if (preview) {
        preview.style.display = "none";
    }

    if (resultImg) {
        resultImg.style.display = "none";
    }

    if (empty) {
        empty.style.display = "block";
    }

    console.log(
        "VisionGuard AI app.js loaded successfully."
    );

    console.log(
        "Patient fields:",
        {
            name: !!patientName,
            age: !!patientAge,
            gender: !!patientGender,
            patientIdField: !!patientIdDisplay,
            screeningIdField: !!screeningIdDisplay
        }
    );
}

// ------------------------------------------------------------
// STARTUP
// ------------------------------------------------------------
initializeUI();

// Load history after Firebase module becomes available.
// This DOES NOT create a new screening.
setTimeout(() => {
    refreshHistoryTable();
}, 300);
