const fields = [
  { key: "N", label: "Nitrogen", unit: "kg/ha", min: 0, max: 300, step: 1, placeholder: "e.g. 90" },
  { key: "P", label: "Phosphorus", unit: "kg/ha", min: 0, max: 300, step: 1, placeholder: "e.g. 42" },
  { key: "K", label: "Potassium", unit: "kg/ha", min: 0, max: 300, step: 1, placeholder: "e.g. 43" },
  { key: "temperature", label: "Temperature", unit: "°C", min: 0, max: 55, step: 0.1, placeholder: "e.g. 21.0" },
  { key: "humidity", label: "Humidity", unit: "%", min: 0, max: 100, step: 0.1, placeholder: "e.g. 82.0" },
  { key: "ph", label: "Soil pH", unit: "pH", min: 0, max: 14, step: 0.1, placeholder: "e.g. 6.5" },
  { key: "rainfall", label: "Rainfall", unit: "mm", min: 0, max: 500, step: 0.1, placeholder: "e.g. 203.0" },
];

const form = document.querySelector("#recommend-form");
const errorBox = document.querySelector("#form-error");
const button = document.querySelector("#submit-button");
const result = document.querySelector("#result-content");

// Render inputs dynamically with accessible labels and attributes
const fieldsContainer = document.querySelector("#fields");
if (fieldsContainer) {
  fieldsContainer.innerHTML = fields.map((field) => `
    <div class="field" id="field-wrap-${field.key}">
      <label for="${field.key}">${field.label} <span class="unit">(${field.unit})</span></label>
      <div class="input-wrap">
        <input id="${field.key}" name="${field.key}" type="number" min="${field.min}" max="${field.max}" step="${field.step}" placeholder="${field.placeholder}" required>
        <span class="suffix">${field.unit}</span>
      </div>
      <div class="field-help">Range ${field.min}–${field.max}</div>
    </div>`).join("");
}

// Track latest request to prevent stale async responses from overwriting newer predictions
let latestRequestId = 0;
let currentAbortController = null;

function validateInputs() {
  const payload = {};
  for (const field of fields) {
    const inputEl = document.getElementById(field.key);
    if (!inputEl) {
      throw new Error(`Form field "${field.label}" is missing.`);
    }
    const rawVal = inputEl.value.trim();
    if (rawVal === "") {
      inputEl.focus();
      throw new Error(`Please enter a value for ${field.label} (${field.key}).`);
    }
    const num = Number(rawVal);
    if (isNaN(num)) {
      inputEl.focus();
      throw new Error(`${field.label} must be a valid number.`);
    }
    if (num < field.min || num > field.max) {
      inputEl.focus();
      throw new Error(`${field.label} (${num} ${field.unit}) must be between ${field.min} and ${field.max} ${field.unit}.`);
    }
    payload[field.key] = num;
  }
  return payload;
}

async function handleRecommendation(event) {
  if (event) {
    event.preventDefault();
  }

  // Clear previous error message
  errorBox.hidden = true;
  errorBox.textContent = "";

  // Validate inputs strictly - never silently substitute defaults or zeros
  let payload;
  try {
    payload = validateInputs();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
    return;
  }

  // Abort any prior in-flight request to prevent race conditions & stale updates
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();
  const requestId = ++latestRequestId;

  // Set loading state on button and result
  button.disabled = true;
  const buttonLabel = button.querySelector("span:first-child");
  const defaultLabel = "Get recommendation";
  if (buttonLabel) {
    buttonLabel.textContent = "Finding your best crop…";
  }
  result.style.opacity = "0.7";

  try {
    const response = await fetch("/v1/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal,
    });

    const data = await response.json();

    // Discard stale responses if a newer request was dispatched in the meantime
    if (requestId !== latestRequestId) {
      return;
    }

    if (!response.ok) {
      let errorMsg = "Could not generate a recommendation. Please check your values.";
      if (typeof data.detail === "string") {
        errorMsg = data.detail;
      } else if (Array.isArray(data.detail) && data.detail[0]?.msg) {
        errorMsg = `${data.detail[0].loc?.slice(-1)[0] || 'Field'}: ${data.detail[0].msg}`;
      }
      throw new Error(errorMsg);
    }

    showResult(data);
  } catch (error) {
    if (error.name === "AbortError") {
      return; // Request was aborted due to a newer submit; ignore
    }
    if (requestId === latestRequestId) {
      errorBox.textContent = error.message || "The recommendation service is unavailable. Please try again.";
      errorBox.hidden = false;
    }
  } finally {
    if (requestId === latestRequestId) {
      button.disabled = false;
      if (buttonLabel) {
        buttonLabel.textContent = defaultLabel;
      }
      result.style.opacity = "1";
    }
  }
}

if (form) {
  form.addEventListener("submit", handleRecommendation);
}

if (button) {
  button.addEventListener("click", (e) => {
    // If outside a standard form submission cycle, trigger directly
    if (form && !form.checkValidity()) {
      handleRecommendation(e);
    }
  });
}

function showResult(data) {
  const confidence = Math.round((data.confidence || 0) * 100);
  const bestCrop = data.recommended_crop || "Unknown";

  // Filter out best match from alternatives to display other strong options cleanly
  const alternatives = (data.top_predictions || []).filter(
    (item) => item.crop.toLowerCase() !== bestCrop.toLowerCase()
  );

  const predictionsHtml = alternatives.length > 0
    ? alternatives.map((item) => {
        const pct = Math.round((item.probability || 0) * 100);
        return `
          <div class="prediction">
            <span>${escapeHtml(item.crop)}</span>
            <span class="bar"><i style="width:${Math.max(0, Math.min(100, pct))}%"></i></span>
            <b>${pct}%</b>
          </div>`;
      }).join("")
    : `<p class="result-foot">No other close alternative crops.</p>`;

  result.classList.add("has-result");
  result.innerHTML = `
    <div class="result-label">BEST MATCH</div>
    <div class="crop-name">${escapeHtml(bestCrop)}</div>
    <p class="confidence-line"><strong>${confidence}% confidence</strong> · based on your field conditions</p>
    <div class="rank-title">OTHER STRONG OPTIONS</div>
    ${predictionsHtml}
    <p class="result-foot">Recommendations are model estimates; local growing conditions may vary.</p>
  `;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}
