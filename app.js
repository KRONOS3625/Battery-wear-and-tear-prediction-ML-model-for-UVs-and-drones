const controls = {
  ir: document.getElementById("internal-resistance"),
  capacity: document.getElementById("capacity"),
  cycles: document.getElementById("cycle-number"),
  temperature: document.getElementById("temperature"),
};

const displays = {
  ir: document.getElementById("ir-display"),
  capacity: document.getElementById("capacity-display"),
  cycles: document.getElementById("cycle-display"),
  temperature: document.getElementById("temperature-display"),
};

function syncDisplays() {
  displays.ir.textContent = controls.ir.value;
  displays.capacity.textContent = Number(controls.capacity.value).toFixed(2);
  displays.cycles.textContent = controls.cycles.value;
  displays.temperature.textContent = controls.temperature.value;
}

function setRing(soh) {
  const ring = document.querySelector(".health-ring");
  const degrees = Math.max(0, Math.min(360, (Number(soh) / 100) * 360));
  let color = "#2dd4bf";
  if (soh < 84) color = "#f59e0b";
  if (soh < 75) color = "#ef4444";
  ring.style.background = `conic-gradient(${color} ${degrees}deg, rgba(255,255,255,0.08) ${degrees}deg)`;
}

function renderForecast(forecast) {
  const svg = document.getElementById("forecast-chart");
  const width = 760;
  const height = 320;
  const padding = 38;

  const xValues = [...forecast.history_cycles, ...forecast.future_cycles];
  const yValues = [...forecast.history_capacity, ...forecast.future_capacity];

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(forecast.eol_capacity, ...yValues) - 0.03;
  const maxY = Math.max(...yValues) + 0.03;

  const toX = (value) =>
    padding + ((value - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2);
  const toY = (value) =>
    height - padding - ((value - minY) / Math.max(maxY - minY, 0.001)) * (height - padding * 2);

  const historyPoints = forecast.history_cycles
    .map((value, index) => `${toX(value)},${toY(forecast.history_capacity[index])}`)
    .join(" ");
  const futurePoints = forecast.future_cycles
    .map((value, index) => `${toX(value)},${toY(forecast.future_capacity[index])}`)
    .join(" ");

  const eolY = toY(forecast.eol_capacity);
  const grid = [0.25, 0.5, 0.75]
    .map((ratio) => {
      const y = padding + ratio * (height - padding * 2);
      return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-width="1" />`;
    })
    .join("");

  svg.innerHTML = `
    <defs>
      <linearGradient id="forecastGradient" x1="0%" x2="100%">
        <stop offset="0%" stop-color="#2dd4bf" />
        <stop offset="100%" stop-color="#f97316" />
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${grid}
    <line x1="${padding}" y1="${eolY}" x2="${width - padding}" y2="${eolY}" stroke="#f59e0b" stroke-dasharray="8 6" stroke-width="2" />
    <polyline fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${historyPoints}" />
    <polyline fill="none" stroke="url(#forecastGradient)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" points="${futurePoints}" />
    <circle cx="${toX(forecast.history_cycles.at(-1))}" cy="${toY(forecast.history_capacity.at(-1))}" r="5.5" fill="#38bdf8"></circle>
    <circle cx="${toX(forecast.future_cycles.at(-1))}" cy="${toY(forecast.future_capacity.at(-1))}" r="6" fill="#f97316"></circle>
    <text x="${padding}" y="${padding - 10}" fill="#c3d1e7" font-size="13">Capacity (Ah)</text>
    <text x="${width - 140}" y="${padding - 10}" fill="#f8fafc" font-size="13">Cycle forecast</text>
    <text x="${width - 170}" y="${eolY - 10}" fill="#f59e0b" font-size="12">End-of-life threshold</text>
    <text x="${padding}" y="${height - 10}" fill="#7f92b0" font-size="12">${Math.round(minX)} cycles</text>
    <text x="${width - 130}" y="${height - 10}" fill="#7f92b0" font-size="12">${Math.round(maxX)} cycles</text>
  `;
}

function renderContributions(contributions) {
  const list = document.getElementById("contribution-list");
  list.innerHTML = contributions
    .map(
      (item) => `
        <div class="contribution-item">
          <div class="contribution-top">
            <span>${item.label}</span>
            <strong>${item.value}%</strong>
          </div>
          <div class="contribution-bar">
            <div class="contribution-fill" style="width:${item.value}%; background:${item.color};"></div>
          </div>
        </div>
      `
    )
    .join("");
}

function fillMetrics(meta) {
  document.getElementById("sample-count").textContent = meta.sample_count;
  document.getElementById("battery-count").textContent = meta.battery_count;
  document.getElementById("soh-metric").textContent = `${meta.metrics.soh_mae.toFixed(2)}%`;
  document.getElementById("rul-metric").textContent = `${meta.metrics.rul_mae.toFixed(1)} cyc`;
  document.getElementById("source-chips").innerHTML = meta.sources
    .map((item) => `<span class="chip">${item.name}</span>`)
    .join("");
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  const meta = await response.json();
  fillMetrics(meta);
}

async function runPrediction() {
  const button = document.getElementById("predict-button");
  button.disabled = true;
  button.textContent = "Running prediction...";

  const payload = {
    internal_resistance_mohm: Number(controls.ir.value),
    capacity_ah: Number(controls.capacity.value),
    cycle_number: Number(controls.cycles.value),
    temperature_c: Number(controls.temperature.value),
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    const output = result.outputs;

    document.getElementById("soh-value").textContent = `${output.soh_percent}%`;
    document.getElementById("wear-value").textContent = `${output.wear_percent}%`;
    document.getElementById("rul-value").textContent = `${output.rul_cycles} cycles`;
    document.getElementById("eol-value").textContent = `${output.estimated_eol_cycle}`;
    document.getElementById("thermal-value").textContent = output.thermal_label;
    document.getElementById("health-badge").textContent = output.health_label;
    document.getElementById("health-text").textContent = output.health_text;
    document.getElementById("forecast-note").textContent =
      `${output.thermal_text} Estimated degradation rate: ${output.degradation_rate_per_cycle}% wear per cycle.`;

    setRing(output.soh_percent);
    renderForecast(result.charts.forecast);
    renderContributions(result.charts.contributions);
  } catch (error) {
    document.getElementById("health-text").textContent = `Prediction failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Predict Wear and Tear";
  }
}

for (const control of Object.values(controls)) {
  control.addEventListener("input", syncDisplays);
}

document.getElementById("predict-button").addEventListener("click", runPrediction);

syncDisplays();
loadMetadata().then(runPrediction);
