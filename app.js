const controls = {
  vehicleType: document.getElementById("vehicle-type"),
  chemistry: document.getElementById("chemistry"),
  ir: document.getElementById("internal-resistance"),
  capacity: document.getElementById("capacity"),
  cycles: document.getElementById("cycle-number"),
  temperature: document.getElementById("temperature"),
  ambient: document.getElementById("ambient-temperature"),
  voltage: document.getElementById("voltage"),
  current: document.getElementById("current"),
  soc: document.getElementById("soc"),
  dod: document.getElementById("dod"),
  rest: document.getElementById("rest-hours"),
  age: document.getElementById("age-days"),
  fastCharge: document.getElementById("fast-charge-count"),
  packCells: document.getElementById("pack-series-cells"),
  payload: document.getElementById("payload"),
  tripDistance: document.getElementById("trip-distance"),
  speed: document.getElementById("average-speed"),
  vibration: document.getElementById("vibration"),
};

const displays = {
  vehicleType: document.getElementById("vehicle-type-display"),
  chemistry: document.getElementById("chemistry-display"),
  ir: document.getElementById("ir-display"),
  capacity: document.getElementById("capacity-display"),
  cycles: document.getElementById("cycle-display"),
  temperature: document.getElementById("temperature-display"),
  ambient: document.getElementById("ambient-display"),
  voltage: document.getElementById("voltage-display"),
  current: document.getElementById("current-display"),
  soc: document.getElementById("soc-display"),
  dod: document.getElementById("dod-display"),
  rest: document.getElementById("rest-display"),
  age: document.getElementById("age-display"),
  fastCharge: document.getElementById("fast-charge-display"),
  packCells: document.getElementById("pack-cells-display"),
  payload: document.getElementById("payload-display"),
  distance: document.getElementById("distance-display"),
  speed: document.getElementById("speed-display"),
  vibration: document.getElementById("vibration-display"),
};

let lastResult = null;

function titleCase(value) {
  return value.replace(/-/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function syncDisplays() {
  displays.vehicleType.textContent = titleCase(controls.vehicleType.value);
  displays.chemistry.textContent = titleCase(controls.chemistry.value);
  displays.ir.textContent = controls.ir.value;
  displays.capacity.textContent = Number(controls.capacity.value).toFixed(2);
  displays.cycles.textContent = controls.cycles.value;
  displays.temperature.textContent = controls.temperature.value;
  displays.ambient.textContent = controls.ambient.value;
  displays.voltage.textContent = Number(controls.voltage.value).toFixed(2);
  displays.current.textContent = Number(controls.current.value).toFixed(1);
  displays.soc.textContent = controls.soc.value;
  displays.dod.textContent = controls.dod.value;
  displays.rest.textContent = Number(controls.rest.value).toFixed(1);
  displays.age.textContent = controls.age.value;
  displays.fastCharge.textContent = controls.fastCharge.value;
  displays.packCells.textContent = controls.packCells.value;
  displays.payload.textContent = Number(controls.payload.value).toFixed(1);
  displays.distance.textContent = controls.tripDistance.value;
  displays.speed.textContent = controls.speed.value;
  displays.vibration.textContent = Number(controls.vibration.value).toFixed(1);
}

function currentPayload() {
  return {
    vehicle_type: controls.vehicleType.value,
    chemistry: controls.chemistry.value,
    internal_resistance_mohm: Number(controls.ir.value),
    capacity_ah: Number(controls.capacity.value),
    cycle_number: Number(controls.cycles.value),
    temperature_c: Number(controls.temperature.value),
    ambient_temperature_c: Number(controls.ambient.value),
    voltage_v: Number(controls.voltage.value),
    current_a: Number(controls.current.value),
    soc_percent: Number(controls.soc.value),
    dod_percent: Number(controls.dod.value),
    rest_hours: Number(controls.rest.value),
    age_days: Number(controls.age.value),
    fast_charge_count: Number(controls.fastCharge.value),
    pack_series_cells: Number(controls.packCells.value),
    payload_kg: Number(controls.payload.value),
    trip_distance_km: Number(controls.tripDistance.value),
    average_speed_kmh: Number(controls.speed.value),
    vibration_g: Number(controls.vibration.value),
  };
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
  const minY = Math.min(forecast.eol_capacity, ...yValues) - 0.04;
  const maxY = Math.max(...yValues) + 0.04;

  const toX = (value) => padding + ((value - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2);
  const toY = (value) => height - padding - ((value - minY) / Math.max(maxY - minY, 0.001)) * (height - padding * 2);

  const historyPoints = forecast.history_cycles
    .map((value, index) => `${toX(value)},${toY(forecast.history_capacity[index])}`)
    .join(" ");
  const futurePoints = forecast.future_cycles
    .map((value, index) => `${toX(value)},${toY(forecast.future_capacity[index])}`)
    .join(" ");

  svg.innerHTML = `
    <defs>
      <linearGradient id="forecastGradient" x1="0%" x2="100%">
        <stop offset="0%" stop-color="#2dd4bf" />
        <stop offset="100%" stop-color="#f97316" />
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    <line x1="${padding}" y1="${toY(forecast.eol_capacity)}" x2="${width - padding}" y2="${toY(forecast.eol_capacity)}" stroke="#f59e0b" stroke-dasharray="8 6" stroke-width="2" />
    <polyline fill="none" stroke="#38bdf8" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${historyPoints}" />
    <polyline fill="none" stroke="url(#forecastGradient)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" points="${futurePoints}" />
    <text x="${padding}" y="${padding - 10}" fill="#d2def0" font-size="13">Capacity (Ah)</text>
    <text x="${width - 152}" y="${padding - 10}" fill="#d2def0" font-size="13">Predicted fade</text>
    <text x="${width - 172}" y="${toY(forecast.eol_capacity) - 10}" fill="#f59e0b" font-size="12">70% end-of-life</text>
  `;
}

function renderContributions(contributions) {
  const list = document.getElementById("contribution-list");
  list.innerHTML = contributions
    .map((item) => `
      <div class="contribution-item">
        <div class="contribution-top">
          <span>${item.label}</span>
          <strong>${item.value}%</strong>
        </div>
        <div class="contribution-bar">
          <div class="contribution-fill" style="width:${item.value}%; background:${item.color};"></div>
        </div>
      </div>
    `)
    .join("");
}

function heatColor(soh) {
  if (soh >= 90) return "linear-gradient(135deg, rgba(34,197,94,0.45), rgba(45,212,191,0.32))";
  if (soh >= 82) return "linear-gradient(135deg, rgba(245,158,11,0.45), rgba(249,115,22,0.28))";
  return "linear-gradient(135deg, rgba(239,68,68,0.45), rgba(248,113,113,0.24))";
}

function renderHeatmap(cells) {
  const grid = document.getElementById("heatmap-grid");
  grid.innerHTML = cells
    .map((cell) => `
      <div class="heatmap-cell" style="background:${heatColor(cell.soh)}">
        <span>Cell ${cell.cell}</span>
        <strong>${cell.soh}%</strong>
      </div>
    `)
    .join("");
}

function renderRecommendations(recommendations) {
  const list = document.getElementById("recommendations-list");
  list.innerHTML = recommendations.map((item) => `<div class="alert-item">${item}</div>`).join("");
}

function renderModelComparison(comparison) {
  const container = document.getElementById("model-comparison");
  const formatCards = (label, data) =>
    Object.entries(data)
      .filter(([key]) => key !== "selected")
      .map(([name, metrics]) => `
        <div class="model-card">
          <h4>${titleCase(name)}</h4>
          <p>${label} MAE: ${metrics.mae.toFixed(2)}</p>
          <p>${label} R2: ${metrics.r2.toFixed(3)}</p>
          ${data.selected?.name === name ? '<span class="selected-pill">Selected</span>' : ""}
        </div>
      `)
      .join("");
  container.innerHTML = formatCards("SoH", comparison.soh) + formatCards("RUL", comparison.rul);
}

function fillMetrics(meta) {
  document.getElementById("sample-count").textContent = meta.sample_count;
  document.getElementById("battery-count").textContent = meta.battery_count;
  document.getElementById("soh-metric").textContent = `${meta.metrics.soh_mae.toFixed(2)}%`;
  document.getElementById("rul-metric").textContent = `${meta.metrics.rul_mae.toFixed(1)} cyc`;
  document.getElementById("sample-detail").textContent =
    `${meta.sample_count} real battery snapshots built from discharge, impedance, voltage, current, and temperature records.`;
  document.getElementById("battery-detail").textContent =
    `${meta.battery_count} unique NASA cells across the public aging archives in this workspace.`;
  document.getElementById("soh-detail").textContent =
    `Selected model: ${titleCase(meta.model_comparison.soh.selected.name)} on the NASA feature set.`;
  document.getElementById("rul-detail").textContent =
    `Selected model: ${titleCase(meta.model_comparison.rul.selected.name)} for remaining useful life prediction.`;
  document.getElementById("source-chips").innerHTML = meta.sources.map((item) => `<span class="chip">${item.name}</span>`).join("");
  renderModelComparison(meta.model_comparison);

  const bounds = meta.ui_bounds;
  controls.capacity.min = bounds.capacity_ah.min;
  controls.capacity.max = bounds.capacity_ah.max;
  controls.capacity.value = Math.min(bounds.capacity_ah.max, Math.max(bounds.capacity_ah.min, bounds.capacity_ah.default));
  controls.ir.min = bounds.internal_resistance_mohm.min;
  controls.ir.max = bounds.internal_resistance_mohm.max;
  controls.ir.value = Math.min(bounds.internal_resistance_mohm.max, Math.max(bounds.internal_resistance_mohm.min, bounds.internal_resistance_mohm.default));
  controls.cycles.max = bounds.cycle_number.max;
  controls.cycles.value = Math.min(bounds.cycle_number.max, Math.max(bounds.cycle_number.min, bounds.cycle_number.default));
  controls.temperature.value = Math.min(bounds.temperature_c.max, Math.max(bounds.temperature_c.min, bounds.temperature_c.default));
  controls.voltage.min = bounds.voltage_v.min;
  controls.voltage.max = bounds.voltage_v.max;
  controls.voltage.value = Math.min(bounds.voltage_v.max, Math.max(bounds.voltage_v.min, bounds.voltage_v.default));
  controls.current.max = bounds.current_a.max;
  controls.current.value = Math.min(bounds.current_a.max, Math.max(bounds.current_a.min, bounds.current_a.default));
  syncDisplays();
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  const meta = await response.json();
  fillMetrics(meta);
}

async function runPrediction() {
  const button = document.getElementById("predict-button");
  button.disabled = true;
  button.textContent = "Running battery intelligence...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentPayload()),
    });
    const result = await response.json();
    const output = result.outputs;
    lastResult = result;

    document.getElementById("soh-value").textContent = `${output.soh_percent}%`;
    document.getElementById("wear-value").textContent = `${output.wear_percent}%`;
    document.getElementById("rul-value").textContent = `${output.rul_cycles} cycles`;
    document.getElementById("eol-value").textContent = `${output.estimated_eol_cycle}`;
    document.getElementById("thermal-value").textContent = output.thermal_label;
    document.getElementById("failure-value").textContent = `${output.failure_probability}%`;
    document.getElementById("confidence-value").textContent = `${output.confidence_percent}%`;
    document.getElementById("stress-value").textContent = `${output.stress_index}`;
    document.getElementById("anomaly-value").textContent = output.anomaly_flag ? "Attention" : "Normal";
    document.getElementById("health-badge").textContent = output.health_label;
    document.getElementById("health-text").textContent = output.health_text;
    document.getElementById("forecast-note").textContent =
      `${output.thermal_text} Degradation rate: ${output.degradation_rate_per_cycle}% wear per cycle.`;
    document.getElementById("whatif-soh").textContent = `${result.what_if.projected_soh_percent}%`;
    document.getElementById("whatif-rul").textContent = `${result.what_if.projected_rul_cycles} cycles`;
    document.getElementById("whatif-text").textContent = result.what_if.savings_summary;

    setRing(output.soh_percent);
    renderForecast(result.charts.forecast);
    renderContributions(result.charts.contributions);
    renderHeatmap(result.charts.pack_heatmap);
    renderRecommendations(result.recommendations);
  } catch (error) {
    document.getElementById("health-text").textContent = `Prediction failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run Battery Intelligence";
  }
}

function downloadReport() {
  const button = document.getElementById("download-report");
  button.disabled = true;
  button.textContent = "Preparing Excel...";

  fetch("/api/report.xlsx", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentPayload()),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Report export failed");
      }
      return response.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "battery_intelligence_report.xlsx";
      link.click();
      URL.revokeObjectURL(url);
    })
    .catch((error) => {
      document.getElementById("health-text").textContent = `Excel export failed: ${error.message}`;
    })
    .finally(() => {
      button.disabled = false;
      button.textContent = "Download Report";
    });
}

for (const control of Object.values(controls)) {
  control.addEventListener("input", syncDisplays);
  control.addEventListener("change", syncDisplays);
}

document.getElementById("predict-button").addEventListener("click", runPrediction);
document.getElementById("download-report").addEventListener("click", downloadReport);

syncDisplays();
loadMetadata().then(runPrediction);
