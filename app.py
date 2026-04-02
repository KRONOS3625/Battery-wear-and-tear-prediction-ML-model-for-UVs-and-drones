from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MODEL_BUNDLE = ROOT / "models" / "battery_health_models.joblib"
MODEL_METADATA = ROOT / "models" / "model_metadata.json"


bundle = joblib.load(MODEL_BUNDLE)
with MODEL_METADATA.open("r", encoding="utf-8") as handle:
    metadata = json.load(handle)

soh_model = bundle["soh_model"]
rul_model = bundle["rul_model"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_risk(value: float, minimum: float, maximum: float, invert: bool = False) -> float:
    if maximum <= minimum:
        return 0.0
    score = (value - minimum) / (maximum - minimum)
    score = clamp(score, 0.0, 1.0)
    return (1.0 - score) if invert else score


def health_band(soh: float) -> tuple[str, str]:
    if soh >= 92:
        return "Stable", "The cell is behaving like a healthy pack that still has useful margin."
    if soh >= 84:
        return "Monitor", "Wear is visible, but the pack is still usable with regular checks."
    if soh >= 75:
        return "Elevated", "Degradation is advancing and maintenance planning should start soon."
    return "Critical", "The battery is close to end-of-life and should be replaced or reconditioned."


def thermal_label(temp_c: float) -> tuple[str, str]:
    if temp_c <= 32:
        return "Low", "Thermal stress is within a safe operating band."
    if temp_c <= 40:
        return "Moderate", "Heat is contributing to aging and should be watched."
    return "High", "Thermal stress is severe and is accelerating degradation."


def build_forecast(cycle_number: float, capacity_ah: float, rul_cycles: float, initial_capacity_ah: float) -> dict[str, list[float] | float]:
    eol_capacity = initial_capacity_ah * 0.70
    horizon = max(int(round(rul_cycles)), 1)

    future_steps = np.linspace(0.0, 1.0, 24)
    future_cycles = cycle_number + (future_steps * horizon)
    if capacity_ah <= eol_capacity:
        future_capacity = np.full_like(future_cycles, capacity_ah)
    else:
        future_capacity = capacity_ah - (capacity_ah - eol_capacity) * np.power(future_steps, 1.08)

    history_length = min(max(int(cycle_number), 1), 8)
    history_steps = np.linspace(0.0, 1.0, history_length)
    history_cycles = np.maximum(cycle_number - history_steps[::-1] * max(cycle_number * 0.45, 1.0), 0.0)
    history_capacity = initial_capacity_ah - (initial_capacity_ah - capacity_ah) * history_steps

    return {
        "history_cycles": [float(round(value, 2)) for value in history_cycles],
        "history_capacity": [float(round(value, 4)) for value in history_capacity],
        "future_cycles": [float(round(value, 2)) for value in future_cycles],
        "future_capacity": [float(round(value, 4)) for value in future_capacity],
        "eol_capacity": float(round(eol_capacity, 4)),
    }


def build_contributions(internal_resistance_ohm: float, capacity_ah: float, cycle_number: float, temperature_c: float) -> list[dict[str, float | str]]:
    ranges = metadata["feature_ranges"]
    resistance = scale_risk(
        internal_resistance_ohm,
        ranges["internal_resistance_ohm"]["p10"],
        ranges["internal_resistance_ohm"]["p90"],
    )
    capacity = scale_risk(
        capacity_ah,
        ranges["capacity_ah"]["p10"],
        ranges["capacity_ah"]["p90"],
        invert=True,
    )
    cycles = scale_risk(
        cycle_number,
        ranges["cycle_number"]["p10"],
        ranges["cycle_number"]["p90"],
    )
    thermal = scale_risk(
        temperature_c,
        ranges["temperature_c"]["p10"],
        ranges["temperature_c"]["p90"],
    )

    weighted = [
        ("Internal resistance", 0.08 + resistance, "#f97316"),
        ("Capacity fade", 0.08 + capacity, "#0f766e"),
        ("Cycle stress", 0.08 + cycles, "#1d4ed8"),
        ("Thermal load", 0.08 + thermal, "#dc2626"),
    ]
    total = sum(value for _, value, _ in weighted) or 1.0
    return [
        {"label": label, "value": round((value / total) * 100.0, 1), "color": color}
        for label, value, color in weighted
    ]


def predict(payload: dict[str, object]) -> dict[str, object]:
    internal_resistance_mohm = float(payload["internal_resistance_mohm"])
    capacity_ah = float(payload["capacity_ah"])
    cycle_number = float(payload["cycle_number"])
    temperature_c = float(payload["temperature_c"])

    feature_vector = np.asarray(
        [[internal_resistance_mohm / 1000.0, capacity_ah, cycle_number, temperature_c]],
        dtype=float,
    )
    predicted_soh = clamp(float(soh_model.predict(feature_vector)[0]), 55.0, 100.0)
    predicted_rul = max(0.0, float(rul_model.predict(feature_vector)[0]))

    wear_percent = clamp(100.0 - predicted_soh, 0.0, 100.0)
    state_label, state_text = health_band(predicted_soh)
    thermal_state, thermal_text = thermal_label(temperature_c)
    baseline_capacity = float(metadata["baseline"]["mean_initial_capacity_ah"])

    return {
        "outputs": {
            "soh_percent": round(predicted_soh, 2),
            "wear_percent": round(wear_percent, 2),
            "rul_cycles": round(predicted_rul, 1),
            "estimated_eol_cycle": round(cycle_number + predicted_rul, 1),
            "degradation_rate_per_cycle": round(wear_percent / max(cycle_number, 1.0), 4),
            "health_label": state_label,
            "health_text": state_text,
            "thermal_label": thermal_state,
            "thermal_text": thermal_text,
        },
        "charts": {
            "forecast": build_forecast(cycle_number, capacity_ah, predicted_rul, baseline_capacity),
            "contributions": build_contributions(
                internal_resistance_ohm=internal_resistance_mohm / 1000.0,
                capacity_ah=capacity_ah,
                cycle_number=cycle_number,
                temperature_c=temperature_c,
            ),
        },
        "meta": {
            "model_metrics": metadata["metrics"],
            "battery_count": metadata["battery_count"],
            "sample_count": metadata["sample_count"],
        },
    }


class BatteryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metadata":
            self._send_json(
                {
                    "sample_count": metadata["sample_count"],
                    "battery_count": metadata["battery_count"],
                    "metrics": metadata["metrics"],
                    "sources": metadata["sources"],
                }
            )
            return

        if parsed.path in {"/", ""}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            response = predict(payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(response)


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), BatteryHandler)
    print(f"Battery predictor running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
