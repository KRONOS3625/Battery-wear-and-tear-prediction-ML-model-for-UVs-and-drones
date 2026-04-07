from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
ARC_DIR = ROOT / "5. Battery Data Set"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

FEATURE_NAMES = [
    "internal_resistance_ohm",
    "capacity_ah",
    "cycle_number",
    "temperature_c",
    "ambient_temperature_c",
    "avg_voltage_v",
    "min_voltage_v",
    "avg_current_a",
    "max_current_a",
    "current_std_a",
    "discharge_time_s",
    "energy_wh",
    "load_c_rate",
]


@dataclass
class BatteryRow:
    battery_id: str
    cycle_number: int
    cycle_index: int
    capacity_ah: float
    temperature_c: float
    ambient_temperature_c: float
    internal_resistance_ohm: float
    avg_voltage_v: float
    min_voltage_v: float
    avg_current_a: float
    max_current_a: float
    current_std_a: float
    discharge_time_s: float
    energy_wh: float
    load_c_rate: float
    soh_percent: float = 0.0
    rul_cycles: float = 0.0


def scalar(value: object) -> float:
    array = np.real(np.asarray(value)).astype(float).reshape(-1)
    if array.size == 0:
        return float("nan")
    return float(array[0])


def vector(value: object) -> np.ndarray:
    return np.real(np.asarray(value)).astype(float).reshape(-1)


def safe_mean(value: object) -> float:
    array = vector(value)
    return float(np.nanmean(array))


def trapezoid_energy(voltage: np.ndarray, current: np.ndarray, time_seconds: np.ndarray) -> float:
    if voltage.size < 2 or current.size < 2 or time_seconds.size < 2:
        return float("nan")
    power = voltage * np.abs(current)
    watt_seconds = np.trapezoid(power, time_seconds)
    return float(watt_seconds / 3600.0)


def iter_arc_mats() -> list[tuple[str, bytes]]:
    seen: set[str] = set()
    files: list[tuple[str, bytes]] = []
    for zip_path in sorted(ARC_DIR.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".mat"):
                    continue
                battery_id = Path(member).stem
                if battery_id in seen:
                    continue
                seen.add(battery_id)
                files.append((battery_id, archive.read(member)))
    return files


def nearest_impedance(impedance_indexes: np.ndarray, impedance_values: np.ndarray, cycle_index: int) -> float:
    if impedance_indexes.size == 0:
        return float("nan")
    insert_at = int(np.searchsorted(impedance_indexes, cycle_index))
    candidates: list[int] = []
    if insert_at < impedance_indexes.size:
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)
    best = min(candidates, key=lambda idx: abs(int(impedance_indexes[idx]) - cycle_index))
    return float(impedance_values[best])


def extract_rows() -> list[BatteryRow]:
    rows: list[BatteryRow] = []
    for battery_id, payload in iter_arc_mats():
        mat = loadmat(BytesIO(payload), squeeze_me=True, struct_as_record=False)
        battery = mat[battery_id]
        cycles = battery.cycle

        impedance_indexes: list[int] = []
        impedance_values: list[float] = []
        discharge_rows: list[BatteryRow] = []
        discharge_count = 0

        for cycle_index, cycle in enumerate(cycles):
            cycle_type = str(cycle.type).lower()
            if cycle_type == "impedance":
                if hasattr(cycle.data, "Re") and hasattr(cycle.data, "Rct"):
                    total_resistance = scalar(cycle.data.Re) + scalar(cycle.data.Rct)
                    if np.isfinite(total_resistance):
                        impedance_indexes.append(cycle_index)
                        impedance_values.append(total_resistance)
                continue

            if cycle_type != "discharge":
                continue

            discharge_count += 1
            capacity_ah = scalar(cycle.data.Capacity)
            voltage = vector(cycle.data.Voltage_measured)
            current = vector(cycle.data.Current_measured)
            temp = vector(cycle.data.Temperature_measured)
            time_seconds = vector(cycle.data.Time)

            avg_voltage_v = float(np.nanmean(voltage))
            min_voltage_v = float(np.nanmin(voltage))
            avg_current_a = float(np.nanmean(np.abs(current)))
            max_current_a = float(np.nanmax(np.abs(current)))
            current_std_a = float(np.nanstd(current))
            discharge_time_s = float(np.nanmax(time_seconds)) if time_seconds.size else float("nan")
            energy_wh = trapezoid_energy(voltage, current, time_seconds)
            load_c_rate = avg_current_a / max(capacity_ah, 0.1)
            temperature_c = float(np.nanmean(temp))

            row = BatteryRow(
                battery_id=battery_id,
                cycle_number=discharge_count,
                cycle_index=cycle_index,
                capacity_ah=capacity_ah,
                temperature_c=temperature_c,
                ambient_temperature_c=float(cycle.ambient_temperature),
                internal_resistance_ohm=float("nan"),
                avg_voltage_v=avg_voltage_v,
                min_voltage_v=min_voltage_v,
                avg_current_a=avg_current_a,
                max_current_a=max_current_a,
                current_std_a=current_std_a,
                discharge_time_s=discharge_time_s,
                energy_wh=energy_wh,
                load_c_rate=load_c_rate,
            )
            discharge_rows.append(row)

        idx_array = np.asarray(impedance_indexes, dtype=int)
        value_array = np.asarray(impedance_values, dtype=float)
        for row in discharge_rows:
            row.internal_resistance_ohm = nearest_impedance(idx_array, value_array, row.cycle_index)
            rows.append(row)

    filtered: list[BatteryRow] = []
    for row in rows:
        if not np.isfinite(row.internal_resistance_ohm):
            continue
        if not (0.01 <= row.internal_resistance_ohm <= 1.0):
            continue
        if not (0.5 <= row.capacity_ah <= 3.0):
            continue
        if not (0.0 <= row.temperature_c <= 80.0):
            continue
        if not (2.0 <= row.avg_voltage_v <= 5.0):
            continue
        if not (1.5 <= row.min_voltage_v <= 4.5):
            continue
        if not (0.05 <= row.avg_current_a <= 10.0):
            continue
        if not (100.0 <= row.discharge_time_s <= 30000.0):
            continue
        if not np.isfinite(row.energy_wh):
            continue
        filtered.append(row)
    return filtered


def label_rows(rows: list[BatteryRow]) -> list[BatteryRow]:
    by_battery: dict[str, list[BatteryRow]] = {}
    for row in rows:
        by_battery.setdefault(row.battery_id, []).append(row)

    labelled: list[BatteryRow] = []
    for battery_rows in by_battery.values():
        battery_rows.sort(key=lambda item: item.cycle_number)
        initial_capacity = max(row.capacity_ah for row in battery_rows)
        eol_cycle = battery_rows[-1].cycle_number
        for row in battery_rows:
            row.soh_percent = (row.capacity_ah / initial_capacity) * 100.0
            if row.soh_percent <= 70.0:
                eol_cycle = row.cycle_number
                break
        for row in battery_rows:
            row.rul_cycles = max(float(eol_cycle - row.cycle_number), 0.0)
            labelled.append(row)
    return labelled


def save_dataset(rows: list[BatteryRow]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "nasa_arc_battery_features.csv"
    fieldnames = list(asdict(rows[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return csv_path


def build_arrays(rows: list[BatteryRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[getattr(row, feature) for feature in FEATURE_NAMES] for row in rows], dtype=float)
    y_soh = np.asarray([row.soh_percent for row in rows], dtype=float)
    y_rul = np.asarray([row.rul_cycles for row in rows], dtype=float)
    return x, y_soh, y_rul


def summarize_feature_ranges(x: np.ndarray) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        column = x[:, idx]
        ranges[name] = {
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "mean": float(np.mean(column)),
            "p10": float(np.percentile(column, 10)),
            "p50": float(np.percentile(column, 50)),
            "p90": float(np.percentile(column, 90)),
        }
    return ranges


def make_regressors() -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=320,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=360,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=280,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }


def evaluate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[object, dict[str, dict[str, float]], np.ndarray]:
    comparisons: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: object | None = None
    best_mae = float("inf")
    best_pred = np.zeros_like(y_test)

    for name, model in make_regressors().items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        mae = float(mean_absolute_error(y_test, prediction))
        r2 = float(r2_score(y_test, prediction))
        comparisons[name] = {"mae": mae, "r2": r2}
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = model
            best_pred = prediction

    assert best_model is not None
    comparisons["selected"] = {"name": best_name}
    return best_model, comparisons, best_pred


def average_importance(soh_model: object, rul_model: object) -> np.ndarray:
    soh_importance = np.asarray(getattr(soh_model, "feature_importances_", np.zeros(len(FEATURE_NAMES))), dtype=float)
    rul_importance = np.asarray(getattr(rul_model, "feature_importances_", np.zeros(len(FEATURE_NAMES))), dtype=float)
    return (soh_importance + rul_importance) / 2.0


def plot_diagnostics(
    y_soh_true: np.ndarray,
    y_soh_pred: np.ndarray,
    y_rul_true: np.ndarray,
    y_rul_pred: np.ndarray,
    importances: np.ndarray,
) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_soh_true, y_soh_pred, alpha=0.72, color="#0f766e")
    soh_line = [float(np.min(y_soh_true)), float(np.max(y_soh_true))]
    axes[0].plot(soh_line, soh_line, linestyle="--", color="#d97706")
    axes[0].set_title("SoH Prediction")
    axes[0].set_xlabel("Actual SoH (%)")
    axes[0].set_ylabel("Predicted SoH (%)")

    axes[1].scatter(y_rul_true, y_rul_pred, alpha=0.72, color="#1d4ed8")
    rul_line = [float(np.min(y_rul_true)), float(np.max(y_rul_true))]
    axes[1].plot(rul_line, rul_line, linestyle="--", color="#dc2626")
    axes[1].set_title("RUL Prediction")
    axes[1].set_xlabel("Actual Remaining Cycles")
    axes[1].set_ylabel("Predicted Remaining Cycles")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "model_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    order = np.argsort(importances)
    ordered_names = [FEATURE_NAMES[index].replace("_", " ").title() for index in order]
    ax.barh(ordered_names, importances[order], color="#2dd4bf")
    ax.set_title("Average Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train() -> dict[str, object]:
    rows = label_rows(extract_rows())
    csv_path = save_dataset(rows)
    x, y_soh, y_rul = build_arrays(rows)

    x_train, x_test, y_soh_train, y_soh_test, y_rul_train, y_rul_test = train_test_split(
        x,
        y_soh,
        y_rul,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    soh_model, soh_comparison, soh_pred = evaluate_models(x_train, x_test, y_soh_train, y_soh_test)
    rul_model, rul_comparison, rul_pred = evaluate_models(x_train, x_test, y_rul_train, y_rul_test)

    metrics = {
        "soh_mae": float(mean_absolute_error(y_soh_test, soh_pred)),
        "soh_r2": float(r2_score(y_soh_test, soh_pred)),
        "rul_mae": float(mean_absolute_error(y_rul_test, rul_pred)),
        "rul_r2": float(r2_score(y_rul_test, rul_pred)),
    }
    plot_diagnostics(y_soh_test, soh_pred, y_rul_test, rul_pred, average_importance(soh_model, rul_model))

    soh_model.fit(x, y_soh)
    rul_model.fit(x, y_rul)

    anomaly_model = IsolationForest(
        n_estimators=240,
        contamination=0.07,
        random_state=42,
    )
    anomaly_model.fit(x)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_names": FEATURE_NAMES,
            "soh_model": soh_model,
            "rul_model": rul_model,
            "anomaly_model": anomaly_model,
        },
        MODELS_DIR / "battery_health_models.joblib",
    )

    initial_capacity_values: dict[str, float] = {}
    eol_cycle_values: dict[str, float] = {}
    for row in rows:
        initial_capacity_values[row.battery_id] = max(initial_capacity_values.get(row.battery_id, 0.0), row.capacity_ah)
        eol_cycle_values[row.battery_id] = max(eol_cycle_values.get(row.battery_id, 0.0), row.cycle_number + row.rul_cycles)

    ranges = summarize_feature_ranges(x)
    metadata = {
        "dataset_csv": str(csv_path),
        "sample_count": len(rows),
        "battery_count": len(initial_capacity_values),
        "features": FEATURE_NAMES,
        "metrics": metrics,
        "feature_ranges": ranges,
        "model_comparison": {
            "soh": soh_comparison,
            "rul": rul_comparison,
        },
        "baseline": {
            "mean_initial_capacity_ah": float(np.mean(list(initial_capacity_values.values()))),
            "median_eol_cycle": float(np.median(list(eol_cycle_values.values()))),
            "median_pack_size": 12,
        },
        "ui_bounds": {
            "internal_resistance_mohm": {"min": 20, "max": 280, "default": 82},
            "capacity_ah": {"min": round(ranges["capacity_ah"]["p10"], 2), "max": round(ranges["capacity_ah"]["p90"] + 0.2, 2), "default": 1.82},
            "cycle_number": {"min": 1, "max": int(max(220, ranges["cycle_number"]["p90"] + 40)), "default": 124},
            "temperature_c": {"min": 15, "max": 60, "default": 31},
            "voltage_v": {"min": round(ranges["min_voltage_v"]["p10"], 2), "max": round(ranges["avg_voltage_v"]["p90"], 2), "default": round(min(3.72, ranges["avg_voltage_v"]["p90"]), 2)},
            "current_a": {"min": 0.5, "max": round(ranges["max_current_a"]["p90"] + 1.0, 1), "default": 2.4},
        },
        "sources": [
            {
                "name": "NASA Li-ion Battery Aging Dataset",
                "local_folder": str(ARC_DIR),
                "url": "https://data.nasa.gov/dataset/?organization=nasa&tags=batteries&tags=phm",
            },
            {
                "name": "Randomized Battery Usage Series",
                "local_folder": str(ROOT / "11. Randomized Battery Usage Data Set"),
                "url": "https://catalog.data.gov/dataset/randomized-battery-usage-4-40c-right-skewed-random-walk-a3e9a",
            },
        ],
    }
    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    metadata = train()
    print("Training complete.")
    print(f"Samples: {metadata['sample_count']}")
    print(f"Batteries: {metadata['battery_count']}")
    print("Metrics:")
    for key, value in metadata["metrics"].items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
