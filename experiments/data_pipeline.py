import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
CONFIG_DIR = ROOT_DIR / "experiments" / "configs"


def _convert_to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _convert_to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_convert_to_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return _convert_to_jsonable(value.tolist())

    if isinstance(value, (np.floating, float)):
        return float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if value is None or isinstance(value, (str, bytes)):
        return value

    return str(value)


def ensure_results_directory() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def ensure_config_directory() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def _to_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, np.ndarray):
        return value.astype(float)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)

    return np.asarray([value], dtype=float)


def _to_python_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_iteration_value(item: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in item and item[key] is not None:
        return item[key]
    return default


def normalize_experiment_data(
    problem_name: str,
    quantum_result: dict[str, Any],
    classical_result: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    problem_fn: Any = None,
    raw_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    quantum_result = quantum_result or {}
    classical_result = classical_result or {}

    initial_guess = config.get("initial_guess")
    if initial_guess is None:
        initial_guess = quantum_result.get("initial_guess")
    if initial_guess is None:
        initial_guess = classical_result.get("root")
    if initial_guess is None:
        initial_guess = quantum_result.get("root")

    dimension = config.get("dimension")
    if dimension is None:
        dimension = int(np.asarray(_to_array(initial_guess)).size) if initial_guess is not None else 0

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    experiment_block = {
        "problem": problem_name,
        "dimension": int(dimension),
        "initial_guess": _to_python_list(initial_guess),
        "tolerance": _safe_float(config.get("tolerance"), 1e-8),
        "max_iterations": int(config.get("max_iterations") or quantum_result.get("max_iterations") or 0),
        "timestamp": timestamp,
        "version": "person-4-dashboard-v1",
    }

    quantum_block = {
        "num_qubits": _safe_float(quantum_result.get("num_qubits"), 0.0),
        "shots": _safe_float(quantum_result.get("shots"), config.get("shots")),
        "simulator": quantum_result.get("simulator") or config.get("simulator") or "default.qubit",
        "circuit_depth": _safe_float(quantum_result.get("circuit_depth"), None),
        "measurement_settings": quantum_result.get("measurement_settings") or config.get("measurement_settings"),
    }

    classical_block = {
        "final_root": _to_python_list(classical_result.get("root")),
        "final_residual": _safe_float(classical_result.get("final_residual"), None),
        "iterations": int(classical_result.get("iterations") or 0),
    }

    quantum_newton_block = {
        "final_root": _to_python_list(quantum_result.get("root")),
        "final_residual": _safe_float(quantum_result.get("final_residual"), None),
        "iterations": int(quantum_result.get("iterations") or 0),
    }

    iteration_entries = []
    history = quantum_result.get("history") or []
    if not history and classical_result is not None:
        history = classical_result.get("history") or []

    for index, item in enumerate(history):
        x_value = _to_array(_infer_iteration_value(item, "x", []))
        x_list = x_value.tolist() if x_value.size else []

        fx_value = _infer_iteration_value(item, "F_x")
        if fx_value is None and problem_fn is not None and x_value.size:
            try:
                fx_value = np.asarray(problem_fn(x_value), dtype=float).tolist()
            except Exception:
                fx_value = []

        correction = _infer_iteration_value(item, "correction")
        if correction is None:
            correction = _infer_iteration_value(item, "delta_x", [])

        delta_array = _to_array(correction)
        delta_list = delta_array.tolist() if delta_array.size else []

        residual = _infer_iteration_value(item, "residual_norm")
        if residual is None and fx_value:
            residual = float(np.linalg.norm(np.asarray(fx_value, dtype=float), ord=2))

        correction_norm = _infer_iteration_value(item, "correction_norm")
        if correction_norm is None and delta_array.size:
            correction_norm = float(np.linalg.norm(delta_array, ord=2))

        quantum_error = _infer_iteration_value(item, "correction_error")
        if quantum_error is None:
            quantum_error = _infer_iteration_value(item, "quantum_error")

        entry = {
            "iteration": int(_infer_iteration_value(item, "iteration", index)),
            "x": x_list,
            "F_x": fx_value if fx_value is not None else [],
            "residual_norm": _safe_float(residual, None),
            "delta_x": delta_list,
            "correction_norm": _safe_float(correction_norm, None),
            "quantum_error": _safe_float(quantum_error, None),
        }
        iteration_entries.append(entry)

    normalized = {
        "experiment": experiment_block,
        "quantum": quantum_block,
        "classical": classical_block,
        "quantum_newton": quantum_newton_block,
        "iterations": iteration_entries,
    }

    if raw_result is not None:
        normalized["raw"] = raw_result

    return normalized


def save_experiment(payload: dict[str, Any], filename: str | None = None) -> Path:
    ensure_results_directory()

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}.json"

    jsonable_payload = _convert_to_jsonable(payload)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(jsonable_payload, indent=2), encoding="utf-8")
    return path


def load_experiment(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in experiment file: {path}") from exc

    validate_experiment(data)
    return data


def validate_experiment(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise ValueError("Experiment payload must be a dictionary.")

    for key in ("experiment", "quantum", "quantum_newton", "iterations"):
        if key not in payload:
            raise ValueError(f"Experiment payload missing required key: {key}")

    if not isinstance(payload["iterations"], list):
        raise ValueError("Experiment iterations must be a list.")

    experiment = payload["experiment"]
    if not isinstance(experiment, dict) or not experiment.get("problem"):
        raise ValueError("Experiment metadata is missing the problem name.")

    return True


def list_saved_experiments() -> list[Path]:
    ensure_results_directory()
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda path: path.name)


def prepare_iteration_dataframe(payload: dict[str, Any]) -> list[dict[str, Any]]:
    validate_experiment(payload)
    rows: list[dict[str, Any]] = []

    for item in payload.get("iterations", []):
        row = {
            "iteration": item.get("iteration"),
            "x": item.get("x", []),
            "F_x": item.get("F_x", []),
            "residual_norm": item.get("residual_norm"),
            "delta_x": item.get("delta_x", item.get("correction", [])),
            "correction_norm": item.get("correction_norm"),
            "quantum_error": item.get("quantum_error", item.get("correction_error")),
        }
        rows.append(row)

    return rows


if __name__ == "__main__":
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Config directory: {CONFIG_DIR}")
