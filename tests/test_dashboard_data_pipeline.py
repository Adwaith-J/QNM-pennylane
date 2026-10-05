import json
from pathlib import Path

import numpy as np

from experiments.data_pipeline import (
    load_experiment,
    normalize_experiment_data,
    prepare_iteration_dataframe,
    save_experiment,
    validate_experiment,
)


def _sample_quantum_result():
    return {
        "root": np.array([1.0, 1.5]),
        "final_residual": 1.0e-9,
        "iterations": 3,
        "history": [
            {
                "iteration": 0,
                "x": np.array([2.0, 1.0]),
                "F_x": np.array([3.0, 1.0]),
                "residual_norm": 3.16227766,
                "delta_x": np.array([-0.5, 0.5]),
                "correction_norm": 0.70710678,
                "correction_error": 1.0e-8,
            },
            {
                "iteration": 1,
                "x": np.array([1.5, 1.5]),
                "F_x": np.array([1.0, 0.0]),
                "residual_norm": 1.0,
                "delta_x": np.array([-0.4, 0.0]),
                "correction_norm": 0.4,
                "correction_error": 2.0e-8,
            },
        ],
        "shots": 1000,
        "simulator": "default.qubit",
        "num_qubits": 2,
        "circuit_depth": 5,
        "measurement_settings": {"shots": 1000},
    }


def _sample_classical_result():
    return {
        "root": np.array([1.5, 0.5]),
        "final_residual": 1.0e-10,
        "iterations": 2,
        "history": [
            {
                "iteration": 0,
                "x": np.array([2.0, 1.0]),
                "residual_norm": 3.16227766,
                "correction": np.array([-0.5, 0.5]),
                "correction_norm": 0.70710678,
            }
        ],
    }


def test_normalize_experiment_data_accumulates_iteration_values():
    payload = normalize_experiment_data(
        "Quadratic system",
        quantum_result=_sample_quantum_result(),
        classical_result=_sample_classical_result(),
        config={"initial_guess": [2.0, 1.0], "tolerance": 1e-8, "max_iterations": 50, "shots": 1000},
        problem_fn=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1] - 1.0]),
    )

    assert payload["experiment"]["problem"] == "Quadratic system"
    assert payload["quantum"]["shots"] == 1000
    assert payload["iterations"][0]["iteration"] == 0
    assert payload["iterations"][0]["residual_norm"] == 3.16227766
    assert payload["iterations"][0]["quantum_error"] == 1.0e-8


def test_save_and_load_experiment_round_trip(tmp_path):
    payload = normalize_experiment_data(
        "Quadratic system",
        quantum_result=_sample_quantum_result(),
        classical_result=_sample_classical_result(),
        config={"initial_guess": [2.0, 1.0], "tolerance": 1e-8, "max_iterations": 50},
        problem_fn=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1] - 1.0]),
    )

    file_path = tmp_path / "saved_exp.json"
    file_path.write_text(json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)), encoding="utf-8")
    loaded = load_experiment(file_path)

    validate_experiment(loaded)
    assert loaded["quantum_newton"]["final_residual"] == 1.0e-9
    assert loaded["iterations"][0]["iteration"] == 0


def test_prepare_iteration_dataframe_uses_expected_columns():
    payload = normalize_experiment_data(
        "Quadratic system",
        quantum_result=_sample_quantum_result(),
        classical_result=_sample_classical_result(),
        config={"initial_guess": [2.0, 1.0], "tolerance": 1e-8, "max_iterations": 50},
        problem_fn=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1] - 1.0]),
    )

    rows = prepare_iteration_dataframe(payload)
    assert rows[0]["iteration"] == 0
    assert rows[0]["delta_x"] == [-0.5, 0.5]
    assert rows[0]["correction_norm"] == 0.70710678


def test_validate_experiment_fails_for_missing_payload_keys():
    try:
        validate_experiment({"experiment": {"problem": "x"}})
        assert False, "validate_experiment should reject incomplete payloads"
    except ValueError:
        pass
