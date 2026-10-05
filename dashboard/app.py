from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data_pipeline import (
    list_saved_experiments,
    load_experiment,
    normalize_experiment_data,
    prepare_iteration_dataframe,
    save_experiment,
    validate_experiment,
)
from src.classical_newton import newton_method
from src.problems import F, INITIAL_GUESS, jacobian
from src.quantum_newton import quantum_newton_method

st.set_page_config(page_title="Quantum Newton Dashboard", layout="wide")

PROBLEM_NAME = "Quadratic system from src/problems.py"


def render_problem_panel() -> dict:
    st.sidebar.header("Problem / Experiment")

    problem_options = {
        PROBLEM_NAME: {
            "F": F,
            "jacobian": jacobian,
            "initial_guess": np.asarray(INITIAL_GUESS, dtype=float),
        }
    }

    selected_problem = st.sidebar.selectbox("Problem", list(problem_options.keys()))
    problem_data = problem_options[selected_problem]

    initial_guess = st.sidebar.text_input(
        "Initial guess",
        value=", ".join(str(v) for v in problem_data["initial_guess"]),
    )
    try:
        guess_values = np.asarray(
            [float(part.strip()) for part in initial_guess.split(",") if part.strip()],
            dtype=float,
        )
        if guess_values.size == 0:
            raise ValueError
    except ValueError:
        st.sidebar.warning("Initial guess is invalid. Using the repository default.")
        guess_values = problem_data["initial_guess"].copy()

    tolerance = st.sidebar.number_input("Tolerance", value=1e-8, format="%.1e")
    max_iterations = st.sidebar.number_input("Maximum iterations", min_value=1, value=50, step=1)
    shots = st.sidebar.number_input("Number of shots", min_value=1, value=1000, step=1)
    tomography_shots = st.sidebar.number_input("Tomography shots", min_value=1, value=10000, step=1)
    alpha_epsilon = st.sidebar.number_input("Alpha epsilon", value=1e-3, format="%.1e")
    simulator = st.sidebar.selectbox("Quantum simulator/device", ["default.qubit"])

    return {
        "problem": selected_problem,
        "F": problem_data["F"],
        "jacobian": problem_data["jacobian"],
        "dimension": int(guess_values.size),
        "initial_guess": guess_values,
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "shots": int(shots),
        "tomography_shots": int(tomography_shots),
        "alpha_epsilon": float(alpha_epsilon),
        "simulator": simulator,
    }


def run_qnm_experiment(config: dict) -> dict:
    problem_config = {
        "problem": config["problem"],
        "dimension": int(config["dimension"]),
        "initial_guess": np.asarray(config["initial_guess"], dtype=float),
        "tolerance": float(config["tolerance"]),
        "max_iterations": int(config["max_iterations"]),
        "shots": int(config["shots"]),
        "simulator": config["simulator"],
        "tomography_shots": int(config["tomography_shots"]),
        "alpha_epsilon": float(config["alpha_epsilon"]),
    }

    quantum_result = quantum_newton_method(
        F=config["F"],
        jacobian=config["jacobian"],
        x0=np.asarray(config["initial_guess"], dtype=float),
        tolerance=float(config["tolerance"]),
        max_iterations=int(config["max_iterations"]),
        shots=int(config["shots"]),
        tomography_shots=int(config["tomography_shots"]),
        alpha_epsilon=float(config["alpha_epsilon"]),
    )

    classical_result = newton_method(
        config["F"],
        config["jacobian"],
        np.asarray(config["initial_guess"], dtype=float),
        tolerance=float(config["tolerance"]),
        max_iterations=int(config["max_iterations"]),
    )

    quantum_result["simulator"] = config["simulator"]
    quantum_result["num_qubits"] = int(np.ceil(np.log2(max(2, len(np.asarray(config["initial_guess"], dtype=float))))))

    normalized = normalize_experiment_data(
        problem_name=config["problem"],
        quantum_result=quantum_result,
        classical_result=classical_result,
        config=problem_config,
        problem_fn=config["F"],
        raw_result={
            "classical_result": classical_result,
            "quantum_result": quantum_result,
        },
    )

    path = save_experiment(normalized)
    normalized["saved_path"] = str(path)
    return normalized


def _format_metric_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{float(item):.6g}" for item in value) + "]"
    if isinstance(value, np.generic):
        return value.item()
    return value


def display_summary(summary_payload: dict) -> None:
    st.subheader("Summary")
    experiment = summary_payload.get("experiment", {})
    quantum_newton = summary_payload.get("quantum_newton", {})

    cols = st.columns(5)
    metrics = [
        ("Initial guess", experiment.get("initial_guess")),
        ("Final root", quantum_newton.get("final_root")),
        ("Final residual", quantum_newton.get("final_residual")),
        ("Iterations", quantum_newton.get("iterations")),
        ("Converged", "Yes" if (quantum_newton.get("final_residual") is not None and quantum_newton.get("final_residual") < experiment.get("tolerance", np.inf)) else "No"),
    ]

    for column, (label, value) in zip(cols, metrics):
        column.metric(label, _format_metric_value(value))

    st.write("- Tolerance:", experiment.get("tolerance"))
    st.write("- Maximum iterations:", experiment.get("max_iterations"))
    st.write("- Shot count:", summary_payload.get("quantum", {}).get("shots"))
    st.write("- Quantum configuration:", summary_payload.get("quantum", {}))


def display_convergence_plot(payload: dict) -> None:
    iterations = payload.get("iterations", [])
    if not iterations:
        st.info("No iteration data is available for the convergence plot.")
        return

    rows = [{"iteration": item.get("iteration"), "residual_norm": item.get("residual_norm")} for item in iterations]
    df = pd.DataFrame(rows)
    if df.empty or "residual_norm" not in df.columns:
        st.info("Residual history is unavailable in the current experiment output.")
        return

    st.subheader("Residual vs Iteration")
    st.line_chart(df.set_index("iteration")["residual_norm"], use_container_width=True)


def display_root_trajectory(payload: dict) -> None:
    iterations = payload.get("iterations", [])
    if not iterations:
        st.info("No iteration data is available for the root trajectory plot.")
        return

    trajectory = []
    for item in iterations:
        x_values = item.get("x") or []
        if len(x_values) == 2:
            trajectory.append({
                "iteration": item.get("iteration"),
                "x1": float(x_values[0]),
                "x2": float(x_values[1]),
            })

    if not trajectory:
        st.info("The 2D Newton root trajectory is unavailable for the current experiment.")
        return

    df = pd.DataFrame(trajectory)
    st.subheader("Newton Root Trajectory")
    st.line_chart(df.set_index("iteration")[["x1", "x2"]], use_container_width=True)


def display_correction_plot(payload: dict) -> None:
    iterations = payload.get("iterations", [])
    if not iterations:
        st.info("No iteration data is available for the correction magnitude plot.")
        return

    rows = []
    for item in iterations:
        if item.get("correction_norm") is not None:
            rows.append({
                "iteration": item.get("iteration"),
                "correction_norm": float(item["correction_norm"]),
            })

    if not rows:
        st.info("Correction magnitude is unavailable in the current experiment output.")
        return

    df = pd.DataFrame(rows)
    st.subheader("Correction Magnitude vs Iteration")
    st.line_chart(df.set_index("iteration")["correction_norm"], use_container_width=True)


def display_quantum_error(payload: dict) -> None:
    iterations = payload.get("iterations", [])
    rows = []
    for item in iterations:
        value = item.get("quantum_error")
        if value is None:
            value = item.get("correction_error")
        if value is not None:
            rows.append({"iteration": item.get("iteration"), "quantum_error": float(value)})

    if not rows:
        st.info("Quantum-vs-classical error data is not available in the current experiment output.")
        return

    st.subheader("Quantum vs Classical Error")
    df = pd.DataFrame(rows)
    st.line_chart(df.set_index("iteration")["quantum_error"], use_container_width=True)


def display_comparison(payload: dict) -> None:
    st.subheader("Classical vs Quantum Comparison")
    classical = payload.get("classical", {})
    quantum = payload.get("quantum_newton", {})
    latest_error = None
    for item in payload.get("iterations", []):
        latest_error = item.get("quantum_error")
        if latest_error is not None:
            break

    comparison_rows = {
        "Metric": [
            "Classical final root",
            "Quantum Newton final root",
            "Classical residual",
            "Quantum Newton residual",
            "Classical iterations",
            "Quantum iterations",
            "Quantum correction error",
        ],
        "Value": [
            classical.get("final_root"),
            quantum.get("final_root"),
            classical.get("final_residual"),
            quantum.get("final_residual"),
            classical.get("iterations"),
            quantum.get("iterations"),
            latest_error,
        ],
    }

    if all(value is None for value in comparison_rows["Value"]):
        st.info("Classical comparison data is not available for the current experiment.")
        return

    st.table(pd.DataFrame(comparison_rows))
    st.caption("Simulator runtime is not evidence of quantum computational speedup; this dashboard visualizes finite PennyLane simulations.")


def display_iteration_table(payload: dict) -> None:
    st.subheader("Iteration Table")
    iterations = prepare_iteration_dataframe(payload)
    if not iterations:
        st.info("No iteration data is available to display in the table.")
        return

    st.dataframe(pd.DataFrame(iterations), use_container_width=True)


def display_quantum_configuration(payload: dict) -> None:
    st.subheader("Quantum Configuration")
    quantum = payload.get("quantum", {})
    config_items = {
        "Number of qubits": quantum.get("num_qubits"),
        "Number of shots": quantum.get("shots"),
        "Simulator/device": quantum.get("simulator"),
        "Circuit depth": quantum.get("circuit_depth"),
        "Measurement settings": quantum.get("measurement_settings"),
    }

    for label, value in config_items.items():
        if value is None:
            st.write(f"- {label}: unavailable")
        else:
            st.write(f"- {label}: {value}")


def display_algorithm_explanation() -> None:
    st.subheader("Algorithm Explanation")
    st.markdown(
        """
        Classical nonlinear problem
        ↓
        Compute F(x)
        ↓
        Compute Jacobian J(x)
        ↓
        Solve J(x) Δx = -F(x)
        ↓
        Quantum linear-system component
        ↓
        Classical/measurement recovery
        ↓
        Newton update
        ↓
        Convergence check
        ↓
        Next iteration

        This dashboard visualizes the existing implementation in the repository without rewriting the underlying algorithm.
        """
    )


def display_caveats() -> None:
    st.subheader("Caveats")
    st.warning(
        "PennyLane simulator results are finite practical simulations. Simulator runtime is not evidence of quantum computational speedup. "
        "The implementation may use approximations required for finite simulation. Classical baseline results are retained for validation."
    )


def render_saved_experiments() -> list[Path]:
    saved_experiments = list_saved_experiments()
    if not saved_experiments:
        st.info("No saved experiment files are available yet.")
        return []

    selected_experiment = st.selectbox("Saved experiments", [path.name for path in saved_experiments])
    return [path for path in saved_experiments if path.name == selected_experiment]


def main() -> None:
    st.title("Quantum Newton’s Method — PennyLane Implementation")
    st.caption("This is a finite PennyLane simulator implementation and should not be presented as proof of asymptotic quantum speedup.")

    config = render_problem_panel()

    col_run, col_load = st.columns(2)
    if col_run.button("Run Quantum Newton"):
        try:
            experiment = run_qnm_experiment(config)
            st.session_state["experiment"] = experiment
            st.success(f"Experiment saved to: {experiment['saved_path']}")
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Experiment failed: {exc}")

    if col_load.button("Load Saved Experiment"):
        try:
            matches = render_saved_experiments()
            if matches:
                loaded = load_experiment(matches[0])
                validate_experiment(loaded)
                st.session_state["experiment"] = loaded
                st.success(f"Loaded saved experiment: {matches[0].name}")
            else:
                st.info("No saved experiments were found in the results directory.")
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.warning(f"Could not load the selected experiment: {exc}")

    current = st.session_state.get("experiment")
    if current is None:
        st.info("Run a new experiment or load a saved experiment to visualize the results.")
        display_algorithm_explanation()
        display_caveats()
        return

    display_summary(current)
    display_convergence_plot(current)
    display_root_trajectory(current)
    display_correction_plot(current)
    display_quantum_error(current)
    display_comparison(current)
    display_iteration_table(current)
    display_quantum_configuration(current)
    display_algorithm_explanation()
    display_caveats()


if __name__ == "__main__":
    main()
