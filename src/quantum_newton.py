import sys
import time
import numpy as np

if __package__ is None:
    sys.path.append(
        str(
            __import__(
                "pathlib"
            ).Path(__file__).resolve().parent.parent
        )
    )

from src.quantum_linear_solver import  (
    normalize_matrix,
    prepare_rhs_state,
    solve_fixed_linear_system
)

from src.tomography import (
    tomography_recovery,
    calculate_alpha,
    estimate_reference_success_probability,
    calculate_correction_scale,
    scale_correction
)


def quantum_newton_method(
    F,
    jacobian,
    x0,
    tolerance=1e-8,
    max_iterations=50,
    shots=None,
    tomography_shots=10000,
    alpha_epsilon=1e-3
):

    x = np.asarray(
        x0,
        dtype=float
    ).copy()

    history = []

    start_time = time.perf_counter()

    for iteration in range(max_iterations):

        fx = np.asarray(
            F(x),
            dtype=float
        )

        residual_norm = np.linalg.norm(
            fx,
            ord=2
        )

        if residual_norm <= tolerance:

            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": np.zeros_like(x),
                "correction_norm": 0.0,
                "correction_error": 0.0,
                "tomography_linf_error": 0.0,
                "alpha": None,
                "reference_success_probability": None,
                "C_delta_x": 0.0,
                "C_b": 0.0,
                "jacobian_scale": 0.0
            })

            break

        J = np.asarray(
            jacobian(x),
            dtype=float
        )

        rhs = -fx

        rhs_norm = np.linalg.norm(
            rhs,
            ord=2
        )

        if rhs_norm <= tolerance:

            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": np.zeros_like(x),
                "correction_norm": 0.0,
                "correction_error": 0.0,
                "tomography_linf_error": 0.0,
                "alpha": None,
                "reference_success_probability": None,
                "C_delta_x": 0.0,
                "C_b": rhs_norm,
                "jacobian_scale": 0.0
            })

            break

        normalized_J, J_scale = normalize_matrix(
            J
        )

        normalized_b, C_b = prepare_rhs_state(
            rhs
        )

        quantum_result = solve_fixed_linear_system(
            J,
            rhs,
            shots=shots,
            run_qlss=True
        )

        normalized_delta_x = np.asarray(
            quantum_result[
                "quantum_solution_state"
            ],
            dtype=float
        )

        tomography_result = tomography_recovery(
            normalized_delta_x,
            shots=tomography_shots
        )

        recovered_normalized_delta_x = np.asarray(
            tomography_result[
                "recovered_state"
            ],
            dtype=float
        )

        alpha = calculate_alpha(
            normalized_J,
            alpha_epsilon
        )

        reference_success_probability = (
            estimate_reference_success_probability(
                normalized_J,
                normalized_b,
                alpha
            )
        )

        C_delta_x = calculate_correction_scale(
            alpha,
            C_b,
            reference_success_probability,
            J
        )

        delta_x = scale_correction(
            recovered_normalized_delta_x,
            C_delta_x
        )

        correction_norm = np.linalg.norm(
            delta_x,
            ord=2
        )

        classical_delta_x = np.linalg.solve(
            J,
            rhs
        )

        correction_error = np.linalg.norm(
            delta_x - classical_delta_x,
            ord=2
        )

        x = x + delta_x

        new_residual_norm = np.linalg.norm(
            F(x),
            ord=2
        )

        history.append({
            "iteration": iteration,
            "x": x.copy(),
            "residual_norm": new_residual_norm,
            "correction": delta_x.copy(),
            "correction_norm": correction_norm,
            "correction_error": correction_error,
            "tomography_linf_error": (
                tomography_result[
                    "linf_error"
                ]
            ),
            "alpha": alpha,
            "reference_success_probability": (
                reference_success_probability
            ),
            "C_delta_x": C_delta_x,
            "C_b": C_b,
            "jacobian_scale": J_scale
        })

        if new_residual_norm <= tolerance:
            break

        if correction_norm <= tolerance:
            break

    execution_time = (
        time.perf_counter()
        - start_time
    )

    final_residual = np.linalg.norm(
        F(x),
        ord=2
    )

    return {
        "root": x,
        "final_residual": final_residual,
        "iterations": len(history),
        "execution_time": execution_time,
        "history": history,
        "shots": shots,
        "tomography_shots": tomography_shots,
        "alpha_epsilon": alpha_epsilon
    }


if __name__ == "__main__":

    from src.problems import (
        F,
        jacobian,
        INITIAL_GUESS
    )

    result = quantum_newton_method(
        F=F,
        jacobian=jacobian,
        x0=INITIAL_GUESS,
        tolerance=1e-8,
        max_iterations=50,
        shots=None,
        tomography_shots=10000,
        alpha_epsilon=1e-3
    )

    print("\n=== Quantum Newton Result ===")

    print("\nRoot:")
    print(result["root"])

    print("\nFinal residual:")
    print(result["final_residual"])

    print("\nIterations:")
    print(result["iterations"])

    print("\nExecution time:")
    print(
        result["execution_time"],
        "seconds"
    )

    print("\n=== Iteration History ===")

    for item in result["history"]:

        print(
            f"Iteration "
            f"{item['iteration']:2d} | "
            f"x = {item['x']} | "
            f"Residual = "
            f"{item['residual_norm']:.6e} | "
            f"Correction = "
            f"{item['correction_norm']:.6e} | "
            f"Correction Error = "
            f"{item['correction_error']:.6e}"
        )

        if item["alpha"] is not None:

            print(
                f"             "
                f"alpha = "
                f"{item['alpha']:.6e} | "
                f"p_ref = "
                f"{item['reference_success_probability']:.6e} | "
                f"C_delta_x = "
                f"{item['C_delta_x']:.6e} | "
                f"Tomography l_inf Error = "
                f"{item['tomography_linf_error']:.6e}"
            )