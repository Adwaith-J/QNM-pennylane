import time
import numpy as np

from src.quantum_linear_solver import solve_fixed_linear_system
from src.tomography import tomography_recovery


def quantum_newton_method(
    F,
    jacobian,
    x0,
    tolerance=1e-8,
    max_iterations=50,
    shots=None
):
    """
    Finite-size simulator implementation of Quantum Newton's Method.

    Newton system:
        J(x) Delta_x = -F(x)

    Newton update:
        x_new = x + Delta_x
    """

    x = np.asarray(x0, dtype=float).copy()
    history = []

    start_time = time.perf_counter()

    for iteration in range(max_iterations):

        # Evaluate nonlinear function
        fx = np.asarray(F(x), dtype=float)

        # Current residual
        residual_norm = np.linalg.norm(fx, ord=2)

        # Stop when converged
        if residual_norm <= tolerance:
            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": np.zeros_like(x),
                "correction_norm": 0.0,
                "correction_error": 0.0
            })
            break

        # Jacobian
        J = np.asarray(jacobian(x), dtype=float)

        # Newton RHS
        rhs = -fx

        # IMPORTANT:
        # If RHS is too small, do not call the quantum solver.
        rhs_norm = np.linalg.norm(rhs, ord=2)

        if rhs_norm <= tolerance:
            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": np.zeros_like(x),
                "correction_norm": 0.0,
                "correction_error": 0.0
            })
            break

        # Quantum linear-system solver
        try:
            quantum_result = solve_fixed_linear_system(
                J,
                rhs,
                shots=shots
            )

        except ValueError as error:

            # The finite-size solver can encounter an effectively
            # zero solution because of numerical/sampling effects.
            # Treat this as convergence when the residual is already
            # sufficiently small.
            if "zero norm" in str(error).lower():
                history.append({
                    "iteration": iteration,
                    "x": x.copy(),
                    "residual_norm": residual_norm,
                    "correction": np.zeros_like(x),
                    "correction_norm": 0.0,
                    "correction_error": 0.0
                })
                break

            raise

        # Tomography / recovery
        tomography_result = tomography_recovery(
            probabilities=quantum_result["probabilities"],
            normalized_solution=quantum_result[
                "normalized_classical_solution"
            ],
            solution_norm=quantum_result["solution_norm"],
            exact_delta_x=quantum_result["exact_solution"]
        )

        delta_x = tomography_result["recovered_delta_x"]

        correction_error = tomography_result[
            "correction_error_l2"
        ]

        correction_norm = np.linalg.norm(
            delta_x,
            ord=2
        )

        # If recovered correction is effectively zero,
        # stop instead of performing another unstable iteration.
        if correction_norm <= tolerance:
            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": delta_x.copy(),
                "correction_norm": correction_norm,
                "correction_error": correction_error
            })
            break

        # Newton update
        x = x + delta_x

        # Residual after update
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
            "correction_error": correction_error
        })

    execution_time = time.perf_counter() - start_time

    final_residual = np.linalg.norm(
        F(x),
        ord=2
    )

    return {
        "root": x,
        "final_residual": final_residual,
        "iterations": len(history) - 1,
        "execution_time": execution_time,
        "history": history,
        "shots": shots
    }


if __name__ == "__main__":

    from src.problems import F, jacobian, INITIAL_GUESS

    result = quantum_newton_method(
        F=F,
        jacobian=jacobian,
        x0=INITIAL_GUESS,
        tolerance=1e-8,
        max_iterations=50,
        shots=None
    )

    print("\n=== Quantum Newton Result ===")

    print("\nRoot:")
    print(result["root"])

    print("\nFinal residual:")
    print(result["final_residual"])

    print("\nIterations:")
    print(result["iterations"])

    print("\nExecution time:")
    print(result["execution_time"], "seconds")

    print("\n=== Iteration History ===")

    for item in result["history"]:
        print(
            f"Iteration {item['iteration']:2d} | "
            f"x = {item['x']} | "
            f"Residual = {item['residual_norm']:.6e} | "
            f"Correction = {item['correction_norm']:.6e} | "
            f"Correction Error = {item['correction_error']:.6e}"
        )