import time

import numpy as np
from scipy.linalg import solve


def newton_method(
    F,
    jacobian,
    x0,
    tolerance=1e-10,
    max_iterations=50
):
    """
    Classical Newton's method for solving F(x) = 0.

    Newton step:

        J(x_k) Δx_k = -F(x_k)

    Update:

        x_(k+1) = x_k + Δx_k
    """

    x = np.asarray(x0, dtype=float).copy()

    history = []

    start_time = time.perf_counter()

    for iteration in range(max_iterations):

        fx = np.asarray(F(x), dtype=float)

        residual_norm = np.linalg.norm(fx, ord=2)

        if residual_norm < tolerance:

            history.append({
                "iteration": iteration,
                "x": x.copy(),
                "residual_norm": residual_norm,
                "correction": np.zeros_like(x),
                "correction_norm": 0.0
            })

            break

        J = np.asarray(jacobian(x), dtype=float)

        # Solve:
        # J Δx = -F(x)

        delta_x = solve(J, -fx)

        correction_norm = np.linalg.norm(
            delta_x,
            ord=2
        )

        history.append({
            "iteration": iteration,
            "x": x.copy(),
            "residual_norm": residual_norm,
            "correction": delta_x.copy(),
            "correction_norm": correction_norm
        })

        # Newton update

        x = x + delta_x

    else:

        fx = np.asarray(F(x), dtype=float)

        history.append({
            "iteration": max_iterations,
            "x": x.copy(),
            "residual_norm": np.linalg.norm(
                fx,
                ord=2
            ),
            "correction": np.zeros_like(x),
            "correction_norm": 0.0
        })

    execution_time = (
        time.perf_counter() - start_time
    )

    final_residual = np.linalg.norm(
        F(x),
        ord=2
    )

    return {
        "root": x,
        "final_residual": final_residual,
        "iterations": len(history) - 1,
        "execution_time": execution_time,
        "history": history
    }
if __name__ == "__main__":

    from problems import F, jacobian, INITIAL_GUESS

    result = newton_method(
        F,
        jacobian,
        INITIAL_GUESS
    )

    print("\n=== Classical Newton Result ===")

    print("Root:", result["root"])

    print(
        "Final residual:",
        result["final_residual"]
    )

    print(
        "Iterations:",
        result["iterations"]
    )

    print(
        "Execution time:",
        result["execution_time"],
        "seconds"
    )

    print("\n=== Iteration History ===")

    for item in result["history"]:

        print(
            f"Iteration {item['iteration']:2d} | "
            f"x = {item['x']} | "
            f"Residual = "
            f"{item['residual_norm']:.6e} | "
            f"Correction = "
            f"{item['correction_norm']:.6e}"
        )