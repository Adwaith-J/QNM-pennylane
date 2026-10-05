import sys
import csv
import time
from pathlib import Path

import numpy as np

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

from src.quantum_newton import quantum_newton_method




def create_problem(n):
    """
    Create an n-dimensional nonlinear system.

    F_i(x) = x_i^2 - (i + 1)

    Jacobian:
    J_ii = 2*x_i
    """

    target = np.arange(1, n + 1, dtype=float)

    def F(x):
        return x**2 - target

    def jacobian(x):
        return np.diag(2.0 * x)

    # Positive initial guess
    x0 = np.ones(n, dtype=float)

    return F, jacobian, x0


def run_scaling_experiment():

    dimensions = [2, 4, 8]
    shots_list = [100, 500, 1000]

    results = []

    for n in dimensions:

        F, jacobian, x0 = create_problem(n)

        for shots in shots_list:

            start_time = time.perf_counter()

            result = quantum_newton_method(
                F=F,
                jacobian=jacobian,
                x0=x0,
                tolerance=1e-8,
                max_iterations=50,
                shots=shots
            )

            runtime = time.perf_counter() - start_time

            qubits = int(np.ceil(np.log2(n)))

            row = {
                "dimension": n,
                "qubits": qubits,
                "shots": shots,
                "iterations": result["iterations"],
                "runtime_seconds": runtime,
                "final_residual": result["final_residual"]
            }

            results.append(row)

            print(
                f"Dimension={n} | "
                f"Qubits={qubits} | "
                f"Shots={shots} | "
                f"Iterations={result['iterations']} | "
                f"Runtime={runtime:.6f}s | "
                f"Residual={result['final_residual']:.6e}"
            )

    output_file = "experiments/results/scaling_results.csv"

    with open(output_file, "w", newline="") as file:

        fieldnames = [
            "dimension",
            "qubits",
            "shots",
            "iterations",
            "runtime_seconds",
            "final_residual"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n================================")
    print("Scaling experiment completed.")
    print("Results saved to:")
    print(output_file)
    print("================================")


if __name__ == "__main__":
    run_scaling_experiment()