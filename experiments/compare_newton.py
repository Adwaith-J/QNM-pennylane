import sys
from pathlib import Path

import numpy as np

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

from src.problems import F, jacobian, INITIAL_GUESS
from src.classical_newton import newton_method
from src.quantum_newton import quantum_newton_method


def run_comparison():

    print("\n=== Classical Newton vs Quantum Newton ===")

    classical_result = newton_method(
        F,
        jacobian,
        INITIAL_GUESS
    )

    quantum_result = quantum_newton_method(
        F,
        jacobian,
        INITIAL_GUESS,
        tolerance=1e-8,
        max_iterations=20,
        shots=None,
        tomography_shots=10000,
        alpha_epsilon=1e-3
    )

    classical_root = np.asarray(
        classical_result["root"],
        dtype=float
    )

    quantum_root = np.asarray(
        quantum_result["root"],
        dtype=float
    )

    root_error = np.linalg.norm(
        quantum_root
        - classical_root,
        ord=2
    )

    print("\nClassical Newton")
    print("----------------")
    print("Root:")
    print(classical_root)

    print("\nFinal residual:")
    print(
        classical_result[
            "final_residual"
        ]
    )

    print("\nIterations:")
    print(
        classical_result[
            "iterations"
        ]
    )

    print("\nExecution time:")
    print(
        classical_result[
            "execution_time"
        ]
    )

    print("\nQuantum Newton")
    print("-------------")
    print("Root:")
    print(quantum_root)

    print("\nFinal residual:")
    print(
        quantum_result[
    "final_residual"
]
    )

    print("\nIterations:")
    print(
        quantum_result[
            "iterations"
        ]
    )

    print("\nExecution time:")
    print(
        quantum_result[
            "execution_time"
        ]
    )

    print("\nComparison")
    print("----------")
    print("Root error:")
    print(root_error)


if __name__ == "__main__":
    run_comparison()