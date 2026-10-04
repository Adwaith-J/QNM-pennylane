import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from quantum_linear_solver import solve_fixed_linear_system


def main():

    
    J = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    b = np.array([
        -1.0,
        0.0
    ])

    
    print("\nQNM Quantum Linear-System Validation")
    
    result_exact = solve_fixed_linear_system(
        J,
        b,
        shots=None
    )

    print("\n--- Exact simulator ---")

    print(
        "Exact classical solution:",
        result_exact["exact_solution"]
    )

    print(
        "Recovered quantum correction:",
        result_exact["recovered_delta_x"]
    )

    print(
        "State error:",
        result_exact["state_error_l2"]
    )

    print(
        "Correction error:",
        result_exact["correction_error_l2"]
    )
    result_shots = solve_fixed_linear_system(
        J,
        b,
        shots=1000
    )

    print("\n--- 1000-shot simulator ---")

    print(
        "Measured probabilities:",
        result_shots["probabilities"]
    )

    print(
        "Recovered quantum correction:",
        result_shots["recovered_delta_x"]
    )

    print(
        "State error:",
        result_shots["state_error_l2"]
    )

    print(
        "Correction error:",
        result_shots["correction_error_l2"]
    )
    print("\n--- Quantum configuration ---")

    print(
        "Dimension:",
        result_exact["dimension"]
    )

    print(
        "Qubits:",
        result_exact["qubits"]
    )

    print(
        "Matrix normalization:",
        result_exact["matrix_scale"]
    )

    print(
        "RHS normalization:",
        result_exact["rhs_scale"]
    )
    expected = np.array([
        -1.0 / 6.0,
        -1.0 / 6.0
    ])

    assert np.allclose(
        result_exact["exact_solution"],
        expected
    )

    assert result_exact["state_error_l2"] < 1e-10
    print("\n VALIDATION PASSED")
    

if __name__ == "__main__":
    main()