import numpy as np

from src.quantum_linear_solver import solve_fixed_linear_system
from src.tomography import tomography_recovery


A = np.array([
    [4.0, 2.0],
    [1.0, -1.0]
])

b = np.array([
    -1.0,
    0.0
])


for shots in [100, 500, 1000]:

    result = solve_fixed_linear_system(
        A,
        b,
        shots=shots
    )

    tomography_result = tomography_recovery(
        probabilities=result["probabilities"],
        normalized_solution=result["normalized_classical_solution"],
        solution_norm=result["solution_norm"],
        exact_delta_x=result["exact_solution"]
    )

    print("\n==============================")
    print(f"Shots: {shots}")
    print("==============================")

    print("Measured probabilities:")
    print(result["probabilities"])

    print("Recovered Delta x:")
    print(tomography_result["recovered_delta_x"])

    print("Exact Delta x:")
    print(tomography_result["exact_delta_x"])

    print("Correction Error:")
    print(tomography_result["correction_error_l2"])