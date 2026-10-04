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


result = solve_fixed_linear_system(
    A,
    b,
    shots=None
)


tomography_result = tomography_recovery(
    probabilities=result["probabilities"],
    normalized_solution=result["normalized_classical_solution"],
    solution_norm=result["solution_norm"],
    exact_delta_x=result["exact_solution"]
)


print("\n=== Tomography Recovery Test ===")

print("Exact Delta x:")
print(tomography_result["exact_delta_x"])

print("\nRecovered Delta x:")
print(tomography_result["recovered_delta_x"])

print("\nCorrection Error:")
print(tomography_result["correction_error_l2"])