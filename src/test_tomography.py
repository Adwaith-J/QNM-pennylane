import numpy as np

from src.tomography import (
    tomography_recovery,
    scale_correction
)


state = np.array([
    -1.0 / np.sqrt(2.0),
    -1.0 / np.sqrt(2.0)
])

C_delta_x = np.sqrt(
    1.0 / 18.0
)

exact_delta_x = np.array([
    -1.0 / 6.0,
    -1.0 / 6.0
])

tomography_result = tomography_recovery(
    state,
    shots=10000
)

recovered_delta_x = scale_correction(
    tomography_result["recovered_state"],
    C_delta_x
)

correction_error = np.linalg.norm(
    recovered_delta_x
    - exact_delta_x,
    ord=2
)

print("\n=== Tomography Recovery Test ===")

print("Recovered Delta x:")
print(recovered_delta_x)

print("\nExact Delta x:")
print(exact_delta_x)

print("\nTomography l_inf Error:")
print(
    tomography_result["linf_error"]
)

print("\nCorrection Error:")
print(correction_error)