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


for shots in [100, 500, 1000]:

    tomography_result = tomography_recovery(
        state,
        shots=shots
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

    print("\n==============================")
    print(f"Shots: {shots}")
    print("==============================")

    print("Measured probabilities:")
    print(
        tomography_result[
            "sampled_probabilities"
        ]
    )

    print("Recovered Delta x:")
    print(recovered_delta_x)

    print("Exact Delta x:")
    print(exact_delta_x)

    print("Tomography l_inf Error:")
    print(
        tomography_result[
            "linf_error"
        ]
    )

    print("Correction Error:")
    print(correction_error)