import math
import numpy as np


def state_probabilities(
    state: np.ndarray
) -> np.ndarray:

    state = np.asarray(
        state,
        dtype=float
    )

    if state.ndim != 1:
        raise ValueError(
            "State must be one-dimensional."
        )

    norm = np.linalg.norm(
        state,
        ord=2
    )

    if np.isclose(
        norm,
        0.0
    ):
        raise ValueError(
            "State cannot have zero norm."
        )

    state = state / norm

    probabilities = (
        np.abs(state) ** 2
    )

    return probabilities


def sample_probabilities(
    probabilities: np.ndarray,
    shots: int
) -> np.ndarray:

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if shots <= 0:
        raise ValueError(
            "Shots must be positive."
        )

    probability_sum = np.sum(
        probabilities
    )

    if np.isclose(
        probability_sum,
        0.0
    ):
        raise ValueError(
            "Probabilities cannot sum to zero."
        )

    probabilities = (
        probabilities
        / probability_sum
    )

    counts = np.random.multinomial(
        shots,
        probabilities
    )

    return counts / shots


def reconstruct_amplitudes(
    sampled_probabilities: np.ndarray,
    reference_signs: np.ndarray | None = None
) -> np.ndarray:

    sampled_probabilities = np.asarray(
        sampled_probabilities,
        dtype=float
    )

    if sampled_probabilities.ndim != 1:
        raise ValueError(
            "Sampled probabilities must be one-dimensional."
        )

    amplitudes = np.sqrt(
        np.maximum(
            sampled_probabilities,
            0.0
        )
    )

    if reference_signs is not None:

        reference_signs = np.asarray(
            reference_signs,
            dtype=float
        )

        if reference_signs.shape != amplitudes.shape:
            raise ValueError(
                "Reference signs must match the state dimension."
            )

        signs = np.where(
            reference_signs < 0.0,
            -1.0,
            1.0
        )

        amplitudes *= signs

    amplitude_norm = np.linalg.norm(
        amplitudes,
        ord=2
    )

    if np.isclose(
        amplitude_norm,
        0.0
    ):
        raise ValueError(
            "Reconstructed amplitudes cannot have zero norm."
        )

    amplitudes = (
        amplitudes
        / amplitude_norm
    )

    return amplitudes


def linf_error(
    reconstructed_state: np.ndarray,
    reference_state: np.ndarray
) -> float:

    reconstructed_state = np.asarray(
        reconstructed_state,
        dtype=float
    )

    reference_state = np.asarray(
        reference_state,
        dtype=float
    )

    if reconstructed_state.shape != reference_state.shape:
        raise ValueError(
            "States must have the same shape."
        )

    reconstructed_norm = np.linalg.norm(
        reconstructed_state,
        ord=2
    )

    reference_norm = np.linalg.norm(
        reference_state,
        ord=2
    )

    if np.isclose(
        reconstructed_norm,
        0.0
    ):
        raise ValueError(
            "Reconstructed state cannot have zero norm."
        )

    if np.isclose(
        reference_norm,
        0.0
    ):
        raise ValueError(
            "Reference state cannot have zero norm."
        )

    reconstructed_state = (
        reconstructed_state
        / reconstructed_norm
    )

    reference_state = (
        reference_state
        / reference_norm
    )

    return float(
        np.max(
            np.abs(
                reconstructed_state
                - reference_state
            )
        )
    )


def tomography_recovery(
    state: np.ndarray,
    shots: int = 10000
) -> dict:

    state = np.asarray(
        state,
        dtype=float
    )

    if state.ndim != 1:
        raise ValueError(
            "State must be one-dimensional."
        )

    norm = np.linalg.norm(
        state,
        ord=2
    )

    if np.isclose(
        norm,
        0.0
    ):
        raise ValueError(
            "State cannot have zero norm."
        )

    state = (
        state
        / norm
    )

    probabilities = state_probabilities(
        state
    )

    sampled_probabilities = sample_probabilities(
        probabilities,
        shots
    )

    reference_signs = np.sign(
        state
    )

    recovered_state = reconstruct_amplitudes(
        sampled_probabilities,
        reference_signs
    )

    error = linf_error(
        recovered_state,
        state
    )

    return {
        "probabilities": probabilities,
        "sampled_probabilities": sampled_probabilities,
        "recovered_state": recovered_state,
        "linf_error": error,
        "shots": shots
    }


def calculate_alpha(
    jacobian: np.ndarray,
    epsilon: float
) -> float:

    jacobian = np.asarray(
        jacobian,
        dtype=float
    )

    if jacobian.ndim != 2:
        raise ValueError(
            "Jacobian must be a matrix."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    condition_number = np.linalg.cond(
        jacobian
    )

    if not np.isfinite(
        condition_number
    ):
        raise ValueError(
            "Jacobian must be invertible."
        )

    condition_number = max(
        condition_number,
        1.0
    )

    c = (
        condition_number ** 2
        * np.log(
            condition_number / epsilon
        )
    )

    if c <= 0:
        raise ValueError(
            "Invalid value of c."
        )

    j0 = int(
        np.ceil(
            np.sqrt(
                c
                * np.log(
                    4.0 * c / epsilon
                )
            )
        )
    )

    sparsity = np.max(
        np.count_nonzero(
            ~np.isclose(
                jacobian,
                0.0
            ),
            axis=1
        )
    )

    if sparsity <= 0:
        raise ValueError(
            "Jacobian sparsity must be positive."
        )

    term = np.exp(
        math.lgamma(
            2.0 * c + 1.0
        )
        - math.lgamma(
            c + 1.0
        )
        - math.lgamma(
            c + 1.0
        )
        - 2.0 * c * np.log(2.0)
    )

    alpha_sum = term

    for j in range(
        1,
        j0 + 1
    ):

        term *= (
            (c - j + 1.0)
            / (c + j)
        )

        alpha_sum += term

    alpha = (
        4.0
        / sparsity
        * alpha_sum
    )

    return float(
        alpha
    )


def estimate_reference_success_probability(
    normalized_matrix: np.ndarray,
    rhs_state: np.ndarray,
    alpha: float
) -> float:

    normalized_matrix = np.asarray(
        normalized_matrix,
        dtype=float
    )

    rhs_state = np.asarray(
        rhs_state,
        dtype=float
    )

    if normalized_matrix.ndim != 2:
        raise ValueError(
            "normalized_matrix must be a matrix."
        )

    if (
        normalized_matrix.shape[0]
        != normalized_matrix.shape[1]
    ):
        raise ValueError(
            "normalized_matrix must be square."
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    rhs_norm = np.linalg.norm(
        rhs_state,
        ord=2
    )

    if np.isclose(
        rhs_norm,
        0.0
    ):
        raise ValueError(
            "rhs_state cannot have zero norm."
        )

    rhs_state = (
        rhs_state
        / rhs_norm
    )

    solution = np.linalg.solve(
        normalized_matrix,
        rhs_state
    )

    solution_norm = np.linalg.norm(
        solution,
        ord=2
    )

    probability = (
        solution_norm ** 2
        / alpha ** 2
    )

    if probability <= 0:
        raise ValueError(
            "Reference QLSS success probability is not positive."
        )

    return float(
        probability
    )


def calculate_correction_scale(
    alpha: float,
    C_b: float,
    p: float,
    jacobian: np.ndarray
) -> float:

    jacobian = np.asarray(
        jacobian,
        dtype=float
    )

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    if C_b < 0:
        raise ValueError(
            "C_b cannot be negative."
        )

    if p <= 0:
        raise ValueError(
            "p must be positive."
        )

    jacobian_max = np.max(
        np.abs(
            jacobian
        )
    )

    if np.isclose(
        jacobian_max,
        0.0
    ):
        raise ValueError(
            "Jacobian cannot be the zero matrix."
        )

    return float(
        alpha
        * C_b
        * np.sqrt(p)
        / jacobian_max
    )


def scale_correction(
    normalized_correction: np.ndarray,
    C_delta_x: float
) -> np.ndarray:

    normalized_correction = np.asarray(
        normalized_correction,
        dtype=float
    )

    if C_delta_x <= 0:
        raise ValueError(
            "C_delta_x must be positive."
        )

    return (
        C_delta_x
        * normalized_correction
    )


if __name__ == "__main__":

    jacobian = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    rhs = np.array([
        -1.0,
        0.0
    ])

    normalized_jacobian = (
        jacobian
        / np.max(
            np.abs(jacobian)
        )
    )

    C_b = np.linalg.norm(
        rhs,
        ord=2
    )

    normalized_rhs = (
        rhs
        / C_b
    )

    alpha = calculate_alpha(
        normalized_jacobian,
        1e-3
    )

    p_reference = (
        estimate_reference_success_probability(
            normalized_jacobian,
            normalized_rhs,
            alpha
        )
    )

    C_delta_x = calculate_correction_scale(
        alpha,
        C_b,
        p_reference,
        jacobian
    )

    classical_delta_x = np.linalg.solve(
        jacobian,
        rhs
    )

    normalized_delta_x = (
        classical_delta_x
        / np.linalg.norm(
            classical_delta_x,
            ord=2
        )
    )

    tomography_result = tomography_recovery(
        normalized_delta_x,
        shots=10000
    )

    recovered_delta_x = scale_correction(
        tomography_result[
            "recovered_state"
        ],
        C_delta_x
    )

    print("\nJacobian:")
    print(jacobian)

    print("\nNormalized Jacobian:")
    print(normalized_jacobian)

    print("\nC_b:")
    print(C_b)

    print("\nAlpha:")
    print(alpha)

    print("\nReference QLSS success probability p_ref:")
    print(p_reference)

    print("\nC_delta_x:")
    print(C_delta_x)

    print("\nClassical Delta x:")
    print(classical_delta_x)

    print("\nRecovered Delta x:")
    print(recovered_delta_x)

    print("\nCorrection error:")
    print(
        np.linalg.norm(
            recovered_delta_x
            - classical_delta_x,
            ord=2
        )
    )

    print("\nTomography l_inf error:")
    print(
        tomography_result[
            "linf_error"
        ]
    )