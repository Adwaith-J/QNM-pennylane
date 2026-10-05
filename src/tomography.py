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

    norm = np.linalg.norm(state)

    if np.isclose(norm, 0.0):
        raise ValueError(
            "State cannot have zero norm."
        )

    state = state / norm

    return np.abs(state) ** 2


def sample_probabilities(
    state: np.ndarray,
    shots: int
) -> np.ndarray:

    if shots <= 0:
        raise ValueError(
            "shots must be positive."
        )

    probabilities = state_probabilities(
        state
    )

    outcomes = np.random.choice(
        len(probabilities),
        size=shots,
        p=probabilities
    )

    counts = np.bincount(
        outcomes,
        minlength=len(probabilities)
    )

    return counts / shots


def reconstruct_amplitudes(
    probabilities: np.ndarray
) -> np.ndarray:

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "Probabilities cannot be negative."
        )

    total = np.sum(probabilities)

    if np.isclose(total, 0.0):
        raise ValueError(
            "Probabilities cannot all be zero."
        )

    probabilities = (
        probabilities / total
    )

    return np.sqrt(
        np.maximum(
            probabilities,
            0.0
        )
    )


def linf_error(
    recovered: np.ndarray,
    exact: np.ndarray
) -> float:

    recovered = np.asarray(
        recovered,
        dtype=float
    )

    exact = np.asarray(
        exact,
        dtype=float
    )

    if recovered.shape != exact.shape:
        raise ValueError(
            "Recovered and exact states must have the same shape."
        )

    return float(
        np.max(
            np.abs(
                recovered - exact
            )
        )
    )


def tomography_recovery(
    state: np.ndarray,
    shots: int
):

    state = np.asarray(
        state,
        dtype=float
    )

    state_norm = np.linalg.norm(state)

    if np.isclose(
        state_norm,
        0.0
    ):
        raise ValueError(
            "State cannot have zero norm."
        )

    normalized_state = (
        state / state_norm
    )

    probabilities = sample_probabilities(
        normalized_state,
        shots
    )

    amplitudes = reconstruct_amplitudes(
        probabilities
    )

    reference_signs = np.sign(
        normalized_state
    )

    reference_signs[
        np.isclose(
            normalized_state,
            0.0
        )
    ] = 0.0

    recovered_state = (
        amplitudes
        * reference_signs
    )

    recovered_norm = np.linalg.norm(
        recovered_state
    )

    if np.isclose(
        recovered_norm,
        0.0
    ):
        raise ValueError(
            "Tomography produced a zero state."
        )

    recovered_state = (
        recovered_state
        / recovered_norm
    )

    error_linf = linf_error(
        recovered_state,
        normalized_state
    )

    return {
        "probabilities": probabilities,
        "amplitudes": amplitudes,
        "recovered_state": recovered_state,
        "linf_error": error_linf,
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
        math.lgamma(2.0 * c + 1.0)
        - math.lgamma(c + 1.0)
        - math.lgamma(c + 1.0)
        - 2.0 * c * np.log(2.0)
    )

    alpha_sum = term

    for j in range(1, j0 + 1):

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

    return float(alpha)


def estimate_success_probability(
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

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    rhs_norm = np.linalg.norm(
        rhs_state
    )

    if np.isclose(
        rhs_norm,
        0.0
    ):
        raise ValueError(
            "rhs_state cannot have zero norm."
        )

    rhs_state = (
        rhs_state / rhs_norm
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
            "Estimated QLSS success probability is not positive."
        )

    return float(
        min(
            probability,
            1.0
        )
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
        np.abs(jacobian)
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

    from quantum_linear_solver import (
    normalize_matrix,
    prepare_rhs_state
)

    J = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    b = np.array([
        -1.0,
        0.0
    ])

    normalized_J, J_scale = normalize_matrix(
        J
    )

    normalized_b, C_b = prepare_rhs_state(
        b
    )

    epsilon = 1e-3

    alpha = calculate_alpha(
        normalized_J,
        epsilon
    )

    p = estimate_success_probability(
        normalized_J,
        normalized_b,
        alpha
    )

    C_delta_x = calculate_correction_scale(
        alpha,
        C_b,
        p,
        J
    )

    classical_delta_x = np.linalg.solve(
        J,
        b
    )

    normalized_delta_x = (
        classical_delta_x
        / np.linalg.norm(
            classical_delta_x
        )
    )

    scaled_delta_x = scale_correction(
        normalized_delta_x,
        C_delta_x
    )

    print("\nJacobian:")
    print(J)

    print("\nNormalized Jacobian:")
    print(normalized_J)

    print("\n||J||_max:")
    print(J_scale)

    print("\nC_b:")
    print(C_b)

    print("\nAlpha:")
    print(alpha)

    print("\nQLSS success probability p:")
    print(p)

    print("\nC_delta_x:")
    print(C_delta_x)

    print("\nClassical Delta x:")
    print(classical_delta_x)

    print("\n||Delta x||:")
    print(
        np.linalg.norm(
            classical_delta_x
        )
    )

    print("\nNormalized Delta x:")
    print(normalized_delta_x)

    print("\nScaled Delta x:")
    print(scaled_delta_x)

    print("\nScaling error:")
    print(
        np.linalg.norm(
            scaled_delta_x
            - classical_delta_x
        )
    )

    tomography_result = tomography_recovery(
        normalized_delta_x,
        shots=10000
    )

    print("\nTomography probabilities:")
    print(
        tomography_result[
            "probabilities"
        ]
    )

    print("\nTomography amplitudes:")
    print(
        tomography_result[
            "amplitudes"
        ]
    )

    print("\nTomography recovered state:")
    print(
        tomography_result[
            "recovered_state"
        ]
    )

    print("\nl_inf tomography error:")
    print(
        tomography_result[
            "linf_error"
        ]
    )