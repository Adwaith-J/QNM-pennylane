import numpy as np


def reconstruct_amplitudes(probabilities: np.ndarray) -> np.ndarray:
    """
    Recover amplitude magnitudes from measured probabilities.

    Since probability p_i = |x_i|^2,
    the amplitude magnitude is sqrt(p_i).
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    return np.sqrt(np.maximum(probabilities, 0.0))


def recover_solution_from_measurements(
    probabilities: np.ndarray,
    signs: np.ndarray,
    solution_norm: float,
    original_dimension: int
) -> np.ndarray:
    """
    Recover the classical correction vector Delta x
    from finite-size measurement probabilities.

    The quantum solver gives probabilities of the encoded
    solution state. We reconstruct amplitudes and restore
    the signs and original solution norm.
    """

    amplitudes = reconstruct_amplitudes(probabilities)

    signs = np.asarray(signs, dtype=float)

    if len(signs) != original_dimension:
        raise ValueError("Sign vector dimension mismatch.")

    if len(amplitudes) < original_dimension:
        raise ValueError("Not enough measured amplitudes.")

    recovered_normalized = (
        amplitudes[:original_dimension] * signs
    )

    norm = np.linalg.norm(recovered_normalized)

    if np.isclose(norm, 0.0):
        raise ValueError(
            "Recovered solution has zero norm."
        )

    recovered_normalized = (
        recovered_normalized / norm
    )

    recovered_delta_x = (
        recovered_normalized * solution_norm
    )

    return recovered_delta_x


def correction_error(
    recovered_delta_x: np.ndarray,
    exact_delta_x: np.ndarray
) -> float:
    """
    Calculate the L2 error between the recovered
    quantum correction and the exact classical correction.
    """

    recovered_delta_x = np.asarray(
        recovered_delta_x,
        dtype=float
    )

    exact_delta_x = np.asarray(
        exact_delta_x,
        dtype=float
    )

    return float(
        np.linalg.norm(
            recovered_delta_x - exact_delta_x,
            ord=2
        )
    )


def tomography_recovery(
    probabilities: np.ndarray,
    normalized_solution: np.ndarray,
    solution_norm: float,
    exact_delta_x: np.ndarray
):
    """
    Complete finite-size tomography/recovery step.

    Returns the recovered correction and its L2 error.
    """

    normalized_solution = np.asarray(
        normalized_solution,
        dtype=float
    )

    exact_delta_x = np.asarray(
        exact_delta_x,
        dtype=float
    )

    signs = np.sign(normalized_solution)

    signs[
        np.isclose(normalized_solution, 0.0)
    ] = 0.0

    recovered_delta_x = recover_solution_from_measurements(
        probabilities=probabilities,
        signs=signs,
        solution_norm=solution_norm,
        original_dimension=len(normalized_solution)
    )

    error = correction_error(
        recovered_delta_x,
        exact_delta_x
    )

    return {
        "recovered_delta_x": recovered_delta_x,
        "exact_delta_x": exact_delta_x,
        "correction_error_l2": error,
        "probabilities": probabilities,
        "normalized_solution": normalized_solution,
        "solution_norm": solution_norm,
    }