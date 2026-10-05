import numpy as np

from src.quantum_linear_solver import (
    normalize_matrix,
    prepare_rhs_state,
    qlss_solver
)

from src.tomography import (
    tomography_recovery,
    calculate_alpha,
    calculate_correction_scale,
    scale_correction
)

from src.quantum_newton import quantum_newton_method

from src.problems import (
    F,
    jacobian,
    INITIAL_GUESS
)


def test_matrix_normalization():

    matrix = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    normalized, scale = normalize_matrix(
        matrix
    )

    expected = np.array([
        [1.0, 0.5],
        [0.25, -0.25]
    ])

    assert np.allclose(
        normalized,
        expected
    )

    assert np.isclose(
        scale,
        4.0
    )


def test_rhs_normalization():

    rhs = np.array([
        -1.0,
        0.0
    ])

    normalized, norm = prepare_rhs_state(
        rhs
    )

    assert np.allclose(
        normalized,
        [-1.0, 0.0]
    )

    assert np.isclose(
        norm,
        1.0
    )


def test_qlss_solver():

    matrix = np.array([
        [1.0, 0.5],
        [0.25, -0.25]
    ])

    rhs_state = np.array([
        -1.0,
        0.0
    ])

    solution = qlss_solver(
        matrix,
        rhs_state
    )

    expected = np.array([
        -1.0 / np.sqrt(2.0),
        -1.0 / np.sqrt(2.0)
    ])

    assert np.allclose(
        solution,
        expected,
        atol=1e-8
    )


def test_tomography_recovery():

    state = np.array([
        -1.0 / np.sqrt(2.0),
        -1.0 / np.sqrt(2.0)
    ])

    result = tomography_recovery(
        state,
        shots=100000
    )

    recovered = result[
        "recovered_state"
    ]

    assert np.allclose(
        np.linalg.norm(
            recovered
        ),
        1.0
    )

    assert result[
        "linf_error"
    ] < 0.01


def test_alpha():

    matrix = np.array([
        [1.0, 0.5],
        [0.25, -0.25]
    ])

    alpha = calculate_alpha(
        matrix,
        1e-3
    )

    assert np.isfinite(
        alpha
    )

    assert alpha > 0


def test_correction_scaling():

    jacobian_matrix = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    alpha = calculate_alpha(
        jacobian_matrix / 4.0,
        1e-3
    )

    p_reference = 0.7931068990858002

    C_b = 1.0

    C_delta_x = calculate_correction_scale(
        alpha,
        C_b,
        p_reference,
        jacobian_matrix
    )

    normalized_correction = np.array([
        -1.0 / np.sqrt(2.0),
        -1.0 / np.sqrt(2.0)
    ])

    correction = scale_correction(
        normalized_correction,
        C_delta_x
    )

    expected = np.array([
        -1.0 / 6.0,
        -1.0 / 6.0
    ])

    assert np.allclose(
        correction,
        expected,
        atol=1e-8
    )


def test_quantum_newton_convergence():

    result = quantum_newton_method(
        F=F,
        jacobian=jacobian,
        x0=INITIAL_GUESS,
        tolerance=1e-8,
        max_iterations=50,
        shots=None,
        tomography_shots=10000,
        alpha_epsilon=1e-3
    )

    assert result[
        "final_residual"
    ] < 1e-6

    assert result[
        "iterations"
    ] <= 50

    assert np.all(
        np.isfinite(
            result["root"]
        )
    )