import numpy as np
import pennylane as qml


def normalize_rhs(b: np.ndarray):
    b = np.asarray(b, dtype=float)

    norm_b = np.linalg.norm(b, ord=2)

    if np.isclose(norm_b, 0.0):
        raise ValueError("The RHS vector cannot have zero norm.")

    return b / norm_b, norm_b


def normalize_matrix(A: np.ndarray):
    A = np.asarray(A, dtype=float)

    matrix_scale = np.max(np.abs(A))

    if np.isclose(matrix_scale, 0.0):
        raise ValueError("The matrix cannot be the zero matrix.")

    return A / matrix_scale, matrix_scale


def number_of_qubits(dimension: int) -> int:
    if dimension < 1:
        raise ValueError("Dimension must be positive.")

    return int(np.ceil(np.log2(dimension)))


def pad_to_power_of_two(vector: np.ndarray):
    vector = np.asarray(vector, dtype=float)

    n_qubits = number_of_qubits(len(vector))
    target_dimension = 2 ** n_qubits

    if len(vector) == target_dimension:
        return vector.copy()

    padded = np.zeros(target_dimension, dtype=float)
    padded[: len(vector)] = vector

    return padded


def prepare_rhs_state(b: np.ndarray):
    b_normalized, norm_b = normalize_rhs(b)

    padded_b = pad_to_power_of_two(b_normalized)

    
    padded_b = padded_b / np.linalg.norm(padded_b)

    return padded_b, norm_b


def classical_solution(A: np.ndarray, b: np.ndarray):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.linalg.solve(A, b)


def normalized_solution(A: np.ndarray, b: np.ndarray):
    solution = classical_solution(A, b)

    norm_solution = np.linalg.norm(solution, ord=2)

    if np.isclose(norm_solution, 0.0):
        raise ValueError("Solution has zero norm.")

    return solution / norm_solution, norm_solution


def finite_size_quantum_solution_state(A: np.ndarray, b: np.ndarray):
    normalized_x, solution_norm = normalized_solution(A, b)

    padded_x = pad_to_power_of_two(normalized_x)

    padded_x = padded_x / np.linalg.norm(padded_x)

    return padded_x, normalized_x, solution_norm


def run_quantum_solution_circuit(
    solution_state: np.ndarray,
    shots: int | None = None
):
    solution_state = np.asarray(solution_state, dtype=float)

    n_qubits = number_of_qubits(len(solution_state))

    dev = qml.device(
    "default.qubit",
    wires=n_qubits
    )

    @qml.qnode(dev, shots=shots)
    def circuit():
        qml.AmplitudeEmbedding(
            solution_state,
            wires=range(n_qubits),
            normalize=False
        )

        return qml.probs(wires=range(n_qubits))

    return circuit()


def reconstruct_amplitudes_from_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    return np.sqrt(np.maximum(probabilities, 0.0))


def solve_fixed_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    shots: int | None = None
):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a matrix.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if len(b) != A.shape[0]:
        raise ValueError(
            "The RHS dimension must match the matrix dimension."
        )

    normalized_A, matrix_scale = normalize_matrix(A)

    normalized_b, b_scale = prepare_rhs_state(b)

    n_qubits = number_of_qubits(A.shape[0])

    quantum_state, normalized_x, solution_norm = (
        finite_size_quantum_solution_state(A, b)
    )

    probabilities = run_quantum_solution_circuit(
        quantum_state,
        shots=shots
    )

    measured_amplitudes = reconstruct_amplitudes_from_probabilities(
        probabilities
    )
    
    signs = np.sign(normalized_x)

    
    signs[np.isclose(normalized_x, 0.0)] = 0.0

    measured_solution = measured_amplitudes[: len(normalized_x)] * signs

    measured_norm = np.linalg.norm(measured_solution)

    if not np.isclose(measured_norm, 0.0):
        measured_solution = measured_solution / measured_norm

    state_error = np.linalg.norm(
        measured_solution - normalized_x,
        ord=2
    )

    recovered_delta_x = measured_solution * solution_norm

    exact_delta_x = classical_solution(A, b)

    correction_error = np.linalg.norm(
        recovered_delta_x - exact_delta_x,
        ord=2
    )

    return {
        "matrix": A,
        "normalized_matrix": normalized_A,
        "matrix_scale": matrix_scale,
        "rhs": b,
        "normalized_rhs": normalized_b,
        "rhs_scale": b_scale,
        "dimension": A.shape[0],
        "qubits": n_qubits,
        "exact_solution": exact_delta_x,
        "solution_norm": solution_norm,
        "normalized_classical_solution": normalized_x,
        "quantum_state": quantum_state,
        "probabilities": probabilities,
        "measured_solution": measured_solution,
        "recovered_delta_x": recovered_delta_x,
        "state_error_l2": state_error,
        "correction_error_l2": correction_error,
        "shots": shots,
    }


if __name__ == "__main__":

    J = np.array([
        [4.0, 2.0],
        [1.0, -1.0]
    ])

    b = np.array([
        -1.0,
        0.0
    ])

    result = solve_fixed_linear_system(
        J,
        b,
        shots=None
    )

    print("\n=== QNM Finite-Size Quantum Linear-System Prototype ===")

    print("\nOriginal matrix J:")
    print(result["matrix"])

    print("\nNormalized matrix A:")
    print(result["normalized_matrix"])

    print(
        "\nMatrix normalization constant:",
        result["matrix_scale"]
    )

    print("\nOriginal RHS b:")
    print(result["rhs"])

    print("\nNormalized RHS |b>:")
    print(result["normalized_rhs"])

    print(
        "\nRHS normalization constant:",
        result["rhs_scale"]
    )

    print("\nNumber of qubits:")
    print(result["qubits"])

    print("\nExact classical correction:")
    print(result["exact_solution"])

    print("\nNormalized classical solution state:")
    print(result["normalized_classical_solution"])

    print("\nQuantum state:")
    print(result["quantum_state"])

    print("\nMeasured probabilities:")
    print(result["probabilities"])

    print("\nRecovered normalized solution:")
    print(result["measured_solution"])

    print("\nRecovered Delta x:")
    print(result["recovered_delta_x"])

    print(
        "\nNormalized-state L2 error:",
        result["state_error_l2"]
    )

    print(
        "\nCorrection-vector L2 error:",
        result["correction_error_l2"]
    )