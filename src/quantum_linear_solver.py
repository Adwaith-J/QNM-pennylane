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
    padded[:len(vector)] = vector

    return padded


def prepare_rhs_state(b: np.ndarray):
    b_normalized, norm_b = normalize_rhs(b)

    padded_b = pad_to_power_of_two(b_normalized)
    padded_b = padded_b / np.linalg.norm(padded_b)

    return padded_b, norm_b


def build_Ob(b: np.ndarray):
    return prepare_rhs_state(b)


def build_OA1(A: np.ndarray):
    A = np.asarray(A, dtype=float)

    column_indices = {}

    for j in range(A.shape[0]):
        column_indices[j] = np.flatnonzero(
            ~np.isclose(A[j], 0.0)
        ).tolist()

    return column_indices


def build_OA2(A: np.ndarray):
    A = np.asarray(A, dtype=float)

    def oracle(j: int, k: int):
        if j < 0 or j >= A.shape[0]:
            raise IndexError("Row index out of range.")

        if k < 0 or k >= A.shape[1]:
            raise IndexError("Column index out of range.")

        return A[j, k]

    return oracle


def finite_difference_jacobian(
    F,
    x: np.ndarray,
    step: float = 1e-6
):
    x = np.asarray(x, dtype=float)
    fx = np.asarray(F(x), dtype=float)

    if fx.ndim != 1:
        raise ValueError("F(x) must return a one-dimensional vector.")

    if step <= 0:
        raise ValueError("step must be positive.")

    J = np.zeros((len(fx), len(x)), dtype=float)

    for k in range(len(x)):
        x_perturbed = x.copy()
        x_perturbed[k] += step

        f_perturbed = np.asarray(
            F(x_perturbed),
            dtype=float
        )

        J[:, k] = (
            f_perturbed - fx
        ) / step

    return J


def classical_solution(A: np.ndarray, b: np.ndarray):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.linalg.solve(A, b)


def run_rhs_preparation_circuit(
    rhs_state: np.ndarray,
    shots: int | None = None
):
    rhs_state = np.asarray(rhs_state, dtype=float)

    n_qubits = number_of_qubits(len(rhs_state))

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
        shots=shots
    )

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(
            rhs_state,
            wires=range(n_qubits),
            normalize=False
        )

        return qml.state()

    return circuit()


def qlss_solver(
    A: np.ndarray,
    rhs_state: np.ndarray,
    shots: int | None = None
):
    raise NotImplementedError(
        "QLSS implementation is pending. "
        "Do not use np.linalg.solve() here."
    )


def solve_fixed_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    shots: int | None = None,
    run_qlss: bool = False
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

    rhs_state, b_scale = build_Ob(b)

    OA1 = build_OA1(normalized_A)
    OA2 = build_OA2(normalized_A)

    result = {
        "matrix": A,
        "normalized_matrix": normalized_A,
        "matrix_scale": matrix_scale,
        "rhs": b,
        "normalized_rhs": rhs_state,
        "rhs_scale": b_scale,
        "dimension": A.shape[0],
        "qubits": number_of_qubits(A.shape[0]),
        "OA1": OA1,
        "OA2": OA2,
        "quantum_solution_state": None,
        "classical_reference_solution": classical_solution(A, b),
        "shots": shots,
    }

    if run_qlss:
        quantum_solution = qlss_solver(
            normalized_A,
            rhs_state,
            shots=shots
        )

        result["quantum_solution_state"] = quantum_solution

        classical = result["classical_reference_solution"]
        classical = classical / np.linalg.norm(classical)

        quantum = quantum_solution[:len(classical)]
        quantum = quantum / np.linalg.norm(quantum)

        result["state_error_l2"] = np.linalg.norm(
            quantum - classical
        )

    return result


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
        shots=None,
        run_qlss=False
    )

    print("\nNormalized matrix:")
    print(result["normalized_matrix"])

    print("\nNormalized RHS:")
    print(result["normalized_rhs"])

    print("\nO_A1:")
    print(result["OA1"])

    print("\nO_A2 values:")

    for j in range(J.shape[0]):
        for k in range(J.shape[1]):
            print(
                f"O_A2({j}, {k}) = "
                f"{result['OA2'](j, k)}"
            )

    print("\nClassical reference solution:")
    print(result["classical_reference_solution"])