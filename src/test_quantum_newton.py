import numpy as np

from src.problems import F, jacobian, INITIAL_GUESS
from src.quantum_newton import quantum_newton_method


for shots in [None, 100, 500, 1000]:

    result = quantum_newton_method(
        F=F,
        jacobian=jacobian,
        x0=INITIAL_GUESS,
        tolerance=1e-8,
        max_iterations=50,
        shots=shots
    )

    print("\n==============================")
    print(f"Shots: {shots}")
    print("==============================")

    print("Root:")
    print(result["root"])

    print("Final residual:")
    print(result["final_residual"])

    print("Iterations:")
    print(result["iterations"])

    print("Execution time:")
    print(result["execution_time"])

    print("\nIteration errors:")

    for item in result["history"]:
        print(
            f"Iteration {item['iteration']} | "
            f"Residual = {item['residual_norm']:.6e} | "
            f"Correction Error = "
            f"{item['correction_error']:.6e}"
        )