import matplotlib.pyplot as plt

from src.problems import F, jacobian, INITIAL_GUESS
from src.classical_newton import newton_method


def main():
    result = newton_method(
        F,
        jacobian,
        INITIAL_GUESS
    )
    history = result["history"]

    if not history:
        print("No iteration history was returned.")
        return

    iterations = [
        item["iteration"]
        for item in history
    ]

    residual_norms = [
        item["residual_norm"]
        for item in history
    ]

    correction_norms = [
        item["correction_norm"]
        for item in history
    ]
    final_solution = history[-1]["x"]

    print()
    
    print("Classical Newton Method Results")
    

    print("\nInitial guess:")
    print(INITIAL_GUESS)

    print("\nFinal solution:")
    print(final_solution)

    print("\nFinal residual norm:")
    print(f"{residual_norms[-1]:.6e}")

    print("\nNumber of iterations:")
    print(len(history))

    print("\nIteration History")
    print("-" * 70)

    print(
        f"{'Iteration':<12}"
        f"{'Residual Norm':<22}"
        f"{'Correction Norm':<22}"
    )

    print("-" * 70)

    for item in history:

        print(
            f"{item['iteration']:<12}"
            f"{item['residual_norm']:<22.6e}"
            f"{item['correction_norm']:<22.6e}"
        )

    print("-" * 70)

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        iterations,
        residual_norms,
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel(r"Residual Norm $\|F(x)\|$")
    plt.title("Classical Newton Method Convergence")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        iterations,
        correction_norms,
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel(r"Correction Norm $\|\Delta x\|$")
    plt.title("Newton Correction Convergence")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()