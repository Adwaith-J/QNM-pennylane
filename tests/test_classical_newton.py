import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import numpy as np

from src.problems import F, jacobian, INITIAL_GUESS, KNOWN_ROOT
from src.classical_newton import newton_method


def test_classical_newton_converges():

    result = newton_method(
        F,
        jacobian,
        INITIAL_GUESS
    )

    history = result["history"]

    final_solution = history[-1]["x"]

    np.testing.assert_allclose(
        final_solution,
        KNOWN_ROOT,
        atol=1e-6
    )

    assert history[-1]["residual_norm"] < 1e-6


def test_residual_decreases():

    result = newton_method(
        F,
        jacobian,
        INITIAL_GUESS
    )

    history = result["history"]

    residuals = [
        item["residual_norm"]
        for item in history
    ]

    assert residuals[-1] < residuals[0]


if __name__ == "__main__":
    test_classical_newton_converges()
    test_residual_decreases()

    print("All tests passed.")