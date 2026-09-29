import numpy as np


def F(x: np.ndarray) -> np.ndarray:
    x1, x2 = x

    return np.array([
        x1**2 + x2**2 - 4.0,
        x1 - x2 - 1.0
    ])


def jacobian(x: np.ndarray) -> np.ndarray:
    x1, x2 = x

    return np.array([
        [2.0 * x1, 2.0 * x2],
        [1.0, -1.0]
    ])


KNOWN_ROOT = np.array([
    (1.0 + np.sqrt(7.0)) / 2.0,
    (-1.0 + np.sqrt(7.0)) / 2.0
])

INITIAL_GUESS = np.array([2.0, 1.0])