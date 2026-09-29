import numpy as np

from src.problems import (
    F,
    jacobian,
    KNOWN_ROOT,
    INITIAL_GUESS
)

print("Initial guess:")
print(INITIAL_GUESS)

print("\nF(initial guess):")
print(F(INITIAL_GUESS))

print("\nJacobian(initial guess):")
print(jacobian(INITIAL_GUESS))

print("\nKnown root:")
print(KNOWN_ROOT)

print("\nF(known root):")
print(F(KNOWN_ROOT))

print("\nResidual at root:")
print(np.linalg.norm(F(KNOWN_ROOT)))