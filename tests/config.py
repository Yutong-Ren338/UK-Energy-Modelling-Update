import numpy as np

# Tolerances for floating point comparisons
ABSOLUTE_TOLERANCE = 1e-3
RELATIVE_TOLERANCE = 1e-3


def check(test: float, expected: float) -> None:
    assert np.isclose(test, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), f"Expected {expected:,}, got {test:,}"
