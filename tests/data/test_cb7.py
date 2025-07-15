# filepath: /Users/sam/work/mzt/UK-Energy-Modelling/tests/data/test_cb7.py
from src.data.cb7 import frac_heat_demand_from_buildings

EXPECTED_FRACTION = 0.5968718478706598
TOLERANCE = 1e-10


def test_frac_heat_demand_from_buildings() -> None:
    """Test that the function returns the expected fraction value."""
    result = frac_heat_demand_from_buildings()

    # Allow for small floating point differences
    assert abs(result - EXPECTED_FRACTION) < TOLERANCE, f"Expected {EXPECTED_FRACTION}, got {result}"

    # Also check that the result is a reasonable fraction (between 0 and 1)
    assert 0 <= result <= 1, f"Result should be between 0 and 1, got {result}"
