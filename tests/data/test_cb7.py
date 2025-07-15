from pint import Quantity

from src import assumptions
from src.data import cb7
from tests.config import check

EXPECTED_FRACTION = assumptions.CB7FractionHeatDemandBuildings
EXPECTED_BUILDINGS_DEMAND = assumptions.CB7EnergyDemand2050Buildings
EXPECTED_TOTAL_DEMAND = assumptions.CB7EnergyDemand2050


def test_frac_heat_demand_from_buildings() -> None:
    """Test that the function returns the expected fraction value."""
    result = cb7.frac_heat_demand_from_buildings()

    # Use check function for float comparison
    check(result, EXPECTED_FRACTION)

    # Also check that the result is a reasonable fraction (between 0 and 1)
    assert 0 <= result <= 1, f"Result should be between 0 and 1, got {result}"


def test_buildings_electricity_demand() -> None:
    """Test that the function returns the expected buildings electricity demand."""
    result = cb7.buildings_electricity_demand()

    # Check that the result is a Pint quantity with TWh units
    assert isinstance(result, Quantity), f"Result should be a Pint quantity, got {type(result)}"
    assert str(result.units) == "terawatt_hour", f"Result should have TWh units, got {result.units}"

    # Use check function for float comparison
    check(result, EXPECTED_BUILDINGS_DEMAND)

    # Check that the result is positive
    assert result > 0, f"Result should be positive, got {result}"


def test_total_demand_2050() -> None:
    """Test that the function returns the expected total electricity demand for 2050."""
    result = cb7.total_demand_2050()

    # Check that the result is a Pint quantity with TWh units
    assert isinstance(result, Quantity), f"Result should be a Pint quantity, got {type(result)}"
    assert str(result.units) == "terawatt_hour", f"Result should have TWh units, got {result.units}"

    # Use check function for float comparison
    check(result, EXPECTED_TOTAL_DEMAND)

    # Check that the result is positive
    assert result > 0, f"Result should be positive, got {result}"

    # Check that total demand is greater than buildings demand
    buildings_demand = cb7.buildings_electricity_demand()
    assert result > buildings_demand, f"Total demand ({result}) should be greater than buildings demand ({buildings_demand})"
