"""Tests for the power system model simulation."""

from pathlib import Path

import pandas as pd
import pytest
from pint import Quantity

import src.assumptions as A
from src.power_system_model import PowerSystemModel
from src.units import Units as U
from tests.config import check

SIMULATION_KWARGS = {
    "renewable_capacity": 250 * U.GW,  # Default renewable capacity for the simulation
    "max_storage_capacity": A.HydrogenStorage.CavernStorage.MaxCapacity,  # Maximum storage capacity
    "electrolyser_power": A.HydrogenStorage.Electrolysis.Power,  # Electrolyser power capacity
    "dac_capacity": A.DAC.Capacity,  # DAC capacity
}


@pytest.fixture
def sample_data() -> pd.DataFrame:
    test_data_path = Path(__file__).parent / "rei_net_supply_df_12gw_nuclear.csv"
    return pd.read_csv(test_data_path)


@pytest.fixture
def power_system_model() -> PowerSystemModel:
    return PowerSystemModel(**SIMULATION_KWARGS)


def test_run_simulation_with_expected_outputs(power_system_model: PowerSystemModel, sample_data: pd.DataFrame) -> None:
    # Run the simulation
    net_supply_df = power_system_model.run_simulation(sample_data)

    # Analyze results
    results = power_system_model.analyze_simulation_results(net_supply_df)

    # Check that the expected outputs match the documented values
    # (with some tolerance for floating point precision)
    expected_values = {
        "minimum_storage": 20.16927245757229 * U.TWh,
        "annual_dac_energy": 38.47911516786211 * U.TWh,
        "dac_capacity_factor": 0.19,  # 19.0%
        "curtailed_energy": 46.846621471892654 * U.TWh,
    }
    check(results["minimum_storage"], expected_values["minimum_storage"])
    check(results["annual_dac_energy"], expected_values["annual_dac_energy"])
    check(results["dac_capacity_factor"], expected_values["dac_capacity_factor"])
    check(results["curtailed_energy"], expected_values["curtailed_energy"])


def test_simulation_creates_expected_columns(power_system_model: PowerSystemModel, sample_data: pd.DataFrame) -> None:
    net_supply_df = power_system_model.run_simulation(sample_data)

    # Check that expected columns exist for 250GW renewable capacity
    expected_columns = [
        "storage_level (TWh),RC=250GW",
        "residual_energy (TWh),RC=250GW",
        "dac_energy (TWh),RC=250GW",
        "curtailed_energy (TWh),RC=250GW",
    ]

    for col in expected_columns:
        assert col in net_supply_df.columns, f"Expected column {col} not found"


def test_simulation_physical_constraints(power_system_model: PowerSystemModel, sample_data: pd.DataFrame) -> None:
    net_supply_df = power_system_model.run_simulation(sample_data)

    # Check storage level constraints
    storage_col = "storage_level (TWh),RC=250GW"
    assert (net_supply_df[storage_col] >= 0).all(), "Storage levels cannot be negative"
    assert (net_supply_df[storage_col] <= power_system_model.max_storage_capacity * U.TWh).all(), "Storage levels cannot exceed maximum capacity"

    # Check that residual energies are non-negative
    residual_col = "residual_energy (TWh),RC=250GW"
    dac_col = "dac_energy (TWh),RC=250GW"
    unused_col = "curtailed_energy (TWh),RC=250GW"

    assert (net_supply_df[residual_col] >= 0).all(), "Residual energy cannot be negative"
    assert (net_supply_df[dac_col] >= 0).all(), "DAC energy cannot be negative"
    assert (net_supply_df[unused_col] >= 0).all(), "Unused energy cannot be negative"

    # Check DAC capacity constraint
    assert (net_supply_df[dac_col] <= power_system_model.dac_max_daily_energy * U.TWh).all(), "DAC energy cannot exceed daily capacity"


def test_analyze_simulation_results_structure(power_system_model: PowerSystemModel, sample_data: pd.DataFrame) -> None:
    net_supply_df = power_system_model.run_simulation(sample_data)
    results = power_system_model.analyze_simulation_results(net_supply_df)

    # Check that all expected keys are present
    expected_keys = {"minimum_storage", "annual_dac_energy", "dac_capacity_factor", "curtailed_energy"}

    assert set(results.keys()) == expected_keys, "Results dictionary missing expected keys"

    # Check value types and ranges
    assert isinstance(results["minimum_storage"], Quantity)
    assert isinstance(results["annual_dac_energy"], Quantity)
    assert isinstance(results["dac_capacity_factor"], float)
    assert isinstance(results["curtailed_energy"], Quantity)

    # Check capacity factor is a valid percentage
    assert 0 <= results["dac_capacity_factor"] <= 1, "DAC capacity factor should be between 0 and 1"


def test_simulation_with_custom_renewable_capacity(sample_data: pd.DataFrame) -> None:
    # Test with different renewable capacities
    custom_model = PowerSystemModel(
        renewable_capacity=300 * U.GW,
        max_storage_capacity=A.HydrogenStorage.CavernStorage.MaxCapacity,
        electrolyser_power=A.HydrogenStorage.Electrolysis.Power,
        dac_capacity=A.DAC.Capacity,
    )
    net_supply_df = custom_model.run_simulation(sample_data)
    results = custom_model.analyze_simulation_results(net_supply_df)

    assert results is not None
    assert isinstance(results, dict)

    # Test that the simulation creates the correct columns for custom capacity
    expected_columns = [
        "storage_level (TWh),RC=300GW",
        "residual_energy (TWh),RC=300GW",
        "dac_energy (TWh),RC=300GW",
        "curtailed_energy (TWh),RC=300GW",
    ]
    for col in expected_columns:
        assert col in net_supply_df.columns, f"Expected column {col} not found"


def test_multiple_renewable_capacities(sample_data: pd.DataFrame) -> None:
    capacities = [200, 250, 300]
    all_results = {}

    for capacity in capacities:
        model = PowerSystemModel(
            renewable_capacity=capacity * U.GW,
            max_storage_capacity=A.HydrogenStorage.CavernStorage.MaxCapacity,
            electrolyser_power=A.HydrogenStorage.Electrolysis.Power,
            dac_capacity=A.DAC.Capacity,
        )
        net_supply_df = model.run_simulation(sample_data)
        results = model.analyze_simulation_results(net_supply_df)
        all_results[capacity] = results

        # Verify that each capacity produces valid results
        assert results["minimum_storage"] >= 0 * U.TWh
        assert results["annual_dac_energy"] >= 0 * U.TWh
        assert 0 <= results["dac_capacity_factor"] <= 1
        assert results["curtailed_energy"] >= 0 * U.TWh

    # Verify that different capacities produce different results
    assert len({r["minimum_storage"] for r in all_results.values()}) > 1, "Different capacities should produce different results"
