"""Tests for the storage model simulation."""

from pathlib import Path

import pandas as pd
import pytest
from pint import Quantity

from src.storage_model import DAC_MAX_DAILY_ENERGY, STORAGE_MAX_CAPACITY, analyze_simulation_results, run_simulation
from src.units import Units as U
from tests.config import check


class TestStorageModel:
    """Test class for storage model functionality."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Load the test data for simulations.

        Returns:
            DataFrame containing test simulation data.
        """
        test_data_path = Path(__file__).parent / "rei_net_supply_df_12gw_nuclear.csv"
        return pd.read_csv(test_data_path)

    def test_run_simulation_with_expected_outputs(self, sample_data: pd.DataFrame) -> None:
        """Test that the simulation produces expected outputs for the standard test case."""
        # Run the simulation
        net_supply_df = run_simulation(sample_data, renewable_capacity=250)

        # Analyze results
        results = analyze_simulation_results(net_supply_df, renewable_capacity=250)

        # Check that the expected outputs match the documented values
        # (with some tolerance for floating point precision)
        expected_values = {
            "minimum_storage": 20.16927245757229 * U.TWh,
            "annual_dac_energy": 38.47911516786211 * U.TWh,
            "dac_capacity_factor": 0.19,  # 19.0%
            "annual_unused_energy": 46.846621471892654 * U.TWh,
        }
        check(results["minimum_storage"], expected_values["minimum_storage"])
        check(results["annual_dac_energy"], expected_values["annual_dac_energy"])
        check(results["dac_capacity_factor"], expected_values["dac_capacity_factor"])
        check(results["annual_unused_energy"], expected_values["annual_unused_energy"])

    def test_simulation_creates_expected_columns(self, sample_data: pd.DataFrame) -> None:
        """Test that the simulation creates the expected columns in the output DataFrame."""
        net_supply_df = run_simulation(sample_data, renewable_capacity=250)

        # Check that expected columns exist for 250GW renewable capacity
        expected_columns = ["L (TWh),RC=250GW", "R_ccs (TWh),RC=250GW", "R_dac (TWh),RC=250GW", "R_unused (TWh),RC=250GW"]

        for col in expected_columns:
            assert col in net_supply_df.columns, f"Expected column {col} not found"

    def test_simulation_physical_constraints(self, sample_data: pd.DataFrame) -> None:
        """Test that simulation results satisfy physical constraints."""
        net_supply_df = run_simulation(sample_data, renewable_capacity=250)

        # Check storage level constraints
        storage_col = "L (TWh),RC=250GW"
        assert (net_supply_df[storage_col] >= 0).all(), "Storage levels cannot be negative"
        assert (net_supply_df[storage_col] <= STORAGE_MAX_CAPACITY).all(), "Storage levels cannot exceed maximum capacity"

        # Check that residual energies are non-negative
        residual_col = "R_ccs (TWh),RC=250GW"
        dac_col = "R_dac (TWh),RC=250GW"
        unused_col = "R_unused (TWh),RC=250GW"

        assert (net_supply_df[residual_col] >= 0).all(), "Residual energy cannot be negative"
        assert (net_supply_df[dac_col] >= 0).all(), "DAC energy cannot be negative"
        assert (net_supply_df[unused_col] >= 0).all(), "Unused energy cannot be negative"

        # Check DAC capacity constraint
        assert (net_supply_df[dac_col] <= DAC_MAX_DAILY_ENERGY).all(), "DAC energy cannot exceed daily capacity"

    def test_analyze_simulation_results_structure(self, sample_data: pd.DataFrame) -> None:
        """Test that analyze_simulation_results returns expected structure."""
        net_supply_df = run_simulation(sample_data, renewable_capacity=250)
        results = analyze_simulation_results(net_supply_df, renewable_capacity=250)

        # Check that all expected keys are present
        expected_keys = {"minimum_storage", "annual_dac_energy", "dac_capacity_factor", "annual_unused_energy"}

        assert set(results.keys()) == expected_keys, "Results dictionary missing expected keys"

        # Check value types and ranges
        assert isinstance(results["minimum_storage"], Quantity)
        assert isinstance(results["annual_dac_energy"], Quantity)
        assert isinstance(results["dac_capacity_factor"], float)
        assert isinstance(results["annual_unused_energy"], Quantity)

        # Check capacity factor is a valid percentage
        assert 0 <= results["dac_capacity_factor"] <= 1, "DAC capacity factor should be between 0 and 1"

    def test_simulation_with_custom_renewable_capacity(self, sample_data: pd.DataFrame) -> None:
        """Test that analyze_simulation_results works with custom renewable capacity."""
        # Test with different renewable capacities
        net_supply_df = run_simulation(sample_data, renewable_capacity=300)
        results = analyze_simulation_results(net_supply_df, renewable_capacity=300)

        assert results is not None
        assert isinstance(results, dict)

        # Test that the simulation creates the correct columns for custom capacity
        expected_columns = ["L (TWh),RC=300GW", "R_ccs (TWh),RC=300GW", "R_dac (TWh),RC=300GW", "R_unused (TWh),RC=300GW"]
        for col in expected_columns:
            assert col in net_supply_df.columns, f"Expected column {col} not found"

    def test_multiple_renewable_capacities(self, sample_data: pd.DataFrame) -> None:
        """Test that the function works correctly when called multiple times with different capacities."""
        capacities = [200, 250, 300]
        all_results = {}

        for capacity in capacities:
            net_supply_df = run_simulation(sample_data, renewable_capacity=capacity)
            results = analyze_simulation_results(net_supply_df, renewable_capacity=capacity)
            all_results[capacity] = results

            # Verify that each capacity produces valid results
            assert results["minimum_storage"] >= 0
            assert results["annual_dac_energy"] >= 0
            assert 0 <= results["dac_capacity_factor"] <= 1
            assert results["annual_unused_energy"] >= 0

        # Verify that different capacities produce different results
        assert len(set(r["minimum_storage"] for r in all_results.values())) > 1, "Different capacities should produce different results"
