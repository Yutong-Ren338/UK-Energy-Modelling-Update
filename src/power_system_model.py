from typing import NamedTuple

import numpy as np
import pandas as pd

import src.assumptions as A
from src.units import Units as U

# Power System Model
# Models renewable energy generation, storage systems, demand response, and excess energy allocation
# Includes energy storage, Direct Air Capture (DAC), and curtailment strategies


class SimulationColumns(NamedTuple):
    """Container for power system simulation column names."""

    storage_level: str
    residual_energy: str
    dac_energy: str
    curtailed_energy: str
    stored_energy: str


class PowerSystemModel:
    """Comprehensive power system simulation model with configurable parameters.

    Models the interaction between renewable energy generation, energy storage systems,
    demand response, Direct Air Capture (DAC), and energy curtailment strategies.
    Can be extended to include additional power system components and control strategies.
    """

    def __init__(
        self,
        renewable_capacity: int,
        max_storage_capacity: float,
        electrolyser_power: float,
        dac_capacity: float,
        *,
        only_dac_if_storage_full: bool = True,
    ) -> None:
        """Initialize the power system model with required parameters.

        Args:
            renewable_capacity: Total renewable generation capacity in GW.
            max_storage_capacity: Maximum energy storage capacity in TWh.
            electrolyser_power: Power capacity for energy conversion to storage in GW.
            dac_capacity: Direct Air Capture system capacity in GW.
            only_dac_if_storage_full: Whether DAC only operates when storage is full.
        """
        # check pint units before running
        assert renewable_capacity.units == U.GW, "Renewable capacity must be in GW"
        assert max_storage_capacity.units == U.TWh, "Max storage capacity must be in TWh"
        assert electrolyser_power.units == U.GW, "Electrolyser power must be in GW"
        assert dac_capacity.units == U.GW, "DAC capacity must be in GW"

        self.renewable_capacity = renewable_capacity.magnitude

        # Use efficiency values from assumptions
        self.e_in = A.HydrogenStorage.Electrolysis.Efficiency
        self.e_out = A.HydrogenStorage.Generation.Efficiency

        # Set storage parameters (store as magnitudes)
        self.max_storage_capacity = max_storage_capacity.magnitude
        self.initial_storage_level = self.max_storage_capacity  # Start with full storage

        # Set electrolyser parameters (store as magnitudes)
        self.electrolyser_power = electrolyser_power.magnitude
        self.electrolyser_max_daily_energy = (electrolyser_power * A.HoursPerDay).to(U.TWh).magnitude

        # Set DAC parameters
        self.dac_capacity = dac_capacity.magnitude
        self.dac_max_daily_energy = (dac_capacity * A.HoursPerDay).to(U.TWh).magnitude
        self.only_dac_if_storage_full = only_dac_if_storage_full

    def _process_timestep(self, supply_demand: float, prev_storage: float) -> tuple[float, float, float, float, float]:
        """Process a single timestep of the simulation.

        Args:
            supply_demand: Energy supply minus demand for this timestep.
            prev_storage: Storage level from previous timestep.

        Returns:
            Tuple of (storage_level, residual_energy, dac_energy, curtailed_energy, stored_energy).
        """
        if supply_demand <= 0:
            # Energy shortage - draw from storage
            available_from_storage = prev_storage * self.e_out
            if supply_demand + available_from_storage <= 0:
                # Not enough storage to meet demand
                return 0.0, 0.0, 0.0, 0.0, 0.0

            # Partial storage draw
            energy_drawn = -supply_demand / self.e_out
            return prev_storage - energy_drawn, 0.0, 0.0, 0.0, 0.0

        # Energy surplus - store energy first, then allocate excess to DAC
        return self._process_energy_surplus_timestep(supply_demand, prev_storage)

    def _process_energy_surplus_timestep(self, supply_demand: float, prev_storage: float) -> tuple[float, float, float, float, float]:
        """Process a timestep with energy surplus.

        Args:
            supply_demand: Energy supply minus demand for this timestep.
            prev_storage: Storage level from previous timestep.

        Returns:
            Tuple of (storage_level, residual_energy, dac_energy, curtailed_energy, stored_energy).
        """
        energy_available_for_electrolyser = min(supply_demand, self.electrolyser_max_daily_energy)
        energy_to_store = energy_available_for_electrolyser * self.e_in

        if self.only_dac_if_storage_full:
            storage_space_available = self.max_storage_capacity - prev_storage

            if energy_to_store <= storage_space_available:
                # All energy can be stored
                energy_used_for_storage = energy_available_for_electrolyser
                return prev_storage + energy_to_store, 0.0, 0.0, 0.0, energy_used_for_storage

            # Storage gets filled, excess energy available for DAC
            energy_used_for_storage = storage_space_available / self.e_in
            residual_energy_val = supply_demand - energy_used_for_storage
            dac_energy_val = min(residual_energy_val, self.dac_max_daily_energy)

            return (self.max_storage_capacity, residual_energy_val, dac_energy_val, residual_energy_val - dac_energy_val, energy_used_for_storage)

        # Alternative allocation strategy
        new_storage_level = min(prev_storage + energy_to_store, self.max_storage_capacity)
        actual_energy_stored = (new_storage_level - prev_storage) / self.e_in
        residual_energy_val = supply_demand - actual_energy_stored
        dac_energy_val = min(residual_energy_val, self.dac_max_daily_energy) if residual_energy_val > 0 else 0.0

        return (new_storage_level, residual_energy_val, dac_energy_val, residual_energy_val - dac_energy_val, actual_energy_stored)

    def run_simulation(self, net_supply_df: pd.DataFrame) -> pd.DataFrame:
        """Run power system simulation for this renewable capacity scenario.

        Optimized vectorized version that avoids slow .loc assignments in loops.

        Args:
            net_supply_df: DataFrame containing supply-demand data.

        Returns:
            DataFrame with simulation results added as new columns.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = net_supply_df.copy()

        # Define column names for this renewable capacity scenario
        supply_demand_col = self.renewable_capacity
        if supply_demand_col not in df.columns:
            supply_demand_col = f"S-D(TWh),Ren={self.renewable_capacity}GW"

        columns = SimulationColumns(
            storage_level=f"storage_level (TWh),RC={self.renewable_capacity}GW",
            residual_energy=f"residual_energy (TWh),RC={self.renewable_capacity}GW",
            dac_energy=f"dac_energy (TWh),RC={self.renewable_capacity}GW",
            curtailed_energy=f"curtailed_energy (TWh),RC={self.renewable_capacity}GW",
            stored_energy=f"stored_energy (TWh),RC={self.renewable_capacity}GW",
        )

        # Get supply-demand values as numpy array for faster processing
        supply_demand_values = df[supply_demand_col].astype(float).to_numpy()
        n_timesteps = len(supply_demand_values)

        # Initialize result arrays
        results = np.zeros((n_timesteps, 5))  # storage, residual, dac, unused, stored

        # Process each timestep
        prev_storage = self.initial_storage_level

        for i in range(n_timesteps):
            storage_level, residual_energy, dac_energy, curtailed_energy, stored_energy = self._process_timestep(
                supply_demand_values[i], prev_storage
            )
            results[i] = [storage_level, residual_energy, dac_energy, curtailed_energy, stored_energy]
            prev_storage = storage_level

        # Assign results back to DataFrame with proper units
        df[columns.storage_level] = pd.Series(results[:, 0], dtype="pint[TWh]")
        df[columns.residual_energy] = pd.Series(results[:, 1], dtype="pint[TWh]")
        df[columns.dac_energy] = pd.Series(results[:, 2], dtype="pint[TWh]")
        df[columns.curtailed_energy] = pd.Series(results[:, 3], dtype="pint[TWh]")
        df[columns.stored_energy] = pd.Series(results[:, 4], dtype="pint[TWh]")

        # === VALIDATE RESULTS ===
        self._validate_simulation_results(df, columns)

        return df

    def _validate_simulation_results(self, df: pd.DataFrame, columns: SimulationColumns) -> None:
        """Validate simulation results to ensure physical constraints are met."""
        assert (df[columns.residual_energy] >= 0).all(), "Residual energy cannot be negative"
        assert (df[columns.curtailed_energy] >= 0).all(), "Unused energy cannot be negative"
        assert (df[columns.storage_level] <= self.max_storage_capacity * U.TWh).all(), "Storage levels cannot exceed maximum capacity"
        assert (df[columns.dac_energy] <= self.dac_max_daily_energy * U.TWh).all(), "DAC energy cannot exceed its maximum daily capacity"
        assert (df[columns.storage_level] >= 0).all(), "Storage levels cannot be negative"

    def analyze_simulation_results(self, net_supply_df: pd.DataFrame) -> dict:
        """Analyze simulation results and return key metrics.

        Args:
            net_supply_df: DataFrame containing simulation results.

        Returns:
            Dictionary containing analysis metrics.
        """
        # Define column names
        storage_column = f"storage_level (TWh),RC={int(self.renewable_capacity)}GW"
        dac_column = f"dac_energy (TWh),RC={int(self.renewable_capacity)}GW"
        unused_column = f"curtailed_energy (TWh),RC={int(self.renewable_capacity)}GW"

        # Calculate key metrics
        minimum_storage = net_supply_df[storage_column].min()
        annual_dac_energy = net_supply_df[dac_column].mean() * 365
        # Calculate capacity factor as actual usage vs maximum possible daily energy
        dac_capacity_factor = (net_supply_df[dac_column] > 0).mean()  # Simplified calculation based on operating days
        curtailed_energy = net_supply_df[unused_column].mean() * 365

        return {
            "minimum_storage": minimum_storage,
            "annual_dac_energy": annual_dac_energy,
            "dac_capacity_factor": dac_capacity_factor,
            "curtailed_energy": curtailed_energy,
        }

    @staticmethod
    def format_simulation_results(results: dict) -> str:
        """Return simulation results in a formatted way."""
        return (
            f"minimum storage is {results['minimum_storage']:.1f}\n"
            f"DAC energy is {results['annual_dac_energy']:.1f}\n"
            f"DAC Capacity Factor is {results['dac_capacity_factor']:.1%}\n"
            f"Curtailed energy is {results['curtailed_energy']:.1f}"
        )


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("tests/rei_net_supply_df_12gw_nuclear.csv")

    # Example: Run simulation with required parameters for 250 GW renewable capacity
    model = PowerSystemModel(
        renewable_capacity=250 * U.GW,
        max_storage_capacity=A.HydrogenStorage.CavernStorage.MaxCapacity,
        electrolyser_power=A.HydrogenStorage.Electrolysis.Power,
        dac_capacity=A.DAC.Capacity,
    )
    net_supply_df = model.run_simulation(df)
    results = model.analyze_simulation_results(net_supply_df)
    model.print_simulation_results(results)
