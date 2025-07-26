from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

import src.assumptions as A
from src.power_system_core import SimulationParameters, simulate_power_system_core
from src.units import Units as U

# Power System Model
# Models renewable energy generation, storage systems, demand response, and excess energy allocation
# Includes energy storage, Direct Air Capture (DAC), and curtailment strategies


class SimulationColumns(NamedTuple):
    """Container for power system simulation column names."""

    hydrogen_storage_level: str
    residual_energy: str
    dac_energy: str
    curtailed_energy: str
    stored_energy: str


class PowerSystem:
    """Comprehensive power system simulation with configurable parameters.

    Models the interaction between renewable energy generation, energy storage systems,
    demand response, Direct Air Capture (DAC), and energy curtailment strategies.
    Can be extended to include additional power system components and control strategies.
    """

    def __init__(
        self,
        renewable_capacity: int,
        max_hydrogen_storage_capacity: float,
        electrolyser_power: float,
        dac_capacity: float,
        *,
        only_dac_if_hydrogen_storage_full: bool = True,
    ) -> None:
        """Initialize the power system model with required parameters.

        Args:
            renewable_capacity: Total renewable generation capacity in GW.
            max_hydrogen_storage_capacity: Maximum hydrogen energy storage capacity in TWh.
            electrolyser_power: Power capacity for energy conversion to hydrogen storage in GW.
            dac_capacity: Direct Air Capture system capacity in GW.
            only_dac_if_hydrogen_storage_full: Whether DAC only operates when hydrogen storage is full.
        """
        # check pint units before running
        assert renewable_capacity.units == U.GW, "Renewable capacity must be in GW"
        assert max_hydrogen_storage_capacity.units == U.TWh, "Max hydrogen storage capacity must be in TWh"
        assert electrolyser_power.units == U.GW, "Electrolyser power must be in GW"
        assert dac_capacity.units == U.GW, "DAC capacity must be in GW"

        self.renewable_capacity = renewable_capacity.magnitude

        # Use efficiency values from assumptions
        self.hydrogen_e_in = A.HydrogenStorage.Electrolysis.Efficiency
        self.hydrogen_e_out = A.HydrogenStorage.Generation.Efficiency

        # Set hydrogen storage parameters (store as magnitudes)
        self.max_hydrogen_storage_capacity = max_hydrogen_storage_capacity.magnitude
        self.initial_hydrogen_storage_level = self.max_hydrogen_storage_capacity  # Start with full storage

        # Set electrolyser parameters (store as magnitudes)
        self.electrolyser_power = electrolyser_power.magnitude
        self.electrolyser_max_daily_energy = (electrolyser_power * A.HoursPerDay).to(U.TWh).magnitude

        # Set DAC parameters
        self.dac_capacity = dac_capacity.magnitude
        self.dac_max_daily_energy = (dac_capacity * A.HoursPerDay).to(U.TWh).magnitude
        self.only_dac_if_hydrogen_storage_full = only_dac_if_hydrogen_storage_full

    def run_simulation(self, net_supply_df: pd.DataFrame) -> pd.DataFrame | None:
        """Run power system simulation for this renewable capacity scenario.

        Uses the core simulation function for optimized processing.

        Args:
            net_supply_df: DataFrame containing supply-demand data.

        Returns:
            DataFrame with simulation results added as new columns, or None if
            simulation failed (storage capacity insufficient to meet demand).
        """
        # Create a copy to avoid modifying the original DataFrame
        df = net_supply_df.copy()

        # Define column names for this renewable capacity scenario
        supply_demand_col = self.renewable_capacity
        if supply_demand_col not in df.columns:
            supply_demand_col = f"S-D(TWh),Ren={self.renewable_capacity}GW"

        columns = SimulationColumns(
            hydrogen_storage_level=f"hydrogen_storage_level (TWh),RC={self.renewable_capacity}GW",
            residual_energy=f"residual_energy (TWh),RC={self.renewable_capacity}GW",
            dac_energy=f"dac_energy (TWh),RC={self.renewable_capacity}GW",
            curtailed_energy=f"curtailed_energy (TWh),RC={self.renewable_capacity}GW",
            stored_energy=f"stored_energy (TWh),RC={self.renewable_capacity}GW",
        )

        # Get supply-demand values as numpy array for faster processing
        supply_demand_values = df[supply_demand_col].astype(float).to_numpy()

        # Create simulation parameters
        params = SimulationParameters(
            initial_hydrogen_storage_level=self.initial_hydrogen_storage_level,
            max_hydrogen_storage_capacity=self.max_hydrogen_storage_capacity,
            electrolyser_max_daily_energy=self.electrolyser_max_daily_energy,
            dac_max_daily_energy=self.dac_max_daily_energy,
            hydrogen_e_in=self.hydrogen_e_in,
            hydrogen_e_out=self.hydrogen_e_out,
            only_dac_if_hydrogen_storage_full=self.only_dac_if_hydrogen_storage_full,
        )

        # Run the core simulation
        results = simulate_power_system_core(supply_demand_values, params)

        # Check if simulation failed (storage hit zero)
        if np.isnan(results).any():
            return None

        # Assign results back to DataFrame with proper units
        df[columns.hydrogen_storage_level] = pd.Series(results[:, 0], dtype="pint[TWh]")
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
        assert (df[columns.hydrogen_storage_level] <= self.max_hydrogen_storage_capacity * U.TWh).all(), (
            "Hydrogen storage cannot exceed maximum capacity"
        )
        assert (df[columns.dac_energy] <= self.dac_max_daily_energy * U.TWh).all(), "DAC energy cannot exceed its maximum daily capacity"
        assert (df[columns.hydrogen_storage_level] >= 0).all(), "Hydrogen storage cannot be negative"

    def analyze_simulation_results(self, sim_df: pd.DataFrame) -> dict | None:
        """Analyze simulation results and return key metrics.

        Args:
            sim_df: DataFrame containing simulation results.

        Returns:
            Dictionary containing analysis metrics, or None if simulation failed.
        """
        # Check if this is a failed simulation (None DataFrame)
        if sim_df is None:
            return None

        # Define column names
        hydrogen_storage_column = f"hydrogen_storage_level (TWh),RC={int(self.renewable_capacity)}GW"
        dac_column = f"dac_energy (TWh),RC={int(self.renewable_capacity)}GW"
        unused_column = f"curtailed_energy (TWh),RC={int(self.renewable_capacity)}GW"

        # Calculate key metrics
        minimum_hydrogen_storage = sim_df[hydrogen_storage_column].min()
        annual_dac_energy = sim_df[dac_column].mean() * 365
        # Calculate capacity factor as actual usage vs maximum possible daily energy
        dac_capacity_factor = (sim_df[dac_column] > 0).mean()  # Simplified calculation based on operating days
        curtailed_energy = sim_df[unused_column].mean() * 365

        return {
            "minimum_hydrogen_storage": minimum_hydrogen_storage,
            "annual_dac_energy": annual_dac_energy,
            "dac_capacity_factor": dac_capacity_factor,
            "curtailed_energy": curtailed_energy,
        }

    @staticmethod
    def format_simulation_results(results: dict) -> str:
        """Return simulation results in a formatted way."""
        return (
            f"minimum hydrogen storage is {results['minimum_hydrogen_storage']:.1f}\n"
            f"DAC energy is {results['annual_dac_energy']:.1f}\n"
            f"DAC Capacity Factor is {results['dac_capacity_factor']:.1%}\n"
            f"Curtailed energy is {results['curtailed_energy']:.1f}"
        )

    def print_simulation_results(self, results: dict | None) -> None:
        """Print simulation results in a formatted way.

        Args:
            results: Dictionary containing analysis metrics from analyze_simulation_results,
                    or None if simulation failed.
        """
        if results is None:
            print("Simulation failed: insufficient storage capacity to meet demand")
        else:
            print(self.format_simulation_results(results))

    def plot_simulation_results(self, sim_df: pd.DataFrame | None, results: dict | None, demand_mode: str, fname: str | None = None) -> None:
        """Plot simulation results showing storage levels and energy flows.

        Args:
            sim_df: DataFrame containing simulation results, or None if simulation failed.
            results: Dictionary containing analysis metrics from analyze_simulation_results,
                    or None if simulation failed.
            demand_mode: Label for the demand scenario.
            fname: Optional filename to save the plot.

        """
        if sim_df is None or results is None:
            print(f"Cannot plot results: simulation failed for {demand_mode} demand scenario")
            return

        fig = plt.figure(figsize=(15, 6))

        # Create gridspec: 2 rows, 4 columns (3 for left plots, 1 for right text)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.0, wspace=0.1)

        # Left plots take first 3 columns
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(
            sim_df[f"hydrogen_storage_level (TWh),RC={self.renewable_capacity}GW"],
            color="green",
            linewidth=0.5,
            label="Energy in Hydrogen Storage",
        )
        ax1.axhline(
            self.max_hydrogen_storage_capacity,
            linestyle="--",
            color="red",
            linewidth=1.5,
            label="Maximum Hydrogen Storage Capacity",
        )
        ax1.axhline(20, linestyle="--", color="blue", linewidth=1.5, label="Contingency Storage")
        ax1.set_ylim(0, self.max_hydrogen_storage_capacity * 1.1)
        ax1.set_ylabel("Hydrogen Storage Level (TWh)")
        ax1.legend(loc="upper right", fontsize=10, facecolor="white", edgecolor="gray", frameon=True, framealpha=0.9)

        ax2 = fig.add_subplot(gs[1, :3])
        ax2.plot(sim_df[f"stored_energy (TWh),RC={self.renewable_capacity}GW"], color="green", linewidth=0.5, label="Stored Energy")
        ax2.plot(sim_df[f"curtailed_energy (TWh),RC={self.renewable_capacity}GW"], color="black", linewidth=0.5, label="Curtailed Energy")
        ax2.plot(sim_df[f"dac_energy (TWh),RC={self.renewable_capacity}GW"], color="red", linewidth=0.5, label="DAC Energy")
        ax2.set_xlabel("Day in 40 Years")
        ax2.set_ylabel("Energy (TWh)")
        ax2.legend(loc="upper right", fontsize=10, facecolor="white", edgecolor="gray", frameon=True, framealpha=0.9)

        # Text subplot spans both rows in the rightmost column
        ax3 = fig.add_subplot(gs[:, 3])
        ax3.axis("off")

        # Create formatted text with parameters and results
        text = (
            f"Parameters:\n"
            f"• Demand Mode: {demand_mode}\n"
            f"• Renewable Capacity: {self.renewable_capacity:.0f} GW\n"
            f"• Hydrogen Storage Capacity: {self.max_hydrogen_storage_capacity:.0f} TWh\n"
            f"• DAC Capacity: {self.dac_capacity:.0f} GW\n"
            f"• Electrolyser Power: {self.electrolyser_power:.0f} GW\n\n"
            f"Results:\n"
            f"{self.format_simulation_results(results)}"
        )

        ax3.text(0, 0.5, text, fontsize=11, verticalalignment="center", fontfamily="monospace")

        if fname:
            fig.savefig(fname, bbox_inches="tight", dpi=300)
