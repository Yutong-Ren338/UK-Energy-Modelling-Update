from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from pint import Quantity

import src.assumptions as A
from src.costs import energy_cost, total_system_cost
from src.power_system_core import SimulationParameters, simulate_power_system_core
from src.units import Units as U

# Power System Model
# Models renewable energy generation, storage systems, demand response, and excess energy allocation
# Includes energy storage, Direct Air Capture (DAC), and curtailment strategies


class SimulationColumns(NamedTuple):
    """Container for power system simulation column names."""

    medium_storage_level: str
    hydrogen_storage_level: str
    dac_energy: str
    curtailed_energy: str
    energy_into_medium_storage: str
    energy_into_hydrogen_storage: str
    gas_ccs_energy: str


class PowerSystem:
    """Comprehensive power system simulation with configurable parameters.

    Models the interaction between renewable energy generation, energy storage systems,
    demand response, Direct Air Capture (DAC), and energy curtailment strategies.
    Can be extended to include additional power system components and control strategies.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        renewable_capacity: Quantity,
        hydrogen_storage_capacity: Quantity,
        electrolyser_power: Quantity,
        dac_capacity: Quantity,
        hydrogen_generation_power: Quantity | None = None,
        medium_storage_capacity: Quantity | None = None,
        medium_storage_power: Quantity | None = None,
        gas_ccs_capacity: Quantity | None = None,
        only_dac_if_hydrogen_storage_full: bool = True,
    ) -> None:
        """Initialize the power system model with required parameters.

        Args:
            renewable_capacity: Total renewable generation capacity in GW.
            hydrogen_storage_capacity: Maximum hydrogen energy storage capacity in TWh.
            electrolyser_power: Power capacity for energy conversion to hydrogen storage in GW.
            dac_capacity: Direct Air Capture system capacity in GW.
            hydrogen_generation_power: Power capacity for generating electricity from hydrogen in GW. If None, uses default from assumptions.
            medium_storage_capacity: Maximum medium-term storage capacity in TWh. If None, uses default from assumptions.
            medium_storage_power: Medium-term storage power capacity in GW. If None, uses default from assumptions.
            gas_ccs_capacity: Dispatchable gas CCS capacity in GW. If None, uses default from assumptions.
            only_dac_if_hydrogen_storage_full: Whether DAC only operates when hydrogen storage is full.
        """
        # check pint units before running
        assert renewable_capacity.units == U.GW, "Renewable capacity must be in GW"
        assert hydrogen_storage_capacity.units == U.TWh, "Max hydrogen storage capacity must be in TWh"
        assert electrolyser_power.units == U.GW, "Electrolyser power must be in GW"
        assert dac_capacity.units == U.GW, "DAC capacity must be in GW"

        if hydrogen_generation_power is None:
            hydrogen_generation_power = A.HydrogenStorage.Generation.Power
        assert hydrogen_generation_power.units == U.GW, "Hydrogen generation power must be in GW"

        # Set medium-term storage parameters with defaults from assumptions
        if medium_storage_capacity is None:
            medium_storage_capacity = A.MediumTermStorage.Capacity
        if medium_storage_power is None:
            medium_storage_power = A.MediumTermStorage.Power

        assert medium_storage_capacity.units == U.TWh, "Medium storage capacity must be in TWh"
        assert medium_storage_power.units == U.GW, "Medium storage power must be in GW"

        # Set gas CCS parameters with defaults from assumptions
        if gas_ccs_capacity is None:
            gas_ccs_capacity = A.DispatchableGasCCS.Capacity

        assert gas_ccs_capacity.units == U.GW, "Gas CCS capacity must be in GW"

        if medium_storage_capacity == 0:
            assert medium_storage_power == 0, "If medium storage capacity is zero, power must also be zero"

        self.renewable_capacity = renewable_capacity.magnitude

        # Use efficiency values from assumptions
        self.hydrogen_e_in = A.HydrogenStorage.Electrolysis.Efficiency
        self.hydrogen_e_out = A.HydrogenStorage.Generation.Efficiency

        # Set hydrogen storage parameters (store as magnitudes)
        self.hydrogen_storage_capacity = hydrogen_storage_capacity.magnitude
        self.initial_hydrogen_storage_level = self.hydrogen_storage_capacity  # Start with full storage

        # Set hydrogen generation parameters
        self.hydrogen_generation_power = hydrogen_generation_power.magnitude
        self.hydrogen_generation_max_daily_energy = (hydrogen_generation_power * A.HoursPerDay).to(U.TWh).magnitude

        # Set medium-term storage parameters (store as magnitudes)
        self.medium_storage_capacity = medium_storage_capacity.magnitude
        self.medium_storage_power = medium_storage_power.magnitude
        self.medium_storage_max_daily_energy = (medium_storage_power * A.HoursPerDay).to(U.TWh).magnitude
        self.medium_storage_efficiency = np.sqrt(A.MediumTermStorage.RoundTripEfficiency)  # Convert round-trip to single-direction efficiency
        self.initial_medium_storage_level = self.medium_storage_capacity  # Start with full storage

        # Set electrolyser parameters (store as magnitudes)
        self.electrolyser_power = electrolyser_power.magnitude
        self.electrolyser_max_daily_energy = (electrolyser_power * A.HoursPerDay).to(U.TWh).magnitude

        # Set DAC parameters
        self.dac_capacity = dac_capacity.magnitude
        self.dac_max_daily_energy = (dac_capacity * A.HoursPerDay).to(U.TWh).magnitude
        self.only_dac_if_hydrogen_storage_full = only_dac_if_hydrogen_storage_full

        # Set gas CCS parameters
        self.gas_ccs_capacity = gas_ccs_capacity.magnitude
        self.gas_ccs_max_daily_energy = (gas_ccs_capacity * A.HoursPerDay).to(U.TWh).magnitude

    def run_simulation(self, net_supply_df: pd.DataFrame) -> pd.DataFrame | None:
        """Run power system simulation for this renewable capacity scenario.

        Uses the core simulation function for optimized processing.

        Args:
            net_supply_df: DataFrame containing supply-demand data.

        Returns:
            DataFrame with simulation results, or None if
            simulation failed (storage capacity insufficient to meet demand).
        """
        # Define column names for this renewable capacity scenario
        supply_demand_col = self.renewable_capacity
        if supply_demand_col not in net_supply_df.columns:
            supply_demand_col = f"S-D(TWh),Ren={self.renewable_capacity}GW"

        columns = SimulationColumns(
            medium_storage_level=f"medium_storage_level (TWh),RC={self.renewable_capacity}GW",
            hydrogen_storage_level=f"hydrogen_storage_level (TWh),RC={self.renewable_capacity}GW",
            dac_energy=f"dac_energy (TWh),RC={self.renewable_capacity}GW",
            curtailed_energy=f"curtailed_energy (TWh),RC={self.renewable_capacity}GW",
            energy_into_medium_storage=f"energy_into_medium_storage (TWh),RC={self.renewable_capacity}GW",
            energy_into_hydrogen_storage=f"energy_into_hydrogen_storage (TWh),RC={self.renewable_capacity}GW",
            gas_ccs_energy=f"gas_ccs_energy (TWh),RC={self.renewable_capacity}GW",
        )

        # Get supply-demand values as numpy array for faster processing
        supply_demand_values = net_supply_df[supply_demand_col].astype(float).to_numpy()

        # Create simulation parameters
        params = SimulationParameters(
            initial_hydrogen_storage_level=self.initial_hydrogen_storage_level,
            hydrogen_storage_capacity=self.hydrogen_storage_capacity,
            electrolyser_max_daily_energy=self.electrolyser_max_daily_energy,
            hydrogen_generation_max_daily_energy=self.hydrogen_generation_max_daily_energy,
            dac_max_daily_energy=self.dac_max_daily_energy,
            hydrogen_e_in=self.hydrogen_e_in,
            hydrogen_e_out=self.hydrogen_e_out,
            only_dac_if_hydrogen_storage_full=self.only_dac_if_hydrogen_storage_full,
            initial_medium_storage_level=self.initial_medium_storage_level,
            medium_storage_capacity=self.medium_storage_capacity,
            medium_storage_max_daily_energy=self.medium_storage_max_daily_energy,
            medium_storage_efficiency=self.medium_storage_efficiency,
            gas_ccs_max_daily_energy=self.gas_ccs_max_daily_energy,
        )

        # Run the core simulation
        results = simulate_power_system_core(supply_demand_values, params)

        # Check if simulation failed (storage hit zero)
        if np.isnan(results).any():
            return None

        # Create new results DataFrame with proper units
        results_df = pd.DataFrame(
            {
                columns.medium_storage_level: pd.Series(results[:, 0], dtype="pint[TWh]"),
                columns.hydrogen_storage_level: pd.Series(results[:, 1], dtype="pint[TWh]"),
                columns.dac_energy: pd.Series(results[:, 2], dtype="pint[TWh]"),
                columns.curtailed_energy: pd.Series(results[:, 3], dtype="pint[TWh]"),
                columns.energy_into_medium_storage: pd.Series(results[:, 4], dtype="pint[TWh]"),
                columns.energy_into_hydrogen_storage: pd.Series(results[:, 5], dtype="pint[TWh]"),
                columns.gas_ccs_energy: pd.Series(results[:, 6], dtype="pint[TWh]"),
            },
            index=net_supply_df.index,
        )

        # === VALIDATE RESULTS ===
        self._validate_simulation_results(results_df, columns)

        return results_df

    def _validate_simulation_results(self, df: pd.DataFrame, columns: SimulationColumns) -> None:
        """Validate simulation results to ensure physical constraints are met."""
        assert (df[columns.curtailed_energy] >= 0).all(), "Unused energy cannot be negative"
        assert (df[columns.hydrogen_storage_level] <= self.hydrogen_storage_capacity * U.TWh).all(), "Hydrogen storage cannot exceed maximum capacity"
        assert (df[columns.medium_storage_level] <= self.medium_storage_capacity * U.TWh).all(), "Medium storage cannot exceed maximum capacity"
        assert (df[columns.dac_energy] <= self.dac_max_daily_energy * U.TWh).all(), "DAC energy cannot exceed its maximum daily capacity"
        assert (df[columns.hydrogen_storage_level] >= 0).all(), "Hydrogen storage cannot be negative"
        assert (df[columns.medium_storage_level] >= 0).all(), "Medium storage cannot be negative"
        assert (df[columns.gas_ccs_energy] >= 0).all(), "Gas CCS energy cannot be negative"
        assert (df[columns.gas_ccs_energy] <= self.gas_ccs_max_daily_energy * U.TWh).all(), "Gas CCS energy cannot exceed its maximum daily capacity"

    def analyze_simulation_results(self, sim_df: pd.DataFrame | None) -> dict | None:
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
        medium_storage_column = f"medium_storage_level (TWh),RC={int(self.renewable_capacity)}GW"
        hydrogen_storage_column = f"hydrogen_storage_level (TWh),RC={int(self.renewable_capacity)}GW"
        dac_column = f"dac_energy (TWh),RC={int(self.renewable_capacity)}GW"
        unused_column = f"curtailed_energy (TWh),RC={int(self.renewable_capacity)}GW"
        gas_ccs_column = f"gas_ccs_energy (TWh),RC={int(self.renewable_capacity)}GW"

        # Calculate key metrics
        minimum_medium_storage = sim_df[medium_storage_column].min()
        minimum_hydrogen_storage = sim_df[hydrogen_storage_column].min()
        annual_dac_energy = sim_df[dac_column].mean() * 365
        # Calculate CO2 removals using pre-calculated DAC energy cost conversion
        annual_co2_removals = annual_dac_energy / A.DAC.EnergyCost.MediumTWhPerMtCO2
        # Calculate capacity factor as actual usage vs maximum possible daily energy
        dac_capacity_factor = (sim_df[dac_column] > 0).mean()  # Simplified calculation based on operating days
        curtailed_energy = sim_df[unused_column].mean() * 365
        annual_gas_ccs_energy = sim_df[gas_ccs_column].mean() * 365
        gas_ccs_capacity_factor = (sim_df[gas_ccs_column] > 0).mean()  # Simplified calculation based on operating days

        return {
            "minimum_medium_storage": minimum_medium_storage,
            "minimum_hydrogen_storage": minimum_hydrogen_storage,
            "annual_dac_energy": annual_dac_energy,
            "annual_co2_removals": annual_co2_removals,
            "dac_capacity_factor": dac_capacity_factor,
            "curtailed_energy": curtailed_energy,
            "annual_gas_ccs_energy": annual_gas_ccs_energy,
            "gas_ccs_capacity_factor": gas_ccs_capacity_factor,
        }

    def calculate_power_system_cost(self, sim_df: pd.DataFrame | None = None) -> Quantity:
        """Calculate the total cost of the power system including operational costs.

        Uses the total_system_cost function from costs.py with parameters from the power system
        and assumptions.py for any missing values. If simulation results are provided, adds
        operational costs for gas CCS and medium-term storage based on energy usage.

        Args:
            sim_df: Optional simulation results DataFrame to calculate operational costs.

        Returns:
            Total system cost in GBP.
        """
        # Calculate base system cost (capital costs)
        base_cost = total_system_cost(
            energy_demand=A.EnergyDemand2050,
            renewable_capacity=self.renewable_capacity * U.GW,
            renewable_capacity_factor=A.Renewables.AverageCapacityFactor,
            renewable_lcoe=A.Renewables.AverageLCOE,
            nuclear_capacity=A.Nuclear.Capacity,
            nuclear_capacity_factor=A.Nuclear.CapacityFactor,
            nuclear_lcoe=A.Nuclear.AverageLCOE,
            storage_capacity=self.hydrogen_storage_capacity * U.TWh,
            electrolyser_power=self.electrolyser_power * U.GW,
            generation_capacity=self.hydrogen_generation_power * U.GW,
        )

        # If no simulation results provided, return base cost only
        if sim_df is None:
            return base_cost

        # Calculate additional operational costs based on energy usage
        additional_costs = 0 * U.GBP

        # Gas CCS operational cost
        gas_ccs_column = f"gas_ccs_energy (TWh),RC={int(self.renewable_capacity)}GW"
        annual_gas_ccs_energy = sim_df[gas_ccs_column].mean() * 365  # Convert daily average to annual
        gas_ccs_cost = annual_gas_ccs_energy * A.DispatchableGasCCS.LCOE
        additional_costs += gas_ccs_cost

        # Medium-term storage operational cost (based on energy throughput)
        medium_storage_column = f"energy_into_medium_storage (TWh),RC={int(self.renewable_capacity)}GW"
        annual_medium_storage_energy = sim_df[medium_storage_column].mean() * 365  # Convert daily average to annual
        medium_storage_cost = annual_medium_storage_energy * A.MediumTermStorage.LCOE
        additional_costs += medium_storage_cost

        return base_cost + additional_costs

    def calculate_energy_cost(self, sim_df: pd.DataFrame | None = None) -> Quantity:
        return energy_cost(self.calculate_power_system_cost(sim_df), A.EnergyDemand2050)

    @staticmethod
    def format_simulation_results(results: dict) -> str:
        """Return simulation results in a formatted way."""
        return (
            f"• Minimum medium storage: {results['minimum_medium_storage']:~0.1f}\n"
            f"• Minimum hydrogen storag:  {results['minimum_hydrogen_storage']:~0.1f}\n"
            f"• DAC energy: {results['annual_dac_energy']:~0.1f}\n"
            f"• DAC CO2 removals: {results['annual_co2_removals']:~0.1f}\n"
            f"• DAC Capacity Factor: {results['dac_capacity_factor']:.1%}\n"
            f"• Gas CCS energy: {results['annual_gas_ccs_energy']:~0.1f}\n"
            f"• Gas CCS Capacity Factor: {results['gas_ccs_capacity_factor']:.1%}\n"
            f"• Curtailed energy: {results['curtailed_energy']:~0.1f}"
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

        # Top plot: Combined storage as percentage filled
        ax1 = fig.add_subplot(gs[0, :3])

        # Calculate percentage filled for both storage types
        medium_storage_pct = (
            (sim_df[f"medium_storage_level (TWh),RC={self.renewable_capacity}GW"] / self.medium_storage_capacity * 100)
            if self.medium_storage_capacity > 0
            else pd.Series([0] * len(sim_df))
        )
        hydrogen_storage_pct = sim_df[f"hydrogen_storage_level (TWh),RC={self.renewable_capacity}GW"] / self.hydrogen_storage_capacity * 100

        ax1.plot(
            medium_storage_pct,
            color="orange",
            linewidth=0.8,
            label="Medium-term Storage",
        )
        ax1.plot(
            hydrogen_storage_pct,
            color="green",
            linewidth=0.8,
            label="Hydrogen Storage",
        )

        ax1.set_ylim(0, 110)
        ax1.set_ylabel("Storage Level (%)")
        ax1.legend(loc="upper right", fontsize=10, facecolor="white", edgecolor="gray", frameon=True, framealpha=0.9)

        # Bottom plot: Energy flows
        ax2 = fig.add_subplot(gs[1, :3])
        ax2.plot(sim_df[f"curtailed_energy (TWh),RC={self.renewable_capacity}GW"], color="black", linewidth=0.5, label="Curtailed Energy")
        ax2.plot(
            sim_df[f"energy_into_hydrogen_storage (TWh),RC={self.renewable_capacity}GW"],
            color="green",
            linewidth=0.5,
            label="Hydrogen Storage",
        )
        ax2.plot(
            sim_df[f"gas_ccs_energy (TWh),RC={self.renewable_capacity}GW"],
            color="purple",
            linewidth=0.5,
            label="Gas CCS",
        )
        ax2.plot(
            sim_df[f"energy_into_medium_storage (TWh),RC={self.renewable_capacity}GW"],
            color="orange",
            linewidth=0.5,
            label="Medium Storage",
        )
        ax2.plot(sim_df[f"dac_energy (TWh),RC={self.renewable_capacity}GW"], color="red", linewidth=0.5, label="DAC Energy")
        ax2.set_xlabel("Day in 40 Years")
        ax2.set_ylabel("Energy (TWh)")
        ax2.legend(loc="upper right", fontsize=10, facecolor="white", edgecolor="gray", frameon=True, framealpha=0.9)

        # Text subplot spans all rows in the rightmost column
        ax3 = fig.add_subplot(gs[:, 3])
        ax3.axis("off")

        # Create formatted text with parameters and results
        text = (
            f"Parameters:\n"
            f"• Demand Mode: {demand_mode}\n"
            f"• Renewables: {self.renewable_capacity:.0f} GW\n"
            f"• Medium Storage: {self.medium_storage_capacity:.1f} TWh, {self.medium_storage_power:.0f} GW\n"
            f"• Hydrogen Storage: {self.hydrogen_storage_capacity:.0f} TWh, {self.electrolyser_power:.0f} GW\n"
            f"• Gas CCS: {self.gas_ccs_capacity:.0f} GW\n"
            f"• DAC: {self.dac_capacity:.1f} GW\n"
            f"• Energy cost: {self.calculate_energy_cost(sim_df):~0.1f}\n"
            f"\nResults:\n"
            f"{self.format_simulation_results(results)}"
        )

        ax3.text(0, 0.5, text, fontsize=11, verticalalignment="center", fontfamily="monospace")

        if fname:
            fig.savefig(fname, bbox_inches="tight", dpi=300)
