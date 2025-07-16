from typing import NamedTuple

import numpy as np
import pandas as pd

import src.assumptions as A
from src.units import Units as U

# Energy Storage and DAC (Direct Air Capture) Simulation
# Models energy storage filling/emptying and allocation of excess energy to DAC


class SimulationColumns(NamedTuple):
    """Container for simulation column names."""

    storage_level: str
    residual_energy: str
    dac_energy: str
    unused_energy: str


e_in = 0.74  # electrolysis
e_out = 0.55  # electricity generation from hydrogen

# === SYSTEM PARAMETERS ===
# Storage system
STORAGE_MAX_CAPACITY = 71.0 * U.TWh
INITIAL_STORAGE_LEVEL = STORAGE_MAX_CAPACITY  # Start with full storage

# Electrolyser system (converts excess energy to stored energy)
ELECTROLYSER_POWER = 50 * U.GW
ELECTROLYSER_MAX_DAILY_ENERGY = (ELECTROLYSER_POWER * A.HoursPerDay).to(U.TWh)

# Direct Air Capture system
DAC_CAPACITY_GW = 27.0 * U.GW
DAC_MAX_DAILY_ENERGY = (DAC_CAPACITY_GW * A.HoursPerDay).to(U.TWh)

# Allow DAC and Electrolysis to operate on the same day
ONLY_DAC_IF_STORAGE_FULL = True

# use floats for comparisons for speed
STORAGE_MAX_CAPACITY_MAG = STORAGE_MAX_CAPACITY.magnitude
INITIAL_STORAGE_LEVEL_MAG = INITIAL_STORAGE_LEVEL.magnitude
ELECTROLYSER_POWER_MAG = ELECTROLYSER_POWER.magnitude
ELECTROLYSER_MAX_DAILY_ENERGY_MAG = ELECTROLYSER_MAX_DAILY_ENERGY.magnitude
DAC_CAPACITY_GW_MAG = DAC_CAPACITY_GW.magnitude
DAC_MAX_DAILY_ENERGY_MAG = DAC_MAX_DAILY_ENERGY.magnitude


def _process_timestep(supply_demand: float, prev_storage: float) -> tuple[float, float, float, float]:
    """
    Process a single timestep of the simulation.

    Args:
        supply_demand: Energy supply minus demand for this timestep
        prev_storage: Storage level from previous timestep

    Returns:
        Tuple of (storage_level, residual_energy, dac_energy, unused_energy)
    """
    if supply_demand <= 0:
        # Energy shortage - draw from storage
        available_from_storage = prev_storage * e_out
        if supply_demand + available_from_storage <= 0:
            # Not enough storage to meet demand
            return 0.0, 0.0, 0.0, 0.0

        # Partial storage draw
        energy_drawn = -supply_demand / e_out
        return prev_storage - energy_drawn, 0.0, 0.0, 0.0

    # Energy surplus - store energy first, then allocate excess to DAC
    return _process_energy_surplus_timestep(supply_demand, prev_storage)


def _process_energy_surplus_timestep(supply_demand: float, prev_storage: float) -> tuple[float, float, float, float]:
    """
    Process a timestep with energy surplus.

    Args:
        supply_demand: Energy supply minus demand for this timestep
        prev_storage: Storage level from previous timestep

    Returns:
        Tuple of (storage_level, residual_energy, dac_energy, unused_energy)
    """
    energy_available_for_electrolyser = min(supply_demand, ELECTROLYSER_MAX_DAILY_ENERGY_MAG)
    energy_to_store = energy_available_for_electrolyser * e_in

    if ONLY_DAC_IF_STORAGE_FULL:
        storage_space_available = STORAGE_MAX_CAPACITY_MAG - prev_storage

        if energy_to_store <= storage_space_available:
            # All energy can be stored
            return prev_storage + energy_to_store, 0.0, 0.0, 0.0

        # Storage gets filled, excess energy available for DAC
        energy_used_for_storage = storage_space_available / e_in
        residual_energy_val = supply_demand - energy_used_for_storage
        dac_energy_val = min(residual_energy_val, DAC_MAX_DAILY_ENERGY_MAG)

        return (STORAGE_MAX_CAPACITY_MAG, residual_energy_val, dac_energy_val, residual_energy_val - dac_energy_val)

    # Alternative allocation strategy
    new_storage_level = min(prev_storage + energy_to_store, STORAGE_MAX_CAPACITY_MAG)
    actual_energy_stored = (new_storage_level - prev_storage) / e_in
    residual_energy_val = supply_demand - actual_energy_stored
    dac_energy_val = min(residual_energy_val, DAC_MAX_DAILY_ENERGY_MAG) if residual_energy_val > 0 else 0.0

    return (new_storage_level, residual_energy_val, dac_energy_val, residual_energy_val - dac_energy_val)


def run_simulation(net_supply_df: pd.DataFrame, renewable_capacity: int) -> pd.DataFrame:
    """
    Run energy storage simulation for a single renewable capacity scenario.

    Optimized vectorized version that avoids slow .loc assignments in loops.

    Args:
        net_supply_df: DataFrame containing supply-demand data
        renewable_capacity: Renewable capacity in GW

    Returns:
        DataFrame with simulation results added as new columns
    """
    # Create a copy to avoid modifying the original DataFrame
    df = net_supply_df.copy()

    # Define column names for this renewable capacity scenario
    supply_demand_col = renewable_capacity
    if supply_demand_col not in df.columns:
        supply_demand_col = f"S-D(TWh),Ren={renewable_capacity}GW"

    columns = SimulationColumns(
        storage_level=f"L (TWh),RC={renewable_capacity}GW",
        residual_energy=f"R_ccs (TWh),RC={renewable_capacity}GW",
        dac_energy=f"R_dac (TWh),RC={renewable_capacity}GW",
        unused_energy=f"R_unused (TWh),RC={renewable_capacity}GW",
    )

    # Get supply-demand values as numpy array for faster processing
    supply_demand_values = df[supply_demand_col].astype(float).to_numpy()
    n_timesteps = len(supply_demand_values)

    # Initialize result arrays
    results = np.zeros((n_timesteps, 4))  # storage, residual, dac, unused

    # Process each timestep
    prev_storage = INITIAL_STORAGE_LEVEL_MAG

    for i in range(n_timesteps):
        storage_level, residual_energy, dac_energy, unused_energy = _process_timestep(supply_demand_values[i], prev_storage)
        results[i] = [storage_level, residual_energy, dac_energy, unused_energy]
        prev_storage = storage_level

    # Assign results back to DataFrame with proper units
    df[columns.storage_level] = pd.Series(results[:, 0], dtype="pint[TWh]")
    df[columns.residual_energy] = pd.Series(results[:, 1], dtype="pint[TWh]")
    df[columns.dac_energy] = pd.Series(results[:, 2], dtype="pint[TWh]")
    df[columns.unused_energy] = pd.Series(results[:, 3], dtype="pint[TWh]")

    # === VALIDATE RESULTS ===
    _validate_simulation_results(df, columns)

    return df


def _validate_simulation_results(df: pd.DataFrame, columns: SimulationColumns) -> None:
    """Validate simulation results to ensure physical constraints are met."""
    assert (df[columns.residual_energy] >= 0).all(), "Residual energy cannot be negative"
    assert (df[columns.unused_energy] >= 0).all(), "Unused energy cannot be negative"
    assert (df[columns.storage_level] <= STORAGE_MAX_CAPACITY).all(), "Storage levels cannot exceed maximum capacity"
    assert (df[columns.dac_energy] <= DAC_MAX_DAILY_ENERGY).all(), "DAC energy cannot exceed its maximum daily capacity"
    assert (df[columns.storage_level] >= 0).all(), "Storage levels cannot be negative"


def analyze_simulation_results(net_supply_df: pd.DataFrame, renewable_capacity: int | None = None) -> dict:
    """
    Analyze simulation results and return key metrics.

    Args:
        net_supply_df: DataFrame containing simulation results
        renewable_capacity: Renewable capacity in GW (defaults to first capacity in RENEWABLE_CAPACITIES)

    Returns:
        Dictionary containing analysis metrics
    """

    # Define column names
    storage_column = f"L (TWh),RC={int(renewable_capacity)}GW"
    dac_column = f"R_dac (TWh),RC={int(renewable_capacity)}GW"
    unused_column = f"R_unused (TWh),RC={int(renewable_capacity)}GW"

    # Calculate key metrics
    min_storage = net_supply_df[storage_column].min()
    annual_dac_energy = net_supply_df[dac_column].mean() * 365
    # Calculate capacity factor as actual usage vs maximum possible daily energy
    dac_capacity_factor = (net_supply_df[dac_column] > 0).mean()  # TODO: improve this calculation
    annual_unused_energy = net_supply_df[unused_column].mean() * 365

    return {
        "minimum_storage": min_storage,
        "annual_dac_energy": annual_dac_energy,
        "dac_capacity_factor": dac_capacity_factor,
        "annual_unused_energy": annual_unused_energy,
    }


def print_simulation_results(results: dict) -> None:
    """Print simulation results in a formatted way."""
    print(f"minimum storage is {results['minimum_storage']}")
    print(f"DAC energy is {results['annual_dac_energy']}")
    print(f"DAC Capacity Factor is {results['dac_capacity_factor']:.1%}")
    print(f"Curtailed energy is {results['annual_unused_energy']}")


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("tests/rei_net_supply_df_12gw_nuclear.csv")

    # Run simulation for each renewable capacity
    for capacity in RENEWABLE_CAPACITIES:
        print(f"\n=== Renewable Capacity: {capacity} GW ===")
        net_supply_df = run_simulation(df, capacity)
        results = analyze_simulation_results(net_supply_df, capacity)
        print_simulation_results(results)
