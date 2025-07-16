from typing import NamedTuple

import pandas as pd

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
STORAGE_MAX_CAPACITY = 71.0  # TWh
INITIAL_STORAGE_LEVEL = STORAGE_MAX_CAPACITY  # Start with full storage

# Electrolyser system (converts excess energy to stored energy)
ELECTROLYSER_POWER = 50  # GW
ELECTROLYSER_MAX_DAILY_ENERGY = ELECTROLYSER_POWER * 24 / 1000  # TWh per day

# Direct Air Capture system
DAC_CAPACITY_GW = 27.0  # GW
DAC_MAX_DAILY_ENERGY = DAC_CAPACITY_GW * 24 / 1000  # TWh per day

# Renewable capacity scenarios to analyze
RENEWABLE_CAPACITIES = [250]  # GW
ONLY_DAC_IF_STORAGE_FULL = True


def run_simulation(net_supply_df: pd.DataFrame, renewable_capacity: int) -> pd.DataFrame:
    """
    Run energy storage simulation for a single renewable capacity scenario.

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

    # Initialize result columns
    df[columns.storage_level] = 0.0
    df[columns.residual_energy] = 0.0
    df[columns.dac_energy] = 0.0
    df[columns.unused_energy] = 0.0

    # === PROCESS ALL TIME STEPS ===
    for i in range(len(df)):
        supply_demand = df.loc[i, supply_demand_col]
        prev_storage = INITIAL_STORAGE_LEVEL if i == 0 else df.loc[i - 1, columns.storage_level]

        if supply_demand <= 0:
            # Energy shortage - draw from storage
            _process_energy_shortage(df, i, supply_demand, prev_storage, columns.storage_level)
        else:
            # Energy surplus - store energy first, then allocate excess to DAC
            _process_energy_surplus(df, i, supply_demand, prev_storage, columns)

    # === VALIDATE RESULTS ===
    _validate_simulation_results(df, columns)

    return df


def _process_energy_shortage(df: pd.DataFrame, i: int, supply_demand: float, prev_storage: float, storage_level_col: str) -> None:
    """Process time step with energy shortage (negative supply-demand)."""
    available_from_storage = prev_storage * e_out

    if supply_demand + available_from_storage <= 0:
        # Not enough storage to meet demand
        df.loc[i, storage_level_col] = 0.0
    else:
        # Partial storage draw
        energy_drawn = -supply_demand / e_out
        df.loc[i, storage_level_col] = prev_storage - energy_drawn


def _process_energy_surplus(df: pd.DataFrame, i: int, supply_demand: float, prev_storage: float, columns: SimulationColumns) -> None:
    """Process time step with energy surplus (positive supply-demand)."""
    energy_available_for_electrolyser = min(supply_demand, ELECTROLYSER_MAX_DAILY_ENERGY)
    energy_to_store = energy_available_for_electrolyser * e_in

    if ONLY_DAC_IF_STORAGE_FULL:
        storage_space_available = STORAGE_MAX_CAPACITY - prev_storage

        if energy_to_store <= storage_space_available:
            # All energy can be stored
            df.loc[i, columns.storage_level] = prev_storage + energy_to_store
            df.loc[i, columns.residual_energy] = 0.0
            df.loc[i, columns.dac_energy] = 0.0
            df.loc[i, columns.unused_energy] = 0.0
        else:
            # Storage gets filled, excess energy available for DAC
            energy_used_for_storage = storage_space_available / e_in
            residual_energy = supply_demand - energy_used_for_storage
            dac_energy = min(residual_energy, DAC_MAX_DAILY_ENERGY)

            df.loc[i, columns.storage_level] = STORAGE_MAX_CAPACITY
            df.loc[i, columns.residual_energy] = residual_energy
            df.loc[i, columns.dac_energy] = dac_energy
            df.loc[i, columns.unused_energy] = residual_energy - dac_energy
    else:
        # Alternative allocation strategy
        new_storage_level = min(prev_storage + energy_to_store, STORAGE_MAX_CAPACITY)
        actual_energy_stored = (new_storage_level - prev_storage) / e_in
        residual_energy = supply_demand - actual_energy_stored
        dac_energy = min(residual_energy, DAC_MAX_DAILY_ENERGY) if residual_energy > 0 else 0.0

        df.loc[i, columns.storage_level] = new_storage_level
        df.loc[i, columns.residual_energy] = residual_energy
        df.loc[i, columns.dac_energy] = dac_energy
        df.loc[i, columns.unused_energy] = residual_energy - dac_energy


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
    if renewable_capacity is None:
        renewable_capacity = RENEWABLE_CAPACITIES[0]

    # Define column names
    storage_column = f"L (TWh),RC={renewable_capacity}GW"
    dac_column = f"R_dac (TWh),RC={renewable_capacity}GW"
    unused_column = f"R_unused (TWh),RC={renewable_capacity}GW"

    # Calculate key metrics
    min_storage = net_supply_df[storage_column].min()
    annual_dac_energy = net_supply_df[dac_column].mean() * 365
    dac_capacity_factor = (net_supply_df[dac_column] > 0).mean()
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
