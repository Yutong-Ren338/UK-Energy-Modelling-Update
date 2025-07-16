import pandas as pd

# Energy Storage and DAC (Direct Air Capture) Simulation
# Models energy storage filling/emptying and allocation of excess energy to DAC

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


def run_simulation(net_supply_df: pd.DataFrame) -> None:
    # === MAIN SIMULATION LOOP ===
    for capacity in RENEWABLE_CAPACITIES:
        # Define column names for this renewable capacity scenario
        supply_demand_col = capacity
        if supply_demand_col not in net_supply_df.columns:  # temporary fix for loading Rei's df
            supply_demand_col = f"S-D(TWh),Ren={capacity}GW"
        storage_level_col = f"L (TWh),RC={capacity}GW"
        residual_energy_col = f"R_ccs (TWh),RC={capacity}GW"
        dac_energy_col = f"R_dac (TWh),RC={capacity}GW"
        unused_energy_col = f"R_unused (TWh),RC={capacity}GW"

        # Initialize result columns
        net_supply_df[storage_level_col] = 0.0
        net_supply_df[residual_energy_col] = 0.0
        net_supply_df[dac_energy_col] = 0.0
        net_supply_df[unused_energy_col] = 0.0

        # === PROCESS ALL TIME STEPS ===
        for i in range(len(net_supply_df)):
            supply_demand = net_supply_df.loc[i, supply_demand_col]
            prev_storage = INITIAL_STORAGE_LEVEL if i == 0 else net_supply_df.loc[i - 1, storage_level_col]
            if supply_demand <= 0:
                # Energy shortage - draw from storage
                available_from_storage = prev_storage * e_out
                if supply_demand + available_from_storage <= 0:
                    net_supply_df.loc[i, storage_level_col] = 0
                else:
                    energy_drawn = -supply_demand / e_out
                    net_supply_df.loc[i, storage_level_col] = prev_storage - energy_drawn

            else:
                # Energy surplus - store energy first, then allocate excess to DAC only if storage is full
                energy_available_for_electrolyser = min(supply_demand, ELECTROLYSER_MAX_DAILY_ENERGY)
                energy_to_store = energy_available_for_electrolyser * e_in

                if ONLY_DAC_IF_STORAGE_FULL:
                    # Calculate storage capacity remaining
                    storage_space_available = STORAGE_MAX_CAPACITY - prev_storage

                    if energy_to_store <= storage_space_available:
                        # All energy can be stored, no DAC needed
                        net_supply_df.loc[i, storage_level_col] = prev_storage + energy_to_store
                        net_supply_df.loc[i, residual_energy_col] = 0
                        net_supply_df.loc[i, dac_energy_col] = 0
                    else:
                        # Storage gets filled, excess energy available for DAC
                        energy_used_for_storage = storage_space_available / e_in
                        residual_energy = supply_demand - energy_used_for_storage
                        dac_energy = min(residual_energy, DAC_MAX_DAILY_ENERGY)

                        net_supply_df.loc[i, storage_level_col] = STORAGE_MAX_CAPACITY
                        net_supply_df.loc[i, residual_energy_col] = residual_energy
                        net_supply_df.loc[i, dac_energy_col] = dac_energy
                        net_supply_df.loc[i, unused_energy_col] = residual_energy - dac_energy

                else:
                    # Calculate new storage level
                    new_storage_level = min(prev_storage + energy_to_store, STORAGE_MAX_CAPACITY)
                    actual_energy_stored = (new_storage_level - prev_storage) / e_in

                    # Calculate residual energy after storage
                    residual_energy = supply_demand - actual_energy_stored

                    # Allocate residual energy to DAC
                    dac_energy = min(residual_energy, DAC_MAX_DAILY_ENERGY) if residual_energy > 0 else 0

                    # Update results
                    net_supply_df.loc[i, storage_level_col] = new_storage_level
                    net_supply_df.loc[i, residual_energy_col] = residual_energy
                    net_supply_df.loc[i, dac_energy_col] = dac_energy
                    net_supply_df.loc[i, unused_energy_col] = residual_energy - dac_energy

    # === CHECK RESULTS ===
    assert (net_supply_df[residual_energy_col] >= 0).all(), "Residual energy cannot be negative"
    assert (net_supply_df[unused_energy_col] >= 0).all(), "Unused energy cannot be negative"
    assert (net_supply_df[storage_level_col] <= STORAGE_MAX_CAPACITY).all(), "Storage levels cannot exceed maximum capacity"
    assert (net_supply_df[dac_energy_col] <= DAC_MAX_DAILY_ENERGY).all(), "DAC energy cannot exceed its maximum daily capacity"
    assert (net_supply_df[storage_level_col] >= 0).all(), "Storage levels cannot be negative"

    return net_supply_df


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


df = pd.read_csv("tests/rei_net_supply_df_12gw_nuclear.csv")
net_supply_df = run_simulation(df)
results = analyze_simulation_results(net_supply_df)
print_simulation_results(results)
