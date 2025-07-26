# ruff: noqa: PLR0913, PLR0917, FBT001
from typing import NamedTuple

import numba
import numpy as np

# Power System Model
# Models renewable energy generation, storage systems, demand response, and excess energy allocation
# Includes energy storage, Direct Air Capture (DAC), and curtailment strategies

# Floating point precision tolerance for residual energy calculations
FLOATING_POINT_TOLERANCE = 1e-10


class SimulationParameters(NamedTuple):
    """Parameters for the core power system simulation."""

    initial_hydrogen_storage_level: float
    max_hydrogen_storage_capacity: float
    electrolyser_max_daily_energy: float
    dac_max_daily_energy: float
    hydrogen_e_in: float
    hydrogen_e_out: float
    only_dac_if_hydrogen_storage_full: bool


@numba.njit(cache=True)
def handle_deficit(
    net_supply: float,
    prev_hydrogen_storage: float,
    hydrogen_e_out: float,
) -> tuple[float, bool]:
    """Handle energy deficit scenario by drawing from storage.

    Args:
        net_supply: Negative supply-demand value (deficit)
        prev_hydrogen_storage: Previous hydrogen storage level
        hydrogen_e_out: Hydrogen storage output efficiency

    Returns:
        Tuple of (new_hydrogen_storage_level, simulation_failed)
    """
    available_from_storage = prev_hydrogen_storage * hydrogen_e_out
    if net_supply + available_from_storage <= 0:
        # Not enough storage to meet demand - simulation failed
        return 0.0, True

    # Partial storage draw
    energy_drawn = -net_supply / hydrogen_e_out
    hydrogen_storage_level = prev_hydrogen_storage - energy_drawn
    return hydrogen_storage_level, False


@numba.njit(cache=True)
def handle_surplus(
    net_supply: float,
    prev_hydrogen_storage: float,
    max_hydrogen_storage: float,
    max_electrolyser: float,
    hydrogen_e_in: float,
    only_dac_if_storage_full: bool,
) -> tuple[float, float, float]:
    """Handle energy surplus with either storage-first or balanced allocation strategy.

    Args:
        net_supply: Positive supply-demand value (surplus)
        prev_hydrogen_storage: Previous hydrogen storage level
        max_hydrogen_storage: Maximum hydrogen storage capacity
        max_electrolyser: Maximum electrolyser daily energy
        hydrogen_e_in: Hydrogen storage input efficiency
        only_dac_if_storage_full: If True, DAC only gets energy when storage is full;
                      if False, DAC can get energy even when storage isn't full

    Returns:
        Tuple of (hydrogen_storage_level, residual_energy, stored_energy)
    """
    energy_available_for_electrolyser = min(net_supply, max_electrolyser)
    energy_to_store = energy_available_for_electrolyser * hydrogen_e_in
    hydrogen_storage_space_available = max_hydrogen_storage - prev_hydrogen_storage

    if only_dac_if_storage_full:  # fill storage completely before any DAC
        if energy_to_store <= hydrogen_storage_space_available:
            # All energy can be stored
            hydrogen_storage_level = prev_hydrogen_storage + energy_to_store
            stored_energy = energy_available_for_electrolyser
            residual_energy = 0.0
        else:
            # Storage gets filled, excess energy available for DAC
            energy_used_for_storage = hydrogen_storage_space_available / hydrogen_e_in
            residual_energy = net_supply - energy_used_for_storage
            hydrogen_storage_level = max_hydrogen_storage
            stored_energy = energy_used_for_storage
    else:  # max out electrolyser and then use DAC (even if storage isn't full)
        new_hydrogen_storage_level = min(prev_hydrogen_storage + energy_to_store, max_hydrogen_storage)
        actual_energy_stored = (new_hydrogen_storage_level - prev_hydrogen_storage) / hydrogen_e_in
        residual_energy = net_supply - actual_energy_stored

        # Fix small negative values due to floating point precision errors
        # But preserve larger negative values that indicate actual logic errors
        if residual_energy < 0 and residual_energy > -FLOATING_POINT_TOLERANCE:
            residual_energy = 0.0

        hydrogen_storage_level = new_hydrogen_storage_level
        stored_energy = actual_energy_stored

    return hydrogen_storage_level, residual_energy, stored_energy


@numba.njit(cache=True)
def simulate_power_system_core(net_supply_values: np.ndarray, params: SimulationParameters) -> np.ndarray:
    """Core simulation function optimized for Numba JIT compilation.

    Uses smaller specialized functions for different scenarios to improve readability
    while maintaining JIT performance.

    Args:
        net_supply_values: Array of supply-demand values for each timestep
        params: Simulation parameters

    Returns:
        Array of shape (n_timesteps, 5) containing:
        [hydrogen_storage_level, residual_energy, dac_energy, curtailed_energy, stored_energy]
        Returns array filled with NaN values if simulation fails (storage hits zero).
    """
    n_timesteps = len(net_supply_values)
    results = np.zeros((n_timesteps, 5))

    # Extract ALL parameters to local variables
    max_hydrogen_storage = params.max_hydrogen_storage_capacity
    max_electrolyser = params.electrolyser_max_daily_energy
    max_dac = params.dac_max_daily_energy
    hydrogen_e_in = params.hydrogen_e_in
    hydrogen_e_out = params.hydrogen_e_out
    only_dac_if_storage_full = params.only_dac_if_hydrogen_storage_full

    prev_hydrogen_storage = params.initial_hydrogen_storage_level

    for i in range(n_timesteps):
        net_supply = net_supply_values[i]

        if net_supply <= 0:
            # Energy shortage - draw from storage
            hydrogen_storage_level, simulation_failed = handle_deficit(net_supply, prev_hydrogen_storage, hydrogen_e_out)
            if simulation_failed:
                results[:] = np.nan
                return results

            # Deficit scenario - all other values are zero
            residual_energy = dac_energy = curtailed_energy = stored_energy = 0.0

        else:
            # Energy surplus - use unified handler for both strategies
            hydrogen_storage_level, residual_energy, stored_energy = handle_surplus(
                net_supply, prev_hydrogen_storage, max_hydrogen_storage, max_electrolyser, hydrogen_e_in, only_dac_if_storage_full
            )

            # Handle DAC allocation for residual energy
            dac_energy = min(residual_energy, max_dac) if residual_energy > 0 else 0.0
            curtailed_energy = max(0.0, residual_energy - dac_energy)

        # Direct array assignment is faster than list creation
        results[i, 0] = hydrogen_storage_level
        results[i, 1] = residual_energy
        results[i, 2] = dac_energy
        results[i, 3] = curtailed_energy
        results[i, 4] = stored_energy

        prev_hydrogen_storage = hydrogen_storage_level

    return results
