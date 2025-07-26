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
    hydrogen_storage_capacity: float
    electrolyser_max_daily_energy: float
    dac_max_daily_energy: float
    hydrogen_e_in: float
    hydrogen_e_out: float
    only_dac_if_hydrogen_storage_full: bool
    # Medium term storage parameters
    initial_medium_storage_level: float
    medium_storage_capacity: float
    medium_storage_power: float
    medium_storage_efficiency: float


@numba.njit(cache=True)
def handle_deficit(
    net_supply: float,
    prev_medium_storage: float,
    prev_hydrogen_storage: float,
    medium_storage_power: float,
    medium_storage_efficiency: float,
    hydrogen_e_out: float,
) -> tuple[float, float, bool]:
    """Handle energy deficit scenario by drawing from storage.

    Priority order: Medium-term storage first, then hydrogen storage.

    Args:
        net_supply: Negative supply-demand value (deficit)
        prev_medium_storage: Previous medium-term storage level
        prev_hydrogen_storage: Previous hydrogen storage level
        medium_storage_power: Maximum daily energy capacity for medium storage (power * 24h)
        medium_storage_efficiency: Medium-term storage round-trip efficiency
        hydrogen_e_out: Hydrogen storage output efficiency

    Returns:
        Tuple of (new_medium_storage_level, new_hydrogen_storage_level, simulation_failed)
    """
    remaining_deficit = -net_supply
    medium_storage_level = prev_medium_storage
    hydrogen_storage_level = prev_hydrogen_storage

    # First, try to meet deficit from medium-term storage
    if remaining_deficit > 0 and prev_medium_storage > 0:
        # Available energy from medium storage (considering efficiency and power constraints)
        available_from_medium = min(prev_medium_storage * medium_storage_efficiency, medium_storage_power)
        energy_from_medium = min(remaining_deficit, available_from_medium)

        # Update medium storage level (accounting for efficiency)
        energy_drawn_from_medium = energy_from_medium / medium_storage_efficiency
        medium_storage_level = prev_medium_storage - energy_drawn_from_medium

        # Fix small negative values due to floating point precision errors
        if medium_storage_level < 0 and medium_storage_level > -FLOATING_POINT_TOLERANCE:
            medium_storage_level = 0.0

        remaining_deficit -= energy_from_medium

    # If deficit remains, use hydrogen storage
    if remaining_deficit > 0 and prev_hydrogen_storage > 0:
        available_from_hydrogen = prev_hydrogen_storage * hydrogen_e_out

        if remaining_deficit > available_from_hydrogen:
            # Not enough total storage to meet demand - simulation failed
            return 0.0, 0.0, True

        # Draw from hydrogen storage
        energy_drawn_from_hydrogen = remaining_deficit / hydrogen_e_out
        hydrogen_storage_level = prev_hydrogen_storage - energy_drawn_from_hydrogen
        remaining_deficit = 0.0

    # Check if deficit was fully met
    if remaining_deficit > 0:
        # Not enough storage to meet demand - simulation failed
        return 0.0, 0.0, True

    return medium_storage_level, hydrogen_storage_level, False


@numba.njit(cache=True)
def handle_surplus(
    net_supply: float,
    prev_medium_storage: float,
    prev_hydrogen_storage: float,
    max_medium_storage: float,
    max_hydrogen_storage: float,
    medium_storage_power: float,
    medium_storage_efficiency: float,
    max_electrolyser: float,
    hydrogen_e_in: float,
    only_dac_if_storage_full: bool,
) -> tuple[float, float, float, float, float, float]:
    """Handle energy surplus allocation between storages and DAC.

    Energy allocation priority system:
    1. Medium-term storage (up to power and capacity limits)
    2. Hydrogen storage via electrolyser (up to electrolyser and capacity limits)
    3. Remaining energy available for DAC (policy-dependent on hydrogen storage only)

    Args:
        net_supply: Positive supply-demand value (surplus energy available)
        prev_medium_storage: Previous medium-term storage level
        prev_hydrogen_storage: Previous hydrogen storage level
        max_medium_storage: Maximum medium-term storage capacity
        max_hydrogen_storage: Maximum hydrogen storage capacity
        medium_storage_power: Maximum daily energy for medium storage (power * 24h)
        medium_storage_efficiency: Medium-term storage round-trip efficiency
        max_electrolyser: Maximum electrolyser daily energy capacity
        hydrogen_e_in: Hydrogen storage input efficiency
        only_dac_if_storage_full: Energy allocation policy (applies to hydrogen storage only)

    Returns:
        Tuple of (medium_storage_level, hydrogen_storage_level, surplus_energy,
                 energy_into_medium_storage, energy_into_hydrogen_storage, curtailed_energy)
    """
    remaining_energy = net_supply
    medium_storage_level = prev_medium_storage
    hydrogen_storage_level = prev_hydrogen_storage
    energy_into_medium_storage = 0.0
    energy_into_hydrogen_storage = 0.0
    curtailed_energy = 0.0

    # First priority: fill medium-term storage
    if remaining_energy > 0 and prev_medium_storage < max_medium_storage:
        available_medium_capacity = max_medium_storage - prev_medium_storage

        # Consider both power constraint and capacity constraint
        max_energy_to_medium = min(remaining_energy, medium_storage_power, available_medium_capacity / medium_storage_efficiency)

        if max_energy_to_medium > 0:
            energy_into_medium_storage = max_energy_to_medium
            # Account for storage efficiency
            actual_stored_medium = max_energy_to_medium * medium_storage_efficiency
            medium_storage_level = prev_medium_storage + actual_stored_medium
            remaining_energy -= max_energy_to_medium

    # Second priority: hydrogen storage via electrolyser
    if remaining_energy > 0:
        energy_available_for_electrolysis = min(remaining_energy, max_electrolyser)
        max_storable_hydrogen = energy_available_for_electrolysis * hydrogen_e_in
        available_hydrogen_capacity = max_hydrogen_storage - prev_hydrogen_storage

        if only_dac_if_storage_full:  # fill hydrogen storage completely before any DAC
            if max_storable_hydrogen <= available_hydrogen_capacity:
                # All electrolyser energy can be stored - no DAC energy available yet
                actual_stored_hydrogen = max_storable_hydrogen
                energy_into_hydrogen_storage = energy_available_for_electrolysis
                surplus_energy = 0.0
                # Energy curtailed due to electrolyser capacity limitation
                electrolyser_curtailed = remaining_energy - energy_available_for_electrolysis
            else:
                # Hydrogen storage capacity exceeded - fill completely, rest available for DAC
                actual_stored_hydrogen = available_hydrogen_capacity
                energy_into_hydrogen_storage = actual_stored_hydrogen / hydrogen_e_in
                surplus_energy = remaining_energy - energy_into_hydrogen_storage
                # Energy curtailed due to electrolyser capacity limitation
                electrolyser_curtailed = remaining_energy - energy_available_for_electrolysis

            hydrogen_storage_level = prev_hydrogen_storage + actual_stored_hydrogen
            curtailed_energy = electrolyser_curtailed

        else:  # max out electrolyser and then use DAC (even if hydrogen storage isn't full)
            # Calculate how much hydrogen can actually be stored
            actual_stored_hydrogen = min(max_storable_hydrogen, available_hydrogen_capacity)
            new_hydrogen_storage_level = prev_hydrogen_storage + actual_stored_hydrogen

            # Calculate energy used for hydrogen storage
            energy_into_hydrogen_storage = actual_stored_hydrogen / hydrogen_e_in

            # Remaining energy is available for DAC
            surplus_energy = remaining_energy - energy_into_hydrogen_storage

            # Fix small negative values due to floating point precision errors
            if surplus_energy < 0 and surplus_energy > -FLOATING_POINT_TOLERANCE:
                surplus_energy = 0.0

            hydrogen_storage_level = new_hydrogen_storage_level
            curtailed_energy = 0.0
    else:
        surplus_energy = 0.0

    return (medium_storage_level, hydrogen_storage_level, surplus_energy, energy_into_medium_storage, energy_into_hydrogen_storage, curtailed_energy)


@numba.njit(cache=True)
def simulate_power_system_core(net_supply_values: np.ndarray, params: SimulationParameters) -> np.ndarray:
    """Core simulation function optimized for Numba JIT compilation.

    Uses smaller specialized functions for different scenarios to improve readability
    while maintaining JIT performance.

    Args:
        net_supply_values: Array of supply-demand values for each timestep
        params: Simulation parameters

    Returns:
        Array of shape (n_timesteps, 7) containing:
        [medium_storage_level, hydrogen_storage_level, surplus_energy, dac_energy,
         curtailed_energy, energy_into_medium_storage, energy_into_hydrogen_storage]
        Returns array filled with NaN values if simulation fails (storage hits zero).
    """
    n_timesteps = len(net_supply_values)
    results = np.zeros((n_timesteps, 7))

    # Extract ALL parameters to local variables
    max_hydrogen_storage = params.hydrogen_storage_capacity
    max_electrolyser = params.electrolyser_max_daily_energy
    max_dac = params.dac_max_daily_energy
    hydrogen_e_in = params.hydrogen_e_in
    hydrogen_e_out = params.hydrogen_e_out
    only_dac_if_storage_full = params.only_dac_if_hydrogen_storage_full

    # Medium-term storage parameters
    max_medium_storage = params.medium_storage_capacity
    medium_storage_power = params.medium_storage_power
    medium_storage_efficiency = params.medium_storage_efficiency

    prev_medium_storage = params.initial_medium_storage_level
    prev_hydrogen_storage = params.initial_hydrogen_storage_level

    for i in range(n_timesteps):
        net_supply = net_supply_values[i]

        if net_supply <= 0:
            # Energy shortage - draw from storage (medium first, then hydrogen)
            medium_storage_level, hydrogen_storage_level, simulation_failed = handle_deficit(
                net_supply, prev_medium_storage, prev_hydrogen_storage, medium_storage_power, medium_storage_efficiency, hydrogen_e_out
            )
            if simulation_failed:
                results[:] = np.nan
                return results

            # Deficit scenario - all other values are zero
            surplus_energy = dac_energy = curtailed_energy = 0.0
            energy_into_medium_storage = energy_into_hydrogen_storage = 0.0

        else:
            # Energy surplus - allocate to storages and DAC
            (
                medium_storage_level,
                hydrogen_storage_level,
                surplus_energy,
                energy_into_medium_storage,
                energy_into_hydrogen_storage,
                electrolyser_curtailed_energy,
            ) = handle_surplus(
                net_supply,
                prev_medium_storage,
                prev_hydrogen_storage,
                max_medium_storage,
                max_hydrogen_storage,
                medium_storage_power,
                medium_storage_efficiency,
                max_electrolyser,
                hydrogen_e_in,
                only_dac_if_storage_full,
            )

            # Handle DAC allocation for residual energy
            dac_energy = min(surplus_energy, max_dac) if surplus_energy > 0 else 0.0
            dac_curtailed_energy = max(0.0, surplus_energy - dac_energy)

            # Total curtailed energy is from both electrolyser capacity and DAC capacity limitations
            curtailed_energy = electrolyser_curtailed_energy + dac_curtailed_energy

        # Direct array assignment is faster than list creation
        results[i, 0] = medium_storage_level
        results[i, 1] = hydrogen_storage_level
        results[i, 2] = surplus_energy
        results[i, 3] = dac_energy
        results[i, 4] = curtailed_energy
        results[i, 5] = energy_into_medium_storage
        results[i, 6] = energy_into_hydrogen_storage

        prev_medium_storage = medium_storage_level
        prev_hydrogen_storage = hydrogen_storage_level

    return results
