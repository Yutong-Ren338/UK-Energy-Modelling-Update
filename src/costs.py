import src.assumptions as A
from src.units import Units as U


def yearly_cost(capacity: float, capacity_factor: float, lcoe: float) -> float:
    """Calculate the total yearly cost of an energy source.

    Calculates cost based on its installed capacity, capacity factor, and levelized cost of energy (LCOE).

    Args:
        capacity: Installed capacity in GW.
        capacity_factor: Capacity factor as a fraction (0.0 to 1.0).
        lcoe: Levelized Cost of Energy in GBP/MWh.

    Returns:
        Total yearly cost in GBP.
    """
    # Calculate the annual energy production in GWh
    annual_energy_production = capacity * capacity_factor * A.HoursPerYear

    # Return the total cost in GBP
    return (annual_energy_production * lcoe).to(U.GBP)


def total_storage_cost(
    storage_capacity: float,
    electrolyser_power: float,
    generation_capacity: float,
) -> float:
    """Calculate the total cost of energy storage.

    Includes electrolysis, storage, and generation costs.

    Args:
        storage_capacity: Storage capacity in kWh.
        electrolyser_power: Power of the electrolyser in kW.
        generation_capacity: Generation capacity in kW.

    Returns:
        Total cost of energy storage in GBP.
    """
    storage_cost = storage_capacity * A.HydrogenStorage.CavernStorage.AnnualisedCost
    electrolyser_cost = electrolyser_power * A.HydrogenStorage.Electrolysis.AnnualisedCost
    generation_cost = generation_capacity * A.HydrogenStorage.Generation.AnnualisedCost
    return (storage_cost + electrolyser_cost + generation_cost).to(U.GBP)


def total_system_cost(
    renewable_capacity: float,
    storage_capacity: float,
    electrolyser_power: float,
    generation_capacity: float,
) -> float:
    """Calculate the total system cost.

    Includes renewable energy, storage, electrolysis, and generation costs.

    Args:
        renewable_capacity: Renewable energy capacity in GW.
        storage_capacity: Storage capacity in kWh.
        electrolyser_power: Power of the electrolyser in kW.
        generation_capacity: Generation capacity in kW.

    Returns:
        Total system cost in GBP.
    """
    renewable_cost = yearly_cost(
        capacity=renewable_capacity,
        capacity_factor=A.Renewables.AverageCapacityFactor,
        lcoe=A.Renewables.AverageLCOE,
    )
    nuclear_cost = yearly_cost(
        capacity=A.Nuclear.Capacity,
        capacity_factor=A.Nuclear.CapacityFactor,
        lcoe=A.Nuclear.AverageLCOE,
    )
    storage_cost = total_storage_cost(storage_capacity, electrolyser_power, generation_capacity)
    additional_costs = A.AdditionalCosts * A.EnergyDemand2050
    return (renewable_cost + nuclear_cost + storage_cost + additional_costs).to(U.GBP)


def energy_cost(system_cost: float, energy_demand: float) -> float:
    """Calculate the cost of energy per MWh.

    Calculates cost based on the total system cost and energy demand (the energy delivered by the system).

    Args:
        system_cost: Total system cost in GBP.
        energy_demand: Total energy demand in MWh.

    Returns:
        Cost of energy in GBP/MWh.
    """
    return (system_cost / energy_demand).to(U.GBP / U.MWh)


system_cost = total_system_cost(
    renewable_capacity=220 * U.GW,
    storage_capacity=165 * U.TWh,
    electrolyser_power=80 * U.GW,
    generation_capacity=100 * U.GW,
)
print(energy_cost(system_cost, A.EnergyDemand2050))


system_cost = total_system_cost(
    renewable_capacity=250 * U.GW,
    storage_capacity=67 * U.TWh,
    electrolyser_power=100 * U.GW,
    generation_capacity=100 * U.GW,
)
print(energy_cost(system_cost, A.EnergyDemand2050))
