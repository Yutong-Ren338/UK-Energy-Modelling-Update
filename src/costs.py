import src.assumptions as A


def cost_contribution_per_mwh(capacity: float, capacity_factor: float, lcoe: float, demand: float) -> float:
    """
    Calculate the cost contribution per MWh of an energy source to the overall energy system.

    This function determines how much each energy source contributes to the average cost per MWh
    of electricity in the system, based on its share of total generation and its individual LCOE.

    Args:
        capacity (float): Installed capacity in GW.
        capacity_factor (float): Capacity factor as a fraction (0.0 to 1.0).
        lcoe (float): Levelized Cost of Energy in GBP/MWh.
        demand (float): Total energy demand across all sources in TWh.

    Returns:
        float: Cost contribution in GBP/MWh. This represents the portion of the
               system-wide average electricity cost attributable to this energy source.
    """
    # Calculate the annual energy production in GWh
    annual_energy_production = capacity * capacity_factor * A.HoursPerYear

    # Share of total demand
    frac_demand = annual_energy_production / (demand * 1e3)  # Convert TWh to GWh

    # Calculate the cost contribution in GBP / MWh
    return frac_demand * lcoe
