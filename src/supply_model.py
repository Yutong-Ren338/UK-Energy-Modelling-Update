import pandas as pd

import src.assumptions as A
from src.data import renewable_capacity_factors
from src.units import Units as U


def daily_renewables_capacity(renewable_capacity: float, capacity_factors: pd.DataFrame) -> pd.DataFrame:
    """Calculate the daily renewable generation capacity.

    Calculates capacity based on the given renewable capacity and capacity factors.

    Args:
        renewable_capacity: Total renewable capacity.
        capacity_factors: DataFrame containing daily capacity factors for solar, offshore wind, and onshore wind.

    Returns:
        A DataFrame with daily renewable generation capacity.
    """
    solar = renewable_capacity * A.Renewables.CapacityRatios.Solar * capacity_factors["solar"]
    offshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OffshoreWind * capacity_factors["offshore"]
    onshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OnshoreWind * capacity_factors["onshore"]
    total_power = solar + offshore_wind + onshore_wind + A.Nuclear.Capacity * A.Nuclear.CapacityFactor
    return (total_power * A.HoursPerDay).astype("pint[TWh]")


def get_net_supply(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Get net supply dataframe (supply minus demand) for analysis.

    Args:
        demand_df: DataFrame containing the projected 2050 demand data.

    Returns:
        DataFrame with renewable capacity as columns and daily net demand (supply - demand) as values.
        Negative values indicate demand exceeds supply.
    """
    # get output for a range of renewable capacities
    daily_capacity_factors = renewable_capacity_factors.get_renewable_capacity_factors(resample="D")
    renewable_capacities = [x * U.GW for x in range(100, 500, 10)]
    supply_df = pd.DataFrame({capacity.magnitude: daily_renewables_capacity(capacity, daily_capacity_factors) for capacity in renewable_capacities})

    # apply losses to supply
    supply_df *= 1 - A.PowerSystem.TotalLosses

    # reindex for subtraction
    common_idx = supply_df.index.intersection(demand_df.index)
    assert len(common_idx) > 0, "No common dates between supply and demand dataframes."
    supply_df = supply_df.reindex(common_idx)
    demand_df = demand_df.reindex(common_idx)

    # subtract the demand from the renewable generation to get the net demand
    return supply_df.sub(demand_df["demand"], axis=0)


def fraction_days_without_excess(net_supply_df: pd.DataFrame, *, return_mean: bool = True) -> pd.Series:
    """Calculate the fraction of days without excess renewable generation.

    Calculates for a range of renewable capacities.

    Args:
        net_supply_df: DataFrame with renewable capacity as columns and daily net supply (supply - demand) as values.
        return_mean: If True, return the mean fraction of days without excess generation.

    Returns:
        A series with renewable capacity as index and the number of days without excess generation as values.
    """
    # count the number of days without excess generation (where net supply is negative)
    days_without_excess = (net_supply_df < 0).mean(axis=0) if return_mean else (net_supply_df < 0).sum(axis=0)
    days_without_excess.index.name = "renewable_capacity_GW"
    days_without_excess.name = "days_without_excess_generation"

    return days_without_excess


def total_unmet_demand(net_supply_df: pd.DataFrame) -> pd.Series:
    """Calculate the total unmet demand.

    Args:
        net_supply_df: DataFrame with renewable capacity as columns and daily net supply (supply - demand) as values.

    Returns:
        A series with renewable capacity as index and the total unmet demand as values.
    """
    unmet_demand = net_supply_df[net_supply_df < 0].sum(axis=0).abs()
    unmet_demand.index.name = "renewable_capacity_GW"
    unmet_demand.name = "total_unmet_demand"

    return unmet_demand
