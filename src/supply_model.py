import pandas as pd

import src.assumptions as A
from src import demand_model
from src.data import renewable_capacity_factors
from src.units import Units as U


def daily_renewables_capacity(renewable_capacity: float, capacity_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the daily renewable generation capacity based on the given renewable capacity and capacity factors.
    Args:
        renewable_capacity (float): Total renewable capacity.
        capacity_factors (pd.DataFrame): DataFrame containing daily capacity factors for solar, offshore wind, and onshore wind.

    Returns:
        pd.DataFrame: A DataFrame with daily renewable generation capacity.
    """
    solar = renewable_capacity * A.Renewables.CapacityRatios.Solar * capacity_factors["solar"]
    offshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OffshoreWind * capacity_factors["offshore"]
    onshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OnshoreWind * capacity_factors["onshore"]
    total_power = solar + offshore_wind + onshore_wind + A.Nuclear.Capacity * A.Nuclear.CapacityFactor
    return (total_power * A.HoursPerDay).astype("pint[TWh]")


def get_net_supply(demand_data: str = "era5", *, naive_demand_scaling: bool = False) -> pd.DataFrame:
    """
    Get net supply dataframe (supply minus demand) for analysis.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".
        naive_demand_scaling (bool): If True, use naive demand scaling; otherwise, use seasonal demand scaling.

    Returns:
        pd.DataFrame: DataFrame with renewable capacity as columns and daily net demand (supply - demand) as values.
                      Negative values indicate demand exceeds supply.
    """
    # get demand
    historical_demand_df = demand_model.historical_electricity_demand(demand_data)
    projected_demand_df = (
        demand_model.naive_demand_scaling(historical_demand_df)
        if naive_demand_scaling
        else demand_model.seasonal_demand_scaling(historical_demand_df)
    )

    # Repeat the average year to match the original dataframe length
    num_years = len(historical_demand_df) // len(projected_demand_df)
    remaining_days = len(historical_demand_df) % len(projected_demand_df)
    repeated_demand = pd.concat([projected_demand_df] * num_years + [projected_demand_df.iloc[:remaining_days]])
    repeated_demand.index = historical_demand_df.index

    # get output for a range of renewable capacities
    daily_capacity_factors = renewable_capacity_factors.get_renewable_capacity_factors(resample="D")
    renewable_capacities = [x * U.GW for x in range(100, 500, 10)]
    supply_df = pd.DataFrame({capacity.magnitude: daily_renewables_capacity(capacity, daily_capacity_factors) for capacity in renewable_capacities})

    # reindex for subtraction
    common_idx = supply_df.index.intersection(historical_demand_df.index)
    supply_df = supply_df.reindex(common_idx)
    repeated_demand = repeated_demand.reindex(common_idx)

    # subtract the demand from the renewable generation to get the net demand
    return supply_df.sub(repeated_demand, axis=0)


def fraction_days_without_excess(net_supply_df: pd.DataFrame, *, return_mean: bool = True) -> pd.Series:
    """
    Calculate the fraction of days without excess renewable generation for a range of renewable capacities.

    Args:
        net_supply_df (pd.DataFrame): DataFrame with renewable capacity as columns and daily net supply (supply - demand) as values.
        return_mean (bool): If True, return the mean fraction of days without excess generation.

    Returns:
        pd.Series: A series with renewable capacity as index and the number of days without excess generation as values.
    """
    # count the number of days without excess generation (where net supply is negative)
    days_without_excess = (net_supply_df < 0).mean(axis=0) if return_mean else (net_supply_df < 0).sum(axis=0)
    days_without_excess.index.name = "renewable_capacity_GW"
    days_without_excess.name = "days_without_excess_generation"

    return days_without_excess


def fraction_unmet_demand(net_supply_df: pd.DataFrame, demand_df: pd.Series) -> pd.Series:
    """
    Calculate the fraction of demand that is unmet for a range of renewable capacities.

    Args:
        net_supply_df: DataFrame with renewable capacity as columns and daily net supply (supply - demand) as values.
        demand_df: Series with daily demand values.

    Returns:
        pd.Series: A series with renewable capacity as index and the fraction of unmet demand as values.
    """
    # only look at days where demand is greater than supply

    unmet_demand = net_supply_df.clip(upper=0).abs().sum(axis=0)
    unmet_demand_fraction = unmet_demand / demand_df.sum()
    unmet_demand_fraction.index.name = "renewable_capacity_GW"
    unmet_demand_fraction.name = "fraction_unmet_demand"

    return unmet_demand_fraction
