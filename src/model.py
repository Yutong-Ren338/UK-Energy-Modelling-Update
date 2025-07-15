import pandas as pd

import src.assumptions as A
from src.data import demand, renewable_capacity_factors
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
    return solar + offshore_wind + onshore_wind + A.Nuclear.Capacity * A.Nuclear.CapacityFactor


def get_net_supply(demand_data: str = "era5") -> pd.DataFrame:
    """
    Get net supply dataframe (supply minus demand) for analysis.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".

    Returns:
        pd.DataFrame: DataFrame with renewable capacity as columns and daily net demand (supply - demand) as values.
                     Negative values indicate demand exceeds supply.

    Raises:
        ValueError: If demand_data is not "era5" or "espeni".
    """
    # get demand
    if demand_data == "era5":
        demand_df = demand.demand_era5("D")
    elif demand_data == "espeni":
        demand_df = demand.demand_espeni("D")
    else:
        raise ValueError("Invalid demand_data. Choose 'era5' or 'espeni'.")

    # scale each year's demand to 2050 demand
    AVERAGE_YEAR = True  # noqa: N806
    if AVERAGE_YEAR:
        # Create average year by averaging each day of year across all years
        demand_df["day_of_year"] = demand_df.index.dayofyear
        average_year = demand_df.groupby("day_of_year")["demand"].mean()

        # Scale the average year to 2050 demand
        demand_2050 = (A.EnergyDemand2050 / A.HoursPerYear).to(U.GW)
        average_year_scaled = average_year * demand_2050 / average_year.mean()

        # Repeat the average year to match the original dataframe length
        num_years = len(demand_df) // len(average_year)
        remaining_days = len(demand_df) % len(average_year)

        # Create repeated pattern
        repeated_demand = pd.concat([average_year_scaled] * num_years + [average_year_scaled.iloc[:remaining_days]])
        repeated_demand.index = demand_df.index
        demand_df["demand"] = repeated_demand.to_numpy()

    else:
        demand_df["year"] = demand_df.index.year
        demand_df["yearly_demand"] = demand_df.groupby("year")["demand"].transform("mean")
        demand_2050 = (A.EnergyDemand2050 / A.HoursPerYear).to(U.GW)
        demand_df["demand"] *= demand_2050 / demand_df["yearly_demand"]

    # get output for a range of renewable capacities
    daily_capacity_factors = renewable_capacity_factors.get_renewable_capacity_factors(resample="D")
    renewable_capacities = [x * U.GW for x in range(100, 500, 10)]
    supply_df = pd.DataFrame({capacity.magnitude: daily_renewables_capacity(capacity, daily_capacity_factors) for capacity in renewable_capacities})

    # reindex for subtraction
    common_idx = supply_df.index.intersection(demand_df.index)
    supply_df = supply_df.reindex(common_idx)
    demand_df = demand_df.reindex(common_idx)

    # subtract the demand from the renewable generation to get the net demand
    return supply_df.sub(demand_df["demand"], axis=0)


def fraction_days_without_excess(demand_data: str = "era5", *, return_mean: bool = True) -> pd.Series:
    """
    Calculate the fraction of days without excess renewable generation for a range of renewable capacities.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".
        return_mean (bool): If True, return the mean fraction of days without excess generation.

    Returns:
        pd.Series: A series with renewable capacity as index and the number of days without excess generation as values.
    """
    # get net supply dataframe (supply - demand)
    net_supply_df = get_net_supply(demand_data)

    # count the number of days without excess generation (where net supply is negative)
    days_without_excess = (net_supply_df < 0).mean(axis=0) if return_mean else (net_supply_df < 0).sum(axis=0)
    days_without_excess.index.name = "renewable_capacity_GW"
    days_without_excess.name = "days_without_excess_generation"

    return days_without_excess
