from pathlib import Path

import pandas as pd

from src import assumptions as A
from src.data import demand
from src.utils import rolling_mean_circular


def get_raw_demand(demand_data: str = "era5") -> pd.DataFrame:
    """
    Get raw demand data for analysis.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".

    Returns:
        pd.DataFrame: DataFrame with daily demand values.

    Raises:
        ValueError: If demand_data is not "era5" or "espeni".
    """
    if demand_data == "era5":
        return demand.demand_era5("D")
    if demand_data == "espeni":
        return demand.demand_espeni("D")
    raise ValueError("Invalid demand_data. Choose 'era5' or 'espeni'.")


def gas_seasonality_index(*, filter_lzd: bool = True) -> pd.DataFrame:
    """
    Calculate the gas seasonality index based on daily gas demand data.

    Args:
        filter_lzd: If True, filters the data for "NTS Energy Offtaken, LDZ Offtake Total".
                    This should always be true for the gas seasonality index calculation, but is provided
                    to measure the impact.

    Returns:
        pd.DataFrame: DataFrame containing the day of the year and the seasonality index.
    """
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "new" / "UK_gas_demand_processed.csv", parse_dates=["date"])

    if filter_lzd:
        df = df[df["use"] == "NTS Energy Offtaken, LDZ Offtake Total"]

    df["demand (TWh)"] = df["demand (TWh)"].astype("pint[terawatt_hour]")

    df["day_of_year"] = df["date"].dt.dayofyear
    df = df.groupby("day_of_year")["demand (TWh)"].mean().reset_index()

    df["demand_smooth"] = rolling_mean_circular(df, "demand (TWh)", window_size=30)
    df["seasonality_index"] = df["demand_smooth"] / df["demand (TWh)"].mean()

    return df[["day_of_year", "seasonality_index"]]


def electricity_seasonality_index(demand_data: str = "era5") -> pd.DataFrame:
    """
    Calculate the electricity seasonality index based on daily electricity demand data.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".

    Returns:
        pd.DataFrame: DataFrame containing the day of the year and the seasonality index.
    """
    df = get_raw_demand(demand_data)

    df["day_of_year"] = df.index.dayofyear
    df = df.groupby("day_of_year")["demand"].mean().reset_index()

    df["demand_smooth"] = rolling_mean_circular(df, "demand", window_size=30)
    df["seasonality_index"] = df["demand_smooth"] / df["demand"].mean()

    return df[["day_of_year", "seasonality_index"]]


def combined_seasonality_index(demand_data: str = "era5", *, filter_ldz: bool = True) -> pd.DataFrame:
    """
    Combine the gas and electricity seasonality indices into a single DataFrame.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".
        filter_ldz (bool): If True, filters the gas data for "NTS Energy Offtaken, LDZ Offtake Total".
                           This should always be true for the gas seasonality index calculation, but is provided
                           to measure the impact.

    Returns:
        pd.DataFrame: DataFrame containing the day of the year, gas seasonality index, and electricity seasonality index.
    """
    gas_df = gas_seasonality_index(filter_lzd=filter_ldz)
    ele_df = electricity_seasonality_index(demand_data=demand_data)
    return gas_df.merge(ele_df, on="day_of_year", suffixes=("_gas", "_electricity"))


def naive_demand_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get naive scaled demand data for analysis. This doesn't take into account increased seasonality
    from electrification of space heating and hot water.

    Args:
        df (pd.DataFrame): DataFrame containing the historical demand data.

    Returns:
        pd.DataFrame: DataFrame with daily demand values scaled to 2050 levels.
    """

    # convert from GW to TWh
    df["demand"] *= A.HoursPerDay

    # Create average year by averaging each day of year across all years
    df["day_of_year"] = df.index.dayofyear
    average_year = df.groupby("day_of_year")["demand"].mean().astype("pint[terawatt_hour]")

    # Scale the average year to 2050 demand
    return average_year * A.EnergyDemand2050 / 365 / average_year.mean()


def demand_scaling(demand_data: str = "era5", *, filter_ldz: bool = True) -> pd.DataFrame:
    """
    Scale the demand data to 2050 levels, taking into account increased seasonality from electrification
    of space heating and hot water.

    Args:
        demand_data (str): The source of demand data, either "era5" or "espeni".
        filter_ldx (bool): If True, filters the gas data for "NTS Energy Offtaken, LDZ Offtake Total".
                           This should always be true for the gas seasonality index calculation, but is provided
                           to measure the impact.

    Returns:
        pd.DataFrame: DataFrame with daily demand values scaled to 2050 levels.
    """

    df_combined = combined_seasonality_index(demand_data=demand_data, filter_ldz=filter_ldz)

    total_2050_heat_demand = A.CB7EnergyDemand2050Buildings * A.CB7FractionHeatDemandBuildings
    daily_2050_heat_demand = total_2050_heat_demand / 365

    # get the daily heating demand
    daily_heating_demand = daily_2050_heat_demand * df_combined["seasonality_index_gas"]

    # daily non-heating demand
    non_heating_demand = A.CB7EnergyDemand2050 - total_2050_heat_demand
    daily_non_heating_demand = non_heating_demand / 365 * df_combined["seasonality_index_electricity"]

    # add in the rest of the demand
    daily_total_demand = daily_heating_demand + daily_non_heating_demand

    return daily_total_demand
