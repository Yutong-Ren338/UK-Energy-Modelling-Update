from typing import Literal

import pandas as pd

from src import assumptions as A
from src.data import cb7, historical_demand
from src.data.historical_demand import HistoricalDemandSource

DemandMode = Literal["naive", "seasonal", "cb7"]


def naive_demand_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Get naive scaled demand data for analysis. This doesn't take into account increased seasonality
    from electrification of space heating and hot water.

    Args:
        df: DataFrame containing the historical electricity demand data.

    Returns:
        DataFrame with daily demand values scaled to 2050 levels.
    """
    # convert from GW to TWh
    df["demand"] *= A.HoursPerDay

    # Calculate yearly totals and scale each year independently
    df["year"] = df.index.year
    yearly_totals = df.groupby("year")["demand"].sum()
    df["yearly_total"] = df["year"].map(yearly_totals)
    scaling_factor = A.EnergyDemand2050 / df["yearly_total"]
    df["demand"] = (df["demand"] * scaling_factor).astype("pint[terawatt_hour]")

    return df[["demand"]]


def seasonality_index(df: pd.DataFrame, column: str, *, average_year: bool = False) -> pd.Series:
    """Calculate the seasonality index for a given column in the DataFrame.

    Args:
        df: DataFrame containing the historical demand data.
        column: The column name for which to calculate the seasonality index.
        average_year: If True, averages the values over different years.

    Returns:
        Series containing the seasonality index.
    """
    if average_year:
        df["day_of_year"] = df.index.dayofyear
        xs = df.groupby("day_of_year")[column].mean()
        return xs / xs.mean()

    df["year"] = df.index.year
    yearly_means = df.groupby("year")[column].mean()
    df["yearly_means"] = df["year"].map(yearly_means)
    return df[column] / df["yearly_means"]


def seasonal_demand_scaling(df: pd.DataFrame, *, old_gas_data: bool = False, filter_ldz: bool = True) -> pd.DataFrame:
    """Scale the electricity demand data, taking into account increased seasonality from electrification
    of space heating and hot water.

    Use the raw historical electricity demand data, but average the gas data over different years.

    Args:
        df: DataFrame containing the historical electricity demand data.
        old_gas_data: If True, uses the old gas demand data. This should be False (just for testing).
        filter_ldz: If True, filters the gas data for "NTS Energy Offtaken, LDZ Offtake Total".
                           This should always be true (just for testing).

    Returns:
        DataFrame with daily demand values scaled to 2050 levels.
    """
    df_gas = historical_demand.historical_gas_demand(old_gas_data=old_gas_data, filter_ldz=filter_ldz)
    gas_seasonality = seasonality_index(df_gas, "demand", average_year=True)
    ele_seasonality = seasonality_index(df, "demand", average_year=False)

    # get the daily heating demand
    total_2050_heat_demand = A.CB7EnergyDemand2050Buildings * A.CB7FractionHeatDemandBuildings
    daily_2050_heat_demand = total_2050_heat_demand / 365
    daily_heating_demand = daily_2050_heat_demand * gas_seasonality

    # daily non-heating demand
    non_heating_demand = A.EnergyDemand2050 - total_2050_heat_demand
    daily_non_heating_demand = non_heating_demand / 365 * ele_seasonality

    # join the two series by extracting day of year from electricity demand index
    daily_non_heating_demand.name = "non_heating_demand"
    df_out = daily_non_heating_demand.to_frame()
    daily_heating_demand.name = "heating_demand"
    daily_heating_demand = daily_heating_demand.to_frame()
    df_out["day_of_year"] = df.index.dayofyear
    df_out = df_out.join(daily_heating_demand, on="day_of_year")

    # compute the total demand
    df_out["demand"] = df_out["heating_demand"] + df_out["non_heating_demand"]
    return df_out[["demand"]]


def predicted_demand(
    mode: DemandMode = "naive",
    historical: HistoricalDemandSource = "era5",
    *,
    old_gas_data: bool = False,
    filter_ldz: bool = True,
    average_year: bool = True,
) -> pd.DataFrame:
    """Get the predicted demand for 2050 based on the specified mode.

    Args:
        mode: The mode of demand prediction.
        historical: The source of historical demand data, either "era5" or "espeni".
        old_gas_data: If True, uses the old gas demand data.
        filter_ldz: If True, filters the gas data for "NTS Energy Offtaken, LDZ Offtake Total".
        average_year: If True, returns the average over different years.

    Returns:
        pd.DataFrame: DataFrame with predicted daily demand values.

    Raises:
        ValueError: If the mode is not a valid DemandMode.
    """
    df = historical_demand.historical_electricity_demand(source=historical)
    if mode == "naive":
        out = naive_demand_scaling(df)
    elif mode == "seasonal":
        out = seasonal_demand_scaling(df, old_gas_data=old_gas_data, filter_ldz=filter_ldz)
    elif mode == "cb7":
        out = cb7.cb7_demand(A.EnergyDemand2050)[["demand"]]
    else:
        raise ValueError(f"Invalid mode. Choose from {DemandMode.__args__}.")

    if average_year:
        out = out.groupby(out.index.dayofyear).mean()

    return out
