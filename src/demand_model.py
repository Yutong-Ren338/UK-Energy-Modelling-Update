from pathlib import Path

import pandas as pd

from src.data import demand
from src.units import Units as U  # noqa
from src.utils import rolling_mean_circular


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


def electricity_seasonality_index() -> pd.DataFrame:
    """
    Calculate the electricity seasonality index based on daily electricity demand data.

    Returns:
        pd.DataFrame: DataFrame containing the day of the year and the seasonality index.
    """
    df = demand.demand_era5("D")

    df["day_of_year"] = df.index.dayofyear
    df = df.groupby("day_of_year")["demand"].mean().reset_index()

    df["demand_smooth"] = rolling_mean_circular(df, "demand", window_size=30)
    df["seasonality_index"] = df["demand_smooth"] / df["demand"].mean()

    return df[["day_of_year", "seasonality_index"]]


def combined_seasonality_index() -> pd.DataFrame:
    """
    Combine the gas and electricity seasonality indices into a single DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the day of the year, gas seasonality index, and electricity seasonality index.
    """
    gas_df = gas_seasonality_index()
    ele_df = electricity_seasonality_index()
    return gas_df.merge(ele_df, on="day_of_year", suffixes=("_gas", "_electricity"))
