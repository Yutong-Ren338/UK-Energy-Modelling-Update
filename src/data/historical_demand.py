from pathlib import Path
from typing import Literal

import pandas as pd

from src import DATA_DIR

HistoricalDemandSource = Literal["era5", "espeni"]


def demand_era5(resample: str | None = None, *, weather_adjusted: bool = False) -> pd.DataFrame:
    """
    Load and return the ERA5 full electricity demand data for the UK.

    Arguments:
        resample: Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).
        weather_adjusted: If True, return weather-adjusted demand; otherwise, return raw demand.

    Returns:
        pd.DataFrame: DataFrame containing the demand data in GW.
    """
    if weather_adjusted:
        data_file = DATA_DIR / "ERA5_weather_dependent_demand_UK_1979_2019_hourly.csv"
    else:
        data_file = DATA_DIR / "ERA5_full_demand_UK_1979_2019_hourly.csv"

    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df[df.columns[df.columns.str.contains("United_Kingdom")]]
    df.columns = ["demand"]
    if resample:
        df = df.resample(resample).mean()

    return df.astype("pint[GW]")


def demand_espeni(resample: str | None = None) -> pd.DataFrame:
    """
    Load and return the Espeni full electricity demand data for the UK.

    Arguments:
        resample (str | None): Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.DataFrame: DataFrame containing the demand data in GW.
    """

    # demand is in MW
    df = pd.read_csv(DATA_DIR / "espeni.csv")
    df = df[["ELEXM_utc", "POWER_ESPENI_MW"]]
    df = df.rename(columns={"ELEXM_utc": "date", "POWER_ESPENI_MW": "demand"})
    df["demand"] /= 1000.0  # convert to GW
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date")
    if resample:
        df = df.resample(resample).mean()

    return df.astype("pint[GW]")


def historical_electricity_demand(source: HistoricalDemandSource = "era5") -> pd.DataFrame:
    """
    Get raw demand data for analysis.

    Args:
        source: The source of demand data, either "era5" or "espeni".

    Returns:
        pd.DataFrame: DataFrame with daily demand values.

    Raises:
        ValueError: If source is not "era5" or "espeni".
    """
    if source == "era5":
        return demand_era5("D")
    if source == "espeni":
        return demand_espeni("D")
    raise ValueError("Invalid source. Choose 'era5' or 'espeni'.")


def historical_gas_demand(*, old_gas_data: bool = False, filter_ldz: bool = True) -> pd.DataFrame:
    """
    Load and return the gas demand data for the UK.

    Args:
        old_gas_data (bool): If True, use the old gas demand data.
        filter_ldz (bool): If True, filter the data for "NTS Energy Offtaken, LDZ Offtake Total".

    Returns:
        pd.DataFrame: DataFrame containing the gas demand data.
    """
    data_dir = Path(__file__).parents[2] / "data"
    if old_gas_data:
        assert not filter_ldz, "Old data does not support filtering by LZD"
        df = pd.read_excel(data_dir / "UKGasDemand2018-17Dec23.xlsx", sheet_name="Sheet1")
        nat_gas_cv = 35.17  # Caloric value of gas in MJ/m3
        df["demand (TWh)"] = df["UK Total Demand (mcm)"] * nat_gas_cv * 1 / 3600
        df["date"] = df["Date"]
    else:
        df = pd.read_csv(data_dir / "new" / "UK_gas_demand_processed.csv", parse_dates=["date"])
        if filter_ldz:
            df = df[df["use"] == "NTS Energy Offtaken, LDZ Offtake Total"]

    # set demand units and rename column
    df["demand (TWh)"] = df["demand (TWh)"].astype("pint[TWh]")
    df = df.rename(columns={"demand (TWh)": "demand"})

    return df.set_index("date")[["demand"]]
