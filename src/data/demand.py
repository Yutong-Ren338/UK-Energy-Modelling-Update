import pandas as pd

from src import DATA_DIR


def demand_era5(resample: str | None = None) -> pd.DataFrame:
    """
    Load and return the ERA5 full demand data for the UK.

    Arguments:
        resample (str | None): Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.DataFrame: DataFrame containing the demand data in GW.
    """

    # demand is in GW
    df = pd.read_csv(DATA_DIR / "ERA5_full_demand_1979_2018.csv", index_col=0)
    df = df[[c for c in df.columns if "United_Kingdom" in c]]
    df = df.reset_index(names="date")
    df = df.rename(columns={"United_Kingdom_full_demand_no_pop_weights_1979_2018.dat": "demand"})
    df["date"] = pd.to_datetime(df["date"].str.strip("(),'"))

    if resample:
        df = df.set_index("date").resample(resample).mean()
    return df


def demand_espeni(resample: str | None = None) -> pd.DataFrame:
    """
    Load and return the Espeni full demand data for the UK.

    Arguments:
        resample (str | None): Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.DataFrame: DataFrame containing the demand data in GW.
    """

    # demand is in MW
    df = pd.read_csv(DATA_DIR / "espeni.csv", parse_dates=True)
    df = df[["ELEXM_utc", "POWER_ESPENI_MW"]]
    df = df.rename(columns={"ELEXM_utc": "date", "POWER_ESPENI_MW": "demand"})
    df["demand"] /= 1000.0  # convert to GW
    df["date"] = pd.to_datetime(df["date"])

    # resample to daily
    if resample:
        df = df.set_index("date").resample(resample).mean()

    return df
