import pandas as pd

from src.units import Units as U


def get_renewable_capacity_factors(rule: str | None) -> pd.DataFrame:
    """
    Load and return the renewable capacity factors for the UK.

    Arguments:
        rule (str): Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.DataFrame: DataFrame containing the capacity factors for PV and wind.
    """

    pv_fname = "data/new/ninja_pv_country_GB_merra-2_corrected.csv"
    wind_fname = "data/new/ninja_wind_country_GB_current-merra-2_corrected.csv"

    # load data
    pv_df = pd.read_csv(pv_fname, index_col=0, parse_dates=True, skiprows=2)
    pv_df = pv_df.rename(columns={"national": "pv"})
    wind_df = pd.read_csv(wind_fname, index_col=0, parse_dates=True, skiprows=2)
    wind_df = wind_df.rename(columns={"national": "wind"})

    # combine columns
    capacity_factors_df = pv_df.join(wind_df)

    # resample time series
    if rule is not None:
        capacity_factors_df = capacity_factors_df.resample(rule).mean()

    # convert to pint quantities
    return capacity_factors_df.astype(f"pint[{U.dimensionless}]")
