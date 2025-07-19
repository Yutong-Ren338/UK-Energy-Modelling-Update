import pandas as pd

from src import DATA_DIR
from src.units import Units as U


def get_renewable_capacity_factors(resample: str | None = None) -> pd.DataFrame:
    """Load and return the renewable capacity factors for the UK.

    Args:
        resample: Resampling rule for the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        DataFrame containing the capacity factors for PV and wind.
    """
    pv_fname = DATA_DIR / "ninja_pv_country_GB_merra-2_corrected.csv"
    wind_fname = DATA_DIR / "ninja_wind_country_GB_current-merra-2_corrected.csv"

    # load data
    pv_df = pd.read_csv(pv_fname, index_col=0, parse_dates=True, skiprows=2)
    pv_df = pv_df.rename(columns={"national": "solar"})
    pv_df.index.name = "date"
    wind_df = pd.read_csv(wind_fname, index_col=0, parse_dates=True, skiprows=2)
    wind_df = wind_df.rename(columns={"national": "wind"})
    wind_df.index.name = "date"

    # combine columns
    capacity_factors_df = pv_df.join(wind_df)

    # resample time series
    if resample is not None:
        capacity_factors_df = capacity_factors_df.resample(resample).mean()

    # convert to pint quantities
    return capacity_factors_df.astype(f"pint[{U.dimensionless}]")
