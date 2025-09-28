import pandas as pd

from src import DATA_DIR
from src.data.era5 import get_2021_data, get_2024_data
from src.units import Units as U


def get_renewable_capacity_factors(source: str = "renewable_ninja", **kwargs: dict) -> pd.DataFrame:
    """Get renewable capacity factors for solar, onshore wind, and offshore wind.

    Args:
        source: Source of the capacity factors. Options are "renewable_ninja", "era5_2021", or "era5_2024".
        **kwargs: Additional keyword arguments to pass to the data loading functions, e.g. resample frequency.

    Returns:
        DataFrame with datetime index and columns "solar", "onshore", and "offshore".

    Raises:
        ValueError: if source is not one of the expected values.
    """
    if source == "renewable_ninja":
        return get_renewable_ninja(**kwargs)
    if source == "era5_2021":
        pv_df = get_2021_data(generation_type="solar", country_code="UK", **kwargs)
        pv_df = pv_df.rename(columns={"capacity_factor": "solar"})
        onshore = get_2021_data(generation_type="onshore_wind", country_code="UK", **kwargs)
        onshore = onshore.rename(columns={"capacity_factor": "onshore"})
        offshore = get_2021_data(generation_type="offshore_wind", country_code="UK", **kwargs)
        offshore = offshore.rename(columns={"capacity_factor": "offshore"})
        df = pv_df.join(onshore).join(offshore)
        return df.astype(f"pint[{U.dimensionless}]")
    if source == "era5_2024":
        pv_df = get_2024_data(generation_type="solar", country_code="UK", **kwargs)
        pv_df = pv_df.rename(columns={"capacity_factor": "solar"})
        onshore = get_2024_data(generation_type="onshore_wind", country_code="UK", **kwargs)
        onshore = onshore.rename(columns={"capacity_factor": "onshore"})
        offshore = get_2024_data(generation_type="offshore_wind", country_code="UK", **kwargs)
        offshore = offshore.rename(columns={"capacity_factor": "offshore"})
        df = pv_df.join(onshore).join(offshore)
        return df.astype(f"pint[{U.dimensionless}]")
    raise ValueError(f"Unknown source: {source}")


def get_renewable_ninja(resample: str | None = None) -> pd.DataFrame:
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
