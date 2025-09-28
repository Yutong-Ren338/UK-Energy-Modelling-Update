from pathlib import Path

import pandas as pd
import xarray as xr

DATA_DIR = Path(__file__).parents[2] / "data"


def get_2024_data(generation_type: str = "solar", country_code: str = "UK", resample: str | None = None) -> pd.DataFrame:
    if country_code == "UK":
        country_code = "GB"

    base = DATA_DIR / "ERA5_2024"
    if generation_type == "solar":
        path = base / "solar_capacity_factor" / f"{country_code}__ERA5__solar__capacity_factor_time_series.nc"
    elif generation_type == "onshore_wind":
        path = base / "wind_capacity_factor" / f"{country_code}__ERA5__wind__capacity_factor_time_series__onshore.nc"
    elif generation_type == "offshore_wind":
        path = base / "wind_capacity_factor" / f"{country_code}__ERA5__wind__capacity_factor_time_series__offshore.nc"
    else:
        raise ValueError(f"Unknown generation type: {generation_type}")

    ds = xr.open_dataset(path)
    df = ds.to_dataframe()
    df.columns = ["capacity_factor"]

    if resample is not None:
        df = df.resample(resample).mean()

    return df


def get_2021_data(generation_type: str = "solar", country_code: str = "UK", resample: str | None = None) -> pd.DataFrame:
    base = DATA_DIR / "ERA5_2021"
    if generation_type == "solar":
        path = base / "solar_power_capacity_factor" / "NUTS_0_sp_historical.nc"
    elif generation_type == "onshore_wind":
        path = base / "wp_onshore" / "NUTS_0_wp_ons_sim_0_historical_loc_weighted.nc"
    elif generation_type == "offshore_wind":
        path = base / "wp_offshore" / "NUTS_0_wp_ofs_sim_0_historical_loc_weighted.nc"
    else:
        raise ValueError(f"Unknown generation type: {generation_type}")

    # open the dataset
    ds = xr.open_dataset(path)

    # Select the data for country of interest
    ds = ds.sel(NUTS=ds.NUTS[ds.NUTS_keys == country_code])

    # convert the time_in_hours_from_first_jan_1950 data array to a datetime index
    ds["time"] = xr.date_range(start="1950-01-01", periods=ds.sizes["time"], freq="h", use_cftime=True)
    ds = ds.set_index(time="time")

    # drop all columns except timeseries_data
    ds = ds.drop_vars(["NUTS_keys", "time_in_hours_from_first_jan_1950"])

    # drop the NUTS dimension
    ds = ds.squeeze(dim="NUTS")

    # convert to pandas dataframe
    df = ds.to_dataframe()

    # convert CFTimeIndex to pandas DateTimeIndex
    df.index = pd.to_datetime(df.index.to_datetimeindex())

    df.columns = ["capacity_factor"]

    if resample is not None:
        df = df.resample(resample).mean()

    return df
