from pathlib import Path

import pandas as pd

from src.units import Units as U

TARGET_YEAR = 2050
DATA_PATH = Path(__file__).parents[2] / "data" / "new"


def frac_heat_demand_from_buildings() -> float:
    """
    Calculate the fraction of energy demand from residential buildings that is for heating.

    Analyzes the Seventh Carbon Budget dataset to determine what proportion of residential
    building energy demand (excluding other home energy use) comes from heating in 2050
    under the Balanced Pathway scenario.

    Returns:
        float: Fraction of energy demand that is for heating (expected ~0.597)
    """
    data_path = DATA_PATH / "The-Seventh-Carbon-Budget-full-dataset.xlsx"

    df = pd.read_excel(data_path, sheet_name="Subsector-level data")

    # Filter for residential buildings in 2050 under Balanced Pathway scenario
    mask = (
        (df["scenario"] == "Balanced Pathway")
        & (df["year"] == TARGET_YEAR)
        & (df["sector"] == "Residential buildings")
        & (df["variable"] == "Energy: gross demand electricity")
    )
    df_filtered = df[mask]

    # Calculate fraction excluding "Other home energy use"
    heating_demand = df_filtered[df_filtered["subsector"] != "Other home energy use"]["value"].sum()
    total_demand = df_filtered["value"].sum()

    return heating_demand / total_demand


def buildings_electricity_demand(*, include_non_residential: bool = True) -> float:
    """
    Calculate the total electricity demand for UK residential and non-residential buildings in 2050 in TWh.

    Args:
        include_non_residential (bool): If True, includes non-residential buildings in the calculation.

    Returns:
        float: Total electricity demand for buildings in 2050 in TWh.
    """
    data_path = DATA_PATH / "The-Seventh-Carbon-Budget-full-dataset.xlsx"
    df = pd.read_excel(data_path, sheet_name="Sector-level data")
    df = df[df["scenario"] == "Balanced Pathway"]
    df = df[df["country"] == "United Kingdom"]
    df = df[df["year"] == TARGET_YEAR]
    sectors = ["Residential buildings"]
    if include_non_residential:
        sectors = ["Residential buildings", "Non-residential buildings"]
    df = df[df["sector"].isin(sectors)]
    df = df[df["variable"] == "Energy: final demand electricity"]
    return df["value"].sum() * U.TWh


def total_demand_2050() -> float:
    """
    Calculate the total electricity demand for the UK in 2050 in TWh.

    Returns:
        float: Total energy demand for buildings in 2050 in TWh.
    """
    data_path = DATA_PATH / "The-Seventh-Carbon-Budget-full-dataset.xlsx"
    df = pd.read_excel(data_path, sheet_name="Economy-wide data")
    df = df[df["scenario"] == "Balanced Pathway"]
    df = df[df["country"] == "United Kingdom"]
    df = df[df["year"] == TARGET_YEAR]
    df = df[df["variable"] == "Energy: final demand electricity"]
    return df["value"].sum() * U.TWh


def extract_daily_2050_demand() -> None:
    """
    Extract the daily electricity demand for 2050 from the Seventh Carbon Budget dataset.

    The dataset contains hourly demand data for different weather years, save it as daily for convenience.
    Note: the demands here are "generation level", rather than "end use" leve, which means they are around 11% large,
    taking into account transmission and distribution losses.

    """
    demand_year = 2050
    df = pd.read_excel(
        DATA_PATH / "The-Seventh-Carbon-Budget-methodology-accompanying-data-electricity-supply-hourly-results.xlsx",
        sheet_name="Data",
        skiprows=4,
    )
    df = df.loc[df["Year"] == demand_year]

    # remove column
    df = df.drop(columns=["Unnamed: 20"])

    # convert Year and Hour columns to datetime
    df["hour_in_day"] = df["Hour"] % 24
    df["day_of_year"] = df["Hour"] // 24
    df["datetime"] = pd.to_datetime(df["Year"], format="%Y") + pd.to_timedelta(df["Hour"], unit="h")

    # resample each weather year to daily sums and combine
    dfs = {}
    for weather_year in df["Weather year"].unique():
        df_ = df[df["Weather year"] == weather_year].copy()
        df_ = df_.resample("D", on="datetime")["Electricity demand without electrolysis"].sum().reset_index()
        df_ = df_.rename(columns={"Electricity demand without electrolysis": "demand (TWh)"})
        df_["weather year"] = weather_year
        dfs[weather_year] = df_
    df_combined = pd.concat(dfs.values(), ignore_index=True)

    # remove rows with datetime with year != demand_year
    df_combined = df_combined[df_combined["datetime"].dt.year == demand_year]

    # save as csv
    df_combined.to_csv(DATA_PATH / f"ccc_daily_demand_{demand_year}.csv", index=False)
