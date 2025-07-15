from pathlib import Path

import pandas as pd

from src.units import Units as U

TARGET_YEAR = 2050


def frac_heat_demand_from_buildings() -> float:
    """
    Calculate the fraction of energy demand from residential buildings that is for heating.

    Analyzes the Seventh Carbon Budget dataset to determine what proportion of residential
    building energy demand (excluding other home energy use) comes from heating in 2050
    under the Balanced Pathway scenario.

    Returns:
        float: Fraction of energy demand that is for heating (expected ~0.597)
    """
    data_path = Path(__file__).parents[2] / "data" / "new" / "The-Seventh-Carbon-Budget-full-dataset.xlsx"

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
    data_path = Path(__file__).parents[2] / "data" / "new" / "The-Seventh-Carbon-Budget-full-dataset.xlsx"
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
    data_path = Path(__file__).parents[2] / "data" / "new" / "The-Seventh-Carbon-Budget-full-dataset.xlsx"
    df = pd.read_excel(data_path, sheet_name="Economy-wide data")
    df = df[df["scenario"] == "Balanced Pathway"]
    df = df[df["country"] == "United Kingdom"]
    df = df[df["year"] == TARGET_YEAR]
    df = df[df["variable"] == "Energy: final demand electricity"]
    return df["value"].sum() * U.TWh
