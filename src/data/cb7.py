from pathlib import Path

import pandas as pd

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
        (df["sector"] == "Residential buildings")
        & (df["year"] == TARGET_YEAR)
        & (df["variable"] == "Energy: gross demand electricity")
        & (df["scenario"] == "Balanced Pathway")
    )
    df_filtered = df[mask]

    # Calculate fraction excluding "Other home energy use"
    heating_demand = df_filtered[df_filtered["subsector"] != "Other home energy use"]["value"].sum()
    total_demand = df_filtered["value"].sum()

    return heating_demand / total_demand
