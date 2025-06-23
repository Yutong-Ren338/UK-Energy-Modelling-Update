import pandas as pd

import src.assumptions as A


def daily_renewables_capacity(renewable_capacity: float, capacity_factors: pd.DataFrame) -> pd.DataFrame:
    solar = renewable_capacity * A.Renewables.CapacityRatios.Solar * capacity_factors["solar"]
    offshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OffshoreWind * capacity_factors["offshore"]
    onshore_wind = renewable_capacity * A.Renewables.CapacityRatios.OnshoreWind * capacity_factors["onshore"]
    return solar + offshore_wind + onshore_wind + A.Nuclear.Capacity * A.Nuclear.CapacityFactor
