import pandas as pd

from src.data import demand


def test_demand_era5() -> None:
    df = demand.demand_era5(resample="D")
    assert df.shape[0] > 0
    assert "demand" in df.columns
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.index.has_duplicates


def test_demand_espeni() -> None:
    df = demand.demand_espeni(resample="D")
    assert df.shape[0] > 0
    assert "demand" in df.columns
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.index.has_duplicates
