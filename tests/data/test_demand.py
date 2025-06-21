from src.data import demand


def test_demand_era5() -> None:
    df = demand.demand_era5(resample="D")
    assert df.shape[0] > 0
    assert "date" in df.columns
    assert "demand" in df.columns
    assert df["demand"].dtype == "float64"
    assert df["date"].dtype == "datetime64[ns]"
    assert df["demand"].min() >= 0.0


def test_demand_espeni() -> None:
    df = demand.demand_espeni(resample="D")
    assert df.shape[0] > 0
    assert "date" in df.columns
    assert "demand" in df.columns
    assert df["demand"].dtype == "float64"
    assert df["date"].dtype == "datetime64[ns, UTC]"
    assert df["demand"].min() >= 0.0
