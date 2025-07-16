import matplotlib.pyplot as plt
import pandas as pd

from src import matplotlib_style  # noqa: F401
from src.data import demand
from src.units import Units as U  # noqa: F401
from tests.config import OUTPUT_DIR


def test_demand_era5() -> None:
    df = demand.demand_era5(resample="D")
    assert df.shape[0] > 0
    assert "demand" in df.columns
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates


def test_demand_espeni() -> None:
    df = demand.demand_espeni(resample="D")
    assert df.shape[0] > 0
    assert "demand" in df.columns
    assert df["demand"].dtype == "pint[GW]"
    assert df["demand"].min() >= 0.0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.dtype == "datetime64[ns]"
    assert not df.index.has_duplicates


def test_demand_comparisons() -> None:
    era5_df = demand.demand_era5("ME")
    espeni_df = demand.demand_espeni("ME")
    era5_yearly = era5_df.resample("ME").mean().reset_index()
    espeni_yearly = espeni_df.resample("ME").mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(era5_yearly["date"], era5_yearly["demand"], label="ERA5", color="blue")
    plt.plot(espeni_yearly["date"], espeni_yearly["demand"], label="ESPENI", color="orange")
    plt.title("Monthly Electricity Demand Comparison")
    plt.xlabel("Year")
    plt.ylabel("Electricity Demand (GW)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "demand_comparison.png")
    plt.close()
