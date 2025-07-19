import matplotlib.pyplot as plt
from config import check

from src import assumptions as A
from src import (
    demand_model,
    matplotlib_style,  # noqa: F401
)
from src.data import cb7
from tests.config import OUTPUT_DIR

# Constants
DAYS_IN_LEAP_YEAR = 366
MAX_DAILY_CHANGE = 0.1  # Maximum allowed day-to-day change in seasonality index


def test_gas_seasonality_index() -> None:
    """Test the gas seasonality index calculation."""
    df = demand_model.gas_seasonality_index()

    # Check that we have the right columns
    assert "day_of_year" in df.columns
    assert "seasonality_index" in df.columns

    # Check that we have 366 days (including leap year)
    assert len(df) == DAYS_IN_LEAP_YEAR

    # Check that day_of_year ranges from 1 to 366
    assert df["day_of_year"].min() == 1
    assert df["day_of_year"].max() == DAYS_IN_LEAP_YEAR

    # Check that seasonality index is positive
    assert (df["seasonality_index"] > 0).all()

    # Check that the mean seasonality index is approximately 1
    check(df["seasonality_index"].mean(), 1.0)

    # Check that winter has higher seasonality index than summer
    winter_months = df[df["day_of_year"].isin(range(1, 90)) | df["day_of_year"].isin(range(335, 367))]
    summer_months = df[df["day_of_year"].isin(range(152, 244))]  # June-August

    assert winter_months["seasonality_index"].mean() > summer_months["seasonality_index"].mean()


def test_electricity_seasonality_index() -> None:
    """Test the electricity seasonality index calculation."""
    df_historical = demand_model.historical_electricity_demand()
    df = demand_model.electricity_seasonality_index(df_historical)

    # Check that we have the right columns
    assert "day_of_year" in df.columns
    assert "seasonality_index" in df.columns

    # Check that we have 366 days (including leap year)
    assert len(df) == DAYS_IN_LEAP_YEAR

    # Check that day_of_year ranges from 1 to 366
    assert df["day_of_year"].min() == 1
    assert df["day_of_year"].max() == DAYS_IN_LEAP_YEAR

    # Check that seasonality index is positive
    assert (df["seasonality_index"] > 0).all()

    # Check that the mean seasonality index is approximately 1
    check(df["seasonality_index"].mean(), 1.0)


def test_combined_seasonality_index() -> None:
    """Test the combined seasonality index and create a plot artifact."""
    df = demand_model.historical_electricity_demand()
    df_combined = demand_model.combined_seasonality_index(df)

    # Check that we have the right columns
    assert "day_of_year" in df_combined.columns
    assert "seasonality_index_gas" in df_combined.columns
    assert "seasonality_index_electricity" in df_combined.columns

    # Check that we have 366 days
    assert len(df_combined) == DAYS_IN_LEAP_YEAR

    # Check that day_of_year ranges from 1 to 366
    assert df_combined["day_of_year"].min() == 1
    assert df_combined["day_of_year"].max() == DAYS_IN_LEAP_YEAR

    # Check that both seasonality indices are positive
    assert (df_combined["seasonality_index_gas"] > 0).all()
    assert (df_combined["seasonality_index_electricity"] > 0).all()

    # Check that both mean seasonality indices are approximately 1
    check(df_combined["seasonality_index_gas"].mean(), 1.0)
    check(df_combined["seasonality_index_electricity"].mean(), 1.0)

    # Check that gas has higher seasonality than electricity (gas demand varies more with weather)
    gas_std = df_combined["seasonality_index_gas"].std()
    electricity_std = df_combined["seasonality_index_electricity"].std()
    assert gas_std > electricity_std

    # Create the plot artifact
    plt.figure()
    plt.plot(df_combined["day_of_year"], df_combined["seasonality_index_gas"], label="Gas Seasonality Index")
    plt.plot(df_combined["day_of_year"], df_combined["seasonality_index_electricity"], label="Electricity Seasonality Index")
    plt.xlabel("Day of Year")
    plt.ylabel("Seasonality Index")
    plt.title("Seasonality Index by Day of Year")
    plt.legend()

    # Save the plot as an artifact
    plt.savefig(OUTPUT_DIR / "seasonality_index_comparison.png")
    plt.close()


def test_gas_seasonality_index_filter_lzd() -> None:
    """Test the gas seasonality index with and without LZD filtering."""
    df_filtered = demand_model.gas_seasonality_index(filter_lzd=True)
    df_unfiltered = demand_model.gas_seasonality_index(filter_lzd=False)

    # Both should have the same structure
    assert len(df_filtered) == len(df_unfiltered) == DAYS_IN_LEAP_YEAR

    # The values should be different when filtering is applied
    assert not df_filtered["seasonality_index"].equals(df_unfiltered["seasonality_index"])

    # Both should still have mean approximately 1
    check(df_filtered["seasonality_index"].mean(), 1.0)
    check(df_unfiltered["seasonality_index"].mean(), 1.0)


def test_seasonality_index_continuity() -> None:
    """Test that the seasonality indices are smooth and continuous."""
    df = demand_model.historical_electricity_demand()
    df_combined = demand_model.combined_seasonality_index(df)

    # Check that there are no large jumps in the data (smoothness test)
    gas_diff = df_combined["seasonality_index_gas"].diff().abs()
    electricity_diff = df_combined["seasonality_index_electricity"].diff().abs()

    # No single day-to-day change should be more than 0.1 (10%)
    assert gas_diff.max() < MAX_DAILY_CHANGE, f"Gas seasonality has large jump: {gas_diff.max()}"
    assert electricity_diff.max() < MAX_DAILY_CHANGE, f"Electricity seasonality has large jump: {electricity_diff.max()}"

    # Check that the beginning and end of year connect smoothly (circular continuity)
    gas_circular_diff = abs(df_combined.iloc[0]["seasonality_index_gas"] - df_combined.iloc[-1]["seasonality_index_gas"])
    electricity_circular_diff = abs(df_combined.iloc[0]["seasonality_index_electricity"] - df_combined.iloc[-1]["seasonality_index_electricity"])

    assert gas_circular_diff < MAX_DAILY_CHANGE, f"Gas seasonality not circular: {gas_circular_diff}"
    assert electricity_circular_diff < MAX_DAILY_CHANGE, f"Electricity seasonality not circular: {electricity_circular_diff}"


def test_demand_scaling_methods() -> None:
    A.CB7EnergyDemand2050Buildings = cb7.buildings_electricity_demand(include_non_residential=True)
    df = demand_model.historical_electricity_demand()
    df["day_of_year"] = df.index.dayofyear
    average_year = (df.groupby("day_of_year")["demand"].mean() * A.HoursPerDay).astype("pint[terawatt_hour]")
    plt.plot(average_year.index, average_year.values, label="Average Historical Demand")

    df_naive = demand_model.historical_electricity_demand()
    df_naive = demand_model.naive_demand_scaling(df_naive)
    plt.plot(df_naive.index, df_naive.values, label="Naive Demand Scaling")

    df_better = demand_model.seasonal_demand_scaling(df)
    plt.plot(df_better.index, df_better.values, label="Seasonal Demand Scaling")

    df_better = demand_model.seasonal_demand_scaling(df, filter_ldz=False)
    plt.plot(df_better.index, df_better.values, label="Seasonal Demand Scaling (No LDZ Filter)")

    A.CB7EnergyDemand2050Buildings = cb7.buildings_electricity_demand(include_non_residential=False)
    df_better = demand_model.seasonal_demand_scaling(df, filter_ldz=False)
    plt.plot(df_better.index, df_better.values, label="Seasonal Demand Scaling (+ No Non-Residential)")

    df_better = demand_model.seasonal_demand_scaling(df, old_gas_data=True, filter_ldz=False)
    plt.plot(df_better.index, df_better.values, label="Seasonal Demand Scaling (+ Old gas data)")

    plt.xlabel("Day of Year")
    plt.ylabel("Electricity Demand (TWh/day)")
    plt.legend(fontsize=8)
    plt.savefig(OUTPUT_DIR / "demand_scaling_comparison.png")
    plt.close()
