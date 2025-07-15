from pathlib import Path

import matplotlib.pyplot as plt
from config import check

from src.demand_model import combined_seasonality_index, electricity_seasonality_index, gas_seasonality_index

# Constants
DAYS_IN_LEAP_YEAR = 366
MAX_DAILY_CHANGE = 0.1  # Maximum allowed day-to-day change in seasonality index


def test_gas_seasonality_index() -> None:
    """Test the gas seasonality index calculation."""
    df = gas_seasonality_index()

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
    df = electricity_seasonality_index()

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
    df_combined = combined_seasonality_index()

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
    plt.grid()

    # Save the plot as an artifact
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "seasonality_index_comparison.png")
    plt.close()


def test_gas_seasonality_index_filter_lzd() -> None:
    """Test the gas seasonality index with and without LZD filtering."""
    df_filtered = gas_seasonality_index(filter_lzd=True)
    df_unfiltered = gas_seasonality_index(filter_lzd=False)

    # Both should have the same structure
    assert len(df_filtered) == len(df_unfiltered) == DAYS_IN_LEAP_YEAR

    # The values should be different when filtering is applied
    assert not df_filtered["seasonality_index"].equals(df_unfiltered["seasonality_index"])

    # Both should still have mean approximately 1
    check(df_filtered["seasonality_index"].mean(), 1.0)
    check(df_unfiltered["seasonality_index"].mean(), 1.0)


def test_seasonality_index_continuity() -> None:
    """Test that the seasonality indices are smooth and continuous."""
    df_combined = combined_seasonality_index()

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
