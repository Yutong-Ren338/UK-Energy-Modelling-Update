import pandas as pd

from src.renewable_capacity_factors import get_renewable_capacity_factors


def test_get_renewable_capacity_factors_loads_data_correctly():
    """Test that the function loads and combines PV and wind data correctly."""
    result = get_renewable_capacity_factors(rule=None)

    # Check that data was loaded correctly
    assert "pv" in result.columns
    assert "wind" in result.columns
    assert len(result) > 0

    # Check that all values are between 0 and 1 (valid capacity factors)
    assert (result["pv"] >= 0).all()
    assert (result["pv"] <= 1).all()
    assert (result["wind"] >= 0).all()
    assert (result["wind"] <= 1).all()

    # Check that units are applied
    assert str(result.dtypes["pv"]).startswith("pint")
    assert str(result.dtypes["wind"]).startswith("pint")

    # Check that index is datetime
    assert isinstance(result.index, pd.DatetimeIndex)


def test_get_renewable_capacity_factors_with_resampling():
    """Test that the function applies resampling when rule is provided."""
    # Get original hourly data
    hourly_result = get_renewable_capacity_factors(rule=None)

    # Get daily resampled data
    daily_result = get_renewable_capacity_factors(rule="D")

    # Daily data should have fewer rows than hourly
    assert len(daily_result) < len(hourly_result)

    # Check that resampled data still has valid capacity factors
    assert (daily_result["pv"] >= 0).all()
    assert (daily_result["pv"] <= 1).all()
    assert (daily_result["wind"] >= 0).all()
    assert (daily_result["wind"] <= 1).all()

    # Check that units are preserved
    assert str(daily_result.dtypes["pv"]).startswith("pint")
    assert str(daily_result.dtypes["wind"]).startswith("pint")

    # Check that index frequency matches resampling rule
    assert daily_result.index.freq == pd.Timedelta(days=1)
