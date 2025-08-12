import pandas as pd

from src import DATA_DIR


def hdd_era5(resample: str | None = None) -> pd.DataFrame:
    """Load and return the ERA5 heating degree days (HDD) data for the UK.

    Args:
        resample: The resampling frequency (e.g., 'M' for monthly). If None, no resampling is applied.

    Returns:
        pd.DataFrame: The HDD data for the UK
    """
    df = pd.read_csv(DATA_DIR / "ERA5_HDD_all_countries_1979_2019_inclusive.csv", index_col=0, parse_dates=True)
    uk_cols = df.columns[df.columns.str.contains("United_Kingdom")].tolist()
    assert len(uk_cols) == 1, f"Expected 1 column for United Kingdom, found {len(uk_cols)}"
    df = df[uk_cols]
    df = df.rename(columns={uk_cols[0]: "hdd"})
    if resample:
        df = df.resample(resample).mean()
    return df
