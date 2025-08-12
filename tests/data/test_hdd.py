import matplotlib.pyplot as plt
import pandas as pd

from src.data.hdd import hdd_era5
from tests.config import OUTPUT_DIR

OUTPUT_PATH = OUTPUT_DIR / "data" / "hdd"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def test_hdd_era5() -> None:
    df = hdd_era5()
    assert df is not None
    assert not df.empty
    assert len(df.columns) == 1
    assert df.index.dtype == "datetime64[ns]"
    assert isinstance(df, pd.DataFrame)


def test_hdd_era5_resample() -> None:
    df_raw = hdd_era5()
    df = hdd_era5(resample="ME")
    assert df is not None
    assert not df.empty
    assert len(df.columns) == 1
    assert df.index.dtype == "datetime64[ns]"
    assert isinstance(df, pd.DataFrame)
    assert len(df) < len(df_raw)


def test_hdd_era5_plot() -> None:
    df = hdd_era5()
    plt.figure()
    plt.plot(df.index, df["hdd"], label="HDD2")
    plt.xlabel("Date")
    plt.ylabel("ERA5 HDD")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "hdd_era5.png")
    plt.close()
