from src import matplotlib_style  # noqa
import matplotlib.pyplot as plt

import src.assumptions as A
from src import supply_model

from src.units import Units as U
from tests.config import OUTPUT_DIR


def test_fraction_days_without_excess() -> None:
    A.Nuclear.Capacity = 12 * U.GW
    era5_nuclear = supply_model.fraction_days_without_excess("era5", return_mean=True)
    espeni_nuclear = supply_model.fraction_days_without_excess("espeni", return_mean=True)
    A.Nuclear.Capacity = 0 * U.GW
    era5_no_nuclear = supply_model.fraction_days_without_excess("era5", return_mean=True)
    espeni_no_nuclear = supply_model.fraction_days_without_excess("espeni", return_mean=True)

    plt.figure()
    plt.plot(era5_nuclear.index.values, era5_nuclear, label="ERA5 12 GW Nuclear")
    plt.plot(era5_no_nuclear.index.values, era5_no_nuclear, label="ERA5 0 GW Nuclear")
    plt.plot(espeni_nuclear.index.values, espeni_nuclear, ls="--", label="ESPENI 12 GW Nuclear", color="blue")
    plt.plot(espeni_no_nuclear.index.values, espeni_no_nuclear, ls="--", label="ESPENI 0 GW Nuclear", color="orange")
    plt.xlabel("Renewable Capacity (GW)")
    plt.ylabel("Days without Excess Generation")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fraction_days_without_excess.png")
    plt.close()
