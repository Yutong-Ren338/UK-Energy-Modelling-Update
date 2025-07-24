import matplotlib.pyplot as plt

import src.assumptions as A
from src import (
    demand_model,
    matplotlib_style,  # noqa: F401
    supply_model,
)
from src.data import renewable_capacity_factors
from src.units import Units as U
from tests.config import OUTPUT_DIR

OUTPUT_PATH = OUTPUT_DIR / "supply_model"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def test_fraction_days_without_excess() -> None:
    demand_era5 = demand_model.predicted_demand(mode="seasonal", historical="era5", average_year=False)
    demand_espeni = demand_model.predicted_demand(mode="seasonal", historical="espeni", average_year=False)

    A.Nuclear.Capacity = 12 * U.GW
    era5_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_era5), return_mean=True)
    espeni_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_espeni), return_mean=True)
    A.Nuclear.Capacity = 0 * U.GW
    era5_no_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_era5), return_mean=True)
    espeni_no_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_espeni), return_mean=True)

    plt.figure()
    plt.plot(era5_nuclear.index.values, era5_nuclear, label="ERA5 12 GW Nuclear")
    plt.plot(era5_no_nuclear.index.values, era5_no_nuclear, label="ERA5 0 GW Nuclear")
    plt.plot(espeni_nuclear.index.values, espeni_nuclear, ls="--", label="ESPENI 12 GW Nuclear", color="blue")
    plt.plot(espeni_no_nuclear.index.values, espeni_no_nuclear, ls="--", label="ESPENI 0 GW Nuclear", color="orange")
    plt.xlabel("Renewable Capacity (GW)")
    plt.ylabel("Days without Excess Generation")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "fraction_days_without_excess.png")
    plt.close()


def test_fraction_days_without_excess_naive_demand() -> None:
    # now a version comparing naive and new demand scaling
    demand_naive = demand_model.predicted_demand(mode="naive", historical="era5", average_year=False)
    demand_seasonal = demand_model.predicted_demand(mode="seasonal", historical="era5", average_year=False)

    A.Nuclear.Capacity = 12 * U.GW
    naive_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_naive), return_mean=True)
    seasonal_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_seasonal), return_mean=True)
    A.Nuclear.Capacity = 0 * U.GW
    naive_no_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_naive), return_mean=True)
    seasonal_no_nuclear = supply_model.fraction_days_without_excess(supply_model.get_net_supply(demand_seasonal), return_mean=True)

    plt.figure()
    plt.plot(naive_nuclear.index.values, naive_nuclear, label="Naive 12 GW Nuclear")
    plt.plot(naive_no_nuclear.index.values, naive_no_nuclear, label="Naive 0 GW Nuclear")
    plt.plot(seasonal_nuclear.index.values, seasonal_nuclear, ls="--", label="Seasonal 12 GW Nuclear", color="blue")
    plt.plot(seasonal_no_nuclear.index.values, seasonal_no_nuclear, ls="--", label="Seasonal 0 GW Nuclear", color="orange")
    plt.xlabel("Renewable Capacity (GW)")
    plt.ylabel("Days without Excess Generation")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "fraction_days_without_excess_naive_demand.png")
    plt.close()


def test_total_unmet_demand() -> None:
    # now a version comparing naive and new demand scaling
    demand_naive = demand_model.predicted_demand(mode="naive", historical="era5", average_year=False)
    demand_seasonal = demand_model.predicted_demand(mode="seasonal", historical="era5", average_year=False)

original_capacity = A.Nuclear.Capacity
try:
    A.Nuclear.Capacity = 12 * U.GW
    naive_nuclear = supply_model.total_unmet_demand(supply_model.get_net_supply(demand_naive))
    seasonal_nuclear = supply_model.total_unmet_demand(supply_model.get_net_supply(demand_seasonal))
    A.Nuclear.Capacity = 0 * U.GW
    naive_no_nuclear = supply_model.total_unmet_demand(supply_model.get_net_supply(demand_naive))
    seasonal_no_nuclear = supply_model.total_unmet_demand(supply_model.get_net_supply(demand_seasonal))
finally:
    A.Nuclear.Capacity = original_capacity

    plt.figure()
    plt.plot(naive_nuclear.index.values, naive_nuclear, label="Naive 12 GW Nuclear")
    plt.plot(naive_no_nuclear.index.values, naive_no_nuclear, label="Naive 0 GW Nuclear")
    plt.plot(seasonal_nuclear.index.values, seasonal_nuclear, ls="--", label="Seasonal 12 GW Nuclear", color="blue")
    plt.plot(seasonal_no_nuclear.index.values, seasonal_no_nuclear, ls="--", label="Seasonal 0 GW Nuclear", color="orange")
    plt.xlabel("Renewable Capacity (GW)")
    plt.ylabel("Total Unmet Demand (TWh)")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "total_unmet_demand.png")
    plt.close()


def test_compare_supply_demand() -> None:
    # supply
    daily_capacity_factors = renewable_capacity_factors.get_renewable_capacity_factors(resample="D")
    supply_df = supply_model.daily_renewables_capacity(300 * U.GW, daily_capacity_factors).to_frame()
    supply_df["day_of_year"] = supply_df.index.dayofyear
    mean = supply_df.groupby("day_of_year").mean().astype(float)
    plt.figure(figsize=(10, 5))
    plt.plot(mean.index, mean, label="Supply")

    # naive demand
    naive_df = demand_model.predicted_demand(mode="naive", average_year=True)
    plt.plot(naive_df.index, naive_df["demand"], label="Naive Demand")

    # seasonal demand
    seasonal_df = demand_model.predicted_demand(mode="seasonal", average_year=True)
    plt.plot(seasonal_df.index, seasonal_df["demand"], label="Seasonal Demand")

    plt.xlabel("Day of Year")
    plt.ylabel("Energy (TWh)")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "compare_supply_demand.png")
    plt.close()
