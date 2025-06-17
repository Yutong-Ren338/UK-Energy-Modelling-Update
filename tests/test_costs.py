from config import check

from src import assumptions as A
from src.costs import cost_contribution_per_mwh


def test_nuclear_cost_contribution() -> None:
    nuclear = cost_contribution_per_mwh(
        capacity=A.Nuclear.Capacity, capacity_factor=A.Nuclear.CapacityFactor, lcoe=A.Nuclear.AverageLCOE, demand=A.EnergyDemand2050
    )
    expected = 12.84
    check(nuclear, expected)
