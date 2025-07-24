from config import check

from src import assumptions as A
from src.costs import yearly_cost
from src.units import Units as U


def test_nuclear_cost_contribution() -> None:
    nuclear_cost = yearly_cost(capacity=A.Nuclear.Capacity, capacity_factor=A.Nuclear.CapacityFactor, lcoe=A.Nuclear.AverageLCOE)
    cost_constribution = nuclear_cost / A.EnergyDemand2050.to(U.MWh)
    expected = 10.67 * U.GBP / U.MWh
    check(cost_constribution, expected)
