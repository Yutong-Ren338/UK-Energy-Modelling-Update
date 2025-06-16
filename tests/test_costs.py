import numpy as np
from config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE

from src import assumptions as A
from src.costs import cost_contribution_per_mhw


def test_nuclear_cost_contribution() -> None:
    nuclear = cost_contribution_per_mhw(
        capacity=A.Nuclear.Capacity, capacity_factor=A.Nuclear.CapacityFactor, lcoe=A.Nuclear.AverageLCOE, demand=A.EnergyDemand2050
    )
    expected = 12.8
    np.isclose(nuclear, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
