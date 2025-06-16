import numpy as np
from config import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE

import src.assumptions as A
from src.misc import annualised_cost


def test_renewable_weighted_average_capacity_factor() -> None:
    expected_value = 0.3064
    # check approximate equality
    assert np.isclose(A.Renewables.WeightedAverageCapacityFactor, expected_value, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), (
        f"Expected {expected_value}, got {A.Renewables.WeightedAverageCapacityFactor}"
    )


def test_annualised_catalyser_cost() -> None:
    # Annualised GBP/GW cost of the catalyser
    annualised_cost_usd_per_kw = annualised_cost(
        A.Catalysers.Capex,
        A.Catalysers.Opex,
        A.Catalysers.Lifetime,
        A.DiscountRate,
    )
    annualised_cost_gbp_per_gw = annualised_cost_usd_per_kw * 1e6 * A.USDToGBP
    expected = 28_364_682
    assert np.isclose(annualised_cost_gbp_per_gw, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), (
        f"Expected {expected}, got {annualised_cost_gbp_per_gw}"
    )


def test_annualised_storage_cost() -> None:
    annualised_storage_cost_gbp_per_twh = (
        annualised_cost(
            A.Storage.Capex,
            A.Storage.Opex,
            A.Storage.Lifetime,
            A.DiscountRate,
        )
        * A.Storage.Efficiency
    )
    expected = 31_987_766
    assert np.isclose(annualised_storage_cost_gbp_per_twh, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), (
        f"Expected {expected}, got {annualised_storage_cost_gbp_per_twh}"
    )
