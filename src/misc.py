def annualised_cost(capex: float, opex: float, lifetime: int, discount_rate: float, efficiency: float = 1.0) -> float:
    """
    Calculate the annualised cost of a project given its capital expenditure (capex), operational expenditure (opex),
    lifetime in years, and discount rate.

    Args:
        capex (float): Capital expenditure.
        opex (float): Annual operational expenditure.
        lifetime (int): Lifetime of the project in years.
        discount_rate (float): Discount rate as a decimal.
    efficiency (float): Efficiency factor for the project.

    Returns:
        float: Annualised cost.
    """
    assert 0 <= discount_rate < 1, "Discount rate must be between 0 and 1"
    assert lifetime > 0, "Lifetime must be greater than 0"
    assert 0 < efficiency <= 1, "Efficiency must be between 0 and 1"

    if discount_rate == 0:
        return capex / lifetime + opex

    annuity_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate
    annualised_capex = capex / annuity_factor
    annualised_cost = annualised_capex + opex
    return annualised_cost / efficiency
