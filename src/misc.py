def annualised_cost(capex: float, opex: float, lifetime: int, discount_rate: float) -> float:
    """
    Calculate the annualised cost of a project given its capital expenditure (capex), operational expenditure (opex),
    lifetime in years, and discount rate.

    Args:
        capex (float): Capital expenditure.
        opex (float): Operational expenditure.
        lifetime (int): Lifetime of the project in years.
        discount_rate (float): Discount rate as a decimal.

    Returns:
        float: Annualised cost.
    """
    assert 0 <= discount_rate < 1, "Discount rate must be between 0 and 1"
    annuity_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate
    return capex / annuity_factor + opex
