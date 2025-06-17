from src.misc import annualised_cost
from src.units import Units as U

# UK Energy Modelling Assumptions
# This module contains key assumptions and parameters used throughout the energy modelling analysis

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
MolecularWeightCO2 = 44.01  # g/mol - Molecular weight of CO2
HoursPerYear = 24 * 365.25 * U.h  # Average hours in a year including for leap years

# ============================================================================
# PROJECTED DEMAND AND EMISSIONS TARGETS
# ============================================================================

EnergyDemand2050 = 575 * U.TWh  # TWh - UK energy demand target for 2050

# CO2 emissions targets and constraints
CO2Emissions2050 = 59  # Mt CO2 - Maximum emissions allowed in 2050
TotalCO2EmissionsCap19902100 = 29636  # Mt CO2 - Total carbon budget 1990-2100
TotalCO2EmissionsCap19902050 = 26626  # Mt CO2 - Total carbon budget 1990-2050
NetNegativeRemovals = 592.720  # Mt CO2 - Yearly removals required to achieve net negative

# ============================================================================
# ECONOMIC PARAMETERS
# ============================================================================

DiscountRate = 0.05  # Discount rate for economic calculations
GBPToUSD = 1.35  # 2024 Average Closing USD/GBP exchange rate
GBPToEuro = 1.18  # Just use the number from the RS report

# ============================================================================
# RENEWABLE ENERGY
# ============================================================================


class Renewables:
    """Parameters for renewable energy technologies including solar, offshore wind, and onshore wind"""

    class CapacityRatios:
        """Mix ratios for different renewable technologies in the energy portfolio"""

        Solar = 0.2  # Share of renewable capacity
        OffshoreWind = 0.56  # Share of renewable capacity
        OnshoreWind = 0.24  # Share of renewable capacity

    class CapacityFactors:
        """Average capacity factors (fraction of nameplate capacity achieved)"""

        Solar = 0.108  # Average capacity factor for solar PV
        OffshoreWind = 0.383  # Capacity factor (61% according to Department for Net Zero)
        OnshoreWind = 0.293  # Capacity factor (45% according to Department for Net Zero)

    class LCOE:
        """Levelized Cost of Energy in 2024 GBP per MWh (Source: BEIS, 2023)"""

        Solar = 30.0 * U.GBP / U.MWh  # £/MWh for solar PV
        OffshoreWind = 41.0 * U.GBP / U.MWh  # £/MWh for offshore wind
        OnshoreWind = 36.0 * U.GBP / U.MWh  # £/MWh for onshore wind

    # Calculated weighted average capacity factor across all renewable technologies
    AverageCapacityFactor = (
        CapacityFactors.Solar * CapacityRatios.Solar
        + CapacityFactors.OffshoreWind * CapacityRatios.OffshoreWind
        + CapacityFactors.OnshoreWind * CapacityRatios.OnshoreWind
    )

    AverageLCOE = LCOE.Solar * CapacityRatios.Solar + LCOE.OffshoreWind * CapacityRatios.OffshoreWind + LCOE.OnshoreWind * CapacityRatios.OnshoreWind


# ============================================================================
# DIRECT AIR CAPTURE (DAC)
# ============================================================================


class DAC:
    """Parameters for Direct Air Capture technology"""

    CarbonStorage = 7.5  # GBP/tonne CO2 - Cost of storing captured CO2

    class EnergyCost:
        """Energy requirements for DAC processes in kJ per mol CO2 captured"""

        Low = 43  # kJ/mol CO2 - Low energy process
        Medium = 101  # kJ/mol CO2 - Medium energy process
        High = 162  # kJ/mol CO2 - High energy process


# ============================================================================
# NUCLEAR POWER
# ============================================================================


class Nuclear:
    """Parameters for nuclear power generation technologies"""

    CapacityFactor = 0.9  # Based on Hinkley Point C performance
    Capacity = 12e3 * U.MW  # GW - Base load capacity of nuclear power

    class CapacityRatios:
        """Mix ratios for different nuclear technologies in the nuclear portfolio"""

        Existing = 0.2  # Existing plants (including Hinkley Point C)
        LargeReactors = 0.2  # New large reactors
        SmallReactors = 0.6  # Small modular reactors (SMRs)

    class LCOE:
        """Levelized Cost of Energy in 2024 GBP per MWh for nuclear technologies"""

        Existing = 130.0 * U.GBP / U.MWh  # £/MWh for existing nuclear plants
        LargeReactors = 80.0 * U.GBP / U.MWh  # £/MWh for new large reactors
        SmallReactors = 60.0 * U.GBP / U.MWh  # £/MWh for small modular reactors

    # Calculated weighted average LCOE across all nuclear technologies
    AverageLCOE = (
        LCOE.Existing * CapacityRatios.Existing
        + LCOE.LargeReactors * CapacityRatios.LargeReactors
        + LCOE.SmallReactors * CapacityRatios.SmallReactors
    )


# ============================================================================
# HYDROGEN STORAGE
# ============================================================================
class Electrolysis:
    """Parameters for hydrogen generation via electrolysis"""

    # Source: IEA via RS report

    Efficiency = 0.74  # Converting electrical energy to hydrogen
    Capex = 450 / GBPToUSD * U.GBP / U.kW  # GBP/kW - Capital expenditure for equipment
    Opex = Capex * 0.015  # GBP/kW - Annual operational expenditure
    Lifetime = 30  # years - Expected operational lifetime

    AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)  # GBP/kW


class Storage:
    """Parameters for large-scale hydrogen storage"""

    # For Capex, H21 NOE assumes £325M for 1.22 TWh. CS Smith et al (2023)
    # take the midpoint of 1-2x this number, which is £399.59M per TWh.

    Efficiency = 0.407  # Round-trip efficiency for storage systems
    Capex = 400 * U.GBP / U.MWh  # GBP/MWh delivered - Capital expenditure for infrastructure
    Opex = Capex * 0.015  # GBP/MWh delivered - Annual operational expenditure
    Lifetime = 30  # years - Expected operational lifetime

    AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)  # GBP/MWh


class Generation:
    """Parameters for electricity generation from stored hydrogen"""

    Efficiency = 0.55  # Efficiency of converting stored hydrogen back to electricity
    Capex = 425 / GBPToUSD * U.GBP / U.kW  # GBP/kW - Capital expenditure for generation equipment
    Opex = Capex * 0.015  # GBP/kW - Annual operational expenditure
    Lifetime = 30  # years - Expected operational lifetime

    AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)  # GBP/kW


# ============================================================================
# MISCELLANEOUS
# ============================================================================
AdditionalCosts = 4 * U.GBP / U.MWh  # GBP/MWh - Transport Rapid Response Costs
