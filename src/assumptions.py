from src.units import Units as U
from src.utils import annualised_cost, convert_energy_cost

# UK Energy Modelling Assumptions
# This module contains key assumptions and parameters used throughout the energy modelling analysis

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
MolecularWeightCO2 = 44.01 * U.g / U.mol  # g/mol
HoursPerDay = 24 * U.h
HoursPerYear = HoursPerDay * 365.25  # Including leap years

# ============================================================================
# PROJECTED DEMAND AND EMISSIONS TARGETS
# ============================================================================

CB7EnergyDemand2050 = 692 * U.TWh  # Seventh Carbon Budget energy demand target for 2050
EnergyDemand2050 = CB7EnergyDemand2050
CB7EnergyDemand2050Buildings = 355.44 * U.TWh  # Seventh Carbon Budget energy demand for buildings in 2050
CB7FractionHeatDemandBuildings = 0.597  # Fraction of energy demand from buildings that is for heating in 2050

FractionHeatDemandSpaceHeating = 0.65  # Fraction of building heating demand that is for space heating - This is an unmotivated assumption for now
CB7FractionSpaceHeatDemandBuildings = CB7FractionHeatDemandBuildings * FractionHeatDemandSpaceHeating  # only space heating

# CO2 emissions targets and constraints
CO2Emissions2050 = 59  # Mt CO2 - Maximum emissions allowed in 2050
TotalCO2EmissionsCap19902100 = 29636  # Mt CO2 - Total carbon budget 1990-2100
TotalCO2EmissionsCap19902050 = 26626  # Mt CO2 - Total carbon budget 1990-2050
NetNegativeRemovals = 592.720  # Mt CO2 - Yearly removals required to achieve net negative

# ============================================================================
# ECONOMIC PARAMETERS
# ============================================================================

DiscountRate = 0.05
GBPToUSD = 1.35  # 2024 Average Closing USD/GBP exchange rate
GBPToEuro = 1.18  # Just use the number from the RS report


# ============================================================================
# POWER SYSTEM
# ============================================================================
class PowerSystem:
    # Transmission and distribution losses in the energy system
    # This is taken by taking the ratio of the CB7 hourly demand data (which is supply side),
    # to the demand in the main CB7 workbook (which is end use demand).
    # Other sources:
    # - FES 2025 states that transmission losses are around 2% today but increasing to 3% by 2050
    # - DUKES 2024 says that 2023 total losses are around 9%
    TotalLosses = 0.113


class DispatchableGasCCS:
    Capacity = 18 * U.GW  # CB7 (Table 7.5.1): 38 GW total capacity (shared with generation from hydrogen). FES 2025 (Table 32) has 48.3 GW.
    LCOE = 180 * U.GBP / U.MWh  # from CB7 (Table 7.5.1): 165-194 £/MWh. Price here includes generation from hydrogen so this is not accurate.


# ============================================================================
# RENEWABLE ENERGY
# ============================================================================


class Renewables:
    """Parameters for renewable energy technologies including solar, offshore wind, and onshore wind."""

    class CapacityRatios:
        """Mix ratios for different renewable technologies in the energy portfolio.

        Source: CB7 Table 7.5.1
        """

        Solar = 0.4
        OffshoreWind = 0.45
        OnshoreWind = 0.15

    class CapacityFactors:
        """Average capacity factors (fraction of nominal capacity achieved).

        Note that these numbers are not used by the core simulation, they are obtained
        as a time series from renewables.ninja instead.
        """

        Solar = 0.108
        OffshoreWind = 0.383  # 61% according to Department for Net Zero
        OnshoreWind = 0.293  # 45% according to Department for Net Zero

    class LCOE:
        """Levelized Cost of Energy (Source: BEIS, 2023)."""

        Solar = 30.0 * U.GBP / U.MWh  # CB7 (Table 7.5.1): 27 £/MWh
        OffshoreWind = 41.0 * U.GBP / U.MWh  # CB7 (Table 7.5.1): 31 £/MWh
        OnshoreWind = 36.0 * U.GBP / U.MWh

    # Calculated weighted average capacity factor across all renewable technologies
    AverageCapacityFactor = (
        CapacityFactors.Solar * CapacityRatios.Solar
        + CapacityFactors.OffshoreWind * CapacityRatios.OffshoreWind
        + CapacityFactors.OnshoreWind * CapacityRatios.OnshoreWind
    )

    AverageLCOE = LCOE.Solar * CapacityRatios.Solar + LCOE.OffshoreWind * CapacityRatios.OffshoreWind + LCOE.OnshoreWind * CapacityRatios.OnshoreWind


# ============================================================================
# NUCLEAR POWER
# ============================================================================


class Nuclear:
    """Parameters for nuclear power generation technologies."""

    Capacity = 12 * U.GW
    CapacityFactor = 0.9  # Based on Hinkley Point C performance

    class CapacityRatios:
        """Mix ratios for different nuclear technologies in the nuclear portfolio."""

        Existing = 0.2  # Including Hinkley Point C
        LargeReactors = 0.2
        SmallReactors = 0.6  # Small modular reactors (SMRs)

    class LCOE:
        """Levelized Cost of Energy for nuclear technologies."""

        Existing = 130.0 * U.GBP / U.MWh
        LargeReactors = 80.0 * U.GBP / U.MWh
        SmallReactors = 60.0 * U.GBP / U.MWh  # Small modular reactors

    # Calculated weighted average LCOE across all nuclear technologies
    AverageLCOE = (
        LCOE.Existing * CapacityRatios.Existing
        + LCOE.LargeReactors * CapacityRatios.LargeReactors
        + LCOE.SmallReactors * CapacityRatios.SmallReactors
    )


# ============================================================================
# MEDIUM-TERM ENERGY STORAGE
# ============================================================================
class MediumTermStorage:
    # from CB7 (Table 7.5.1)
    Power = 7 * U.GW  # FES 2025 (Table 28) has 39.3 GW
    Capacity = 0.433 * U.TWh  # Energy capacity of the medium-term storage system
    RoundTripEfficiency = 0.70  # Round-trip efficiency for medium-term storage
    LCOE = 100 * U.GBP / U.MWh  # just a placeholder for now, need to update


# ============================================================================
# HYDROGEN STORAGE
# ============================================================================
class HydrogenStorage:
    """Parameters for hydrogen energy storage system including electrolysis, cavern storage, and generation."""

    class Electrolysis:
        """Parameters for hydrogen generation via electrolysis."""

        # Source: IEA via RS report

        Power = 40 * U.GW  # Electrolyser power capacity
        Efficiency = 0.74  # Converting electrical energy to hydrogen
        Capex = 450 / GBPToUSD * U.GBP / U.kW
        Opex = Capex * 0.015
        Lifetime = 30  # years

        AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)

    class CavernStorage:
        """Parameters for large-scale hydrogen storage in underground caverns."""

        # For Capex, H21 NOE assumes £325M for 1.22 TWh. CS Smith et al (2023)
        # take the midpoint of 1-2x this number, which is £399.59M per TWh.

        Capacity = 50.0 * U.TWh  # CB7 has 5-9 TWh, FES 2025 (Table 38) has 12 TWh.
        Efficiency = 0.407  # Round-trip efficiency (electrolysis * generation efficiencies
        Capex = 400 * U.GBP / U.MWh
        Opex = Capex * 0.015
        Lifetime = 30  # years

        AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)

    class Generation:
        """Parameters for electricity generation from stored hydrogen."""

        Power = 100 * U.GW  # FES 2025 (Table 29) has 16.5 GW (+ constraints on combined capacity with gas+CCS, see above)
        Efficiency = 0.55  # Converting stored hydrogen back to electricity
        Capex = 425 / GBPToUSD * U.GBP / U.kW
        Opex = Capex * 0.015
        Lifetime = 30  # years

        AnnualisedCost = annualised_cost(Capex, Opex, Lifetime, DiscountRate)


# ============================================================================
# DIRECT AIR CAPTURE (DAC)
# ============================================================================


class DAC:
    """Parameters for Direct Air Capture technology."""

    # System capacity parameters
    Capacity = 1.1 * U.GW

    CarbonStorage = 7.5  # GBP/tonne CO2

    class EnergyCost:
        """Energy requirements for DAC processes."""

        Low = 43 * U.kJ / U.mol
        Medium = 101 * U.kJ / U.mol
        High = 162 * U.kJ / U.mol

        # Energy cost per unit CO2 removed (TWh/Mt)
        LowTWhPerMtCO2 = convert_energy_cost(Low, MolecularWeightCO2)
        MediumTWhPerMtCO2 = convert_energy_cost(Medium, MolecularWeightCO2)
        HighTWhPerMtCO2 = convert_energy_cost(High, MolecularWeightCO2)


# ============================================================================
# MISCELLANEOUS
# ============================================================================
AdditionalCosts = 4 * U.GBP / U.MWh  # Transport Rapid Response Costs
