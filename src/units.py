import pint_pandas
from pint import UnitRegistry

ureg = UnitRegistry(auto_reduce_dimensions=True)
pint_pandas.PintType.ureg = ureg

ureg.define("GBP = [currency]")


class Units:
    """Alias some common units for convenience"""

    # dimensionless
    dimensionless = ureg.dimensionless

    # basic
    h = ureg.hour
    y = ureg.year

    # currency
    GBP = ureg.GBP

    # power
    kW = ureg.kilowatt
    MW = ureg.megawatt
    GW = ureg.gigawatt
    TW = ureg.terawatt

    # energy
    kWh = ureg.kilowatt_hour
    MWh = ureg.megawatt_hour
    GWh = ureg.gigawatt_hour
    TWh = ureg.terawatt_hour
