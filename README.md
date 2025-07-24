# UK Energy Modelling with CO2 Removal

A Python-based energy system model for simulating the UK's 2050 net-zero energy transition with integrated Direct Air Capture (DAC) capabilities.

## Overview

This model simulates the UK's future energy system using hourly/daily time series data with the following key capabilities:

- **Renewable energy generation** from wind and solar with realistic ERA5-based capacity factors from Renewables.ninja
- **Multiple demand scenarios**: Switch between naive scaling, seasonal patterns with heat pump electrification, and CCC CB7 2050 projections
- **Energy storage systems** with configurable capacity and round-trip efficiency losses
- **Direct Air Capture (DAC)** integration for CO2 removal using excess renewable energy
- **Economic analysis**: Cost modeling and optimization capabilities (in development)

## Getting Started

Install dependencies using `uv`:

```bash
uv install
```

Run tests to ensure everything works:

```bash
uv run pytest
```

Check code style:

```bash
uv run ruff check .
uv run ruff format .
```


## Development Status

### Current Capabilities
- ✅ Core power system simulation with storage
- ✅ Multiple demand modeling approaches 
- ✅ Renewable supply modeling with capacity factors
- ✅ DAC integration for excess energy allocation
- ✅ Transmission/distribution losses (11.3% total)

### Planned Enhancements
- 🔄 Hourly time resolution (currently daily)
- 🔄 Interconnector modeling (28 GW capacity by 2050)
- 🔄 Dispatchable low-carbon generation (gas + CCS)
- 🔄 Medium-term storage options (pumped hydro, compressed air)
- 🔄 Economic optimization and cost modeling

See `todo.md` for detailed development roadmap.

## Data Sources

Historical weather data from ERA5, renewable capacity factors from Renewables.ninja, and demand projections from the CCC Seventh Carbon Budget. See [src/data/README.md](src/data/README.md) for complete data source documentation.

## Related Work

| Title | Author | Date | Type |
|:------|:--------|:-----|:-----|
| [Large-scale electricity storage](https://royalsociety.org/news-resources/projects/low-carbon-energy-programme/large-scale-electricity-storage/) | Royal Society | 2023 | Report |
| [Exploration of Great Britain's Optimal Energy Supply Mixture and Energy Storage Scenarios Upon a Transition to Net-Zero](https://github.com/majmis1/Energy-Transition-Modelling) | Maj Mis | 2024 | Master's thesis |
| [Modelling the UK's 2050 Energy System with Carbon Dioxide Removal](https://github.com/RSuz1/UK-Energy-Model-with-CO2-Removal) | Rei Suzuki | 2025 | Master's thesis |
| [The Seventh Carbon Budget](https://www.theccc.org.uk/publication/the-seventh-carbon-budget/) | CCC | 2025 | Report |
| [Future Energy Scenarios (FES)](https://www.neso.energy/publications/future-energy-scenarios-fes) | NESO | 2025 | Report |
| [Net Zero Power and Hydrogen: Capacity Requirements for Flexibility (AFRY BID3)](https://www.theccc.org.uk/publication/net-zero-power-and-hydrogen-capacity-requirements-for-flexibility-afry/) | CCC | 2023 | Report |
| [Delivering a reliable decarbonised power system](https://www.theccc.org.uk/publication/delivering-a-reliable-decarbonised-power-system/) | CCC | 2023 | Report |
