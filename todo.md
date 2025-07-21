### Related work

- [ ] look at NESO FES 2025 and extract assumptions
    - Extract alternative 2050 demand scenarios
    - Extract heating demand, wind+solar capacity, peak demand
    - Look for heating seasonality assumptions
    - Review workbook "Key Stats" tab
    - Analyze F.15 weekly demand curves (note: 2050 profiles are much flatter than present day)
    - Review "Hydrogen storage and networks" chapter for different storage scenarios
- [x] look at CCC hourly demand data ([source](https://www.theccc.org.uk/wp-content/uploads/2025/05/The-Seventh-Carbon-Budget-methodology-accompanying-data-electricity-supply-hourly-results.xlsx))
    - [x] compare our demand model to CCC

### Demand modelling

- [ ] do better than averaging the gas demand (repeat it, or use a more sophisticated model)
- [ ] look for updated version of ESPENI demand data
- [ ] plug in different CCC demand scenarios
- [ ] make it trivial to switch between seasonal, naive, and CCC demand models
- [ ] use heating degree days for an independent way to model seasonality

### Energy system modelling

- [x] implement storage model
- [x] add losses from transmission and distribution
    - FES 2025 states that transmission losses are around 2% today but increasing to 3% by 2050
    - Distribution losses are higher, typically around 5-8%
    - DUKES 2024 says that 2023 total losses are around 9%
    - If we look at the ratio of demand from the CCC hourly data which accounts for losses and compare with the end use demand, we get around 11.3% total losses
- [ ] add interconnectors
    - Maj looked at this already, found approx 14 GW capacity meeting on average 6% of demand per year
    - Can probably get these numbers from ESPENI dataset, and other sources
- [ ] add dispatchable low carbon generation (gas + CCS)
    - [ ] Review CB7, FES 2025, and RS report assumptions
- [ ] move to hourly time resolution

### Economic modelling

- [ ] add costings for everything
- [ ] cost optimised solutions

### Net zero modelling

- [ ] add Enhanced Rock Weathering for removals
- [ ] add BECCS

### Analysis & visualisation

- [ ] Look at fraction unmet demand instead of fraction days with unmet demand
