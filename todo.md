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
- [x] plug in CCC CB7 2050 demand scenarios
- [x] make it trivial to switch between seasonal, naive, and CCC demand models
- [ ] use heating degree days for an independent way to model seasonality

### Supply modelling

- [ ] Granular supply model using renewables ninja


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
    - CB7 says 28 GW capacity by 2050
- [ ] add dispatchable low carbon generation (gas + CCS)
    - Review CB7, FES 2025, and RS report assumptions
    - In CB7 it's about 18 GW (20 GW is cited for electrolyser and total low carbon dispatchable generation is 38 GW). They also emphasise that the exact tradeoff between gas and hydrogen generation is uncertain, and will depend on the evolution of costs and efficiencies.
- [ ] move to hourly time resolution
- [ ] Medium term storage (CB7): A range of other options can provide storage over the medium term (days-to-weeks), including pumped hydro and other technologies at different stages of commercialisation (for example, compressed and liquid air storage, flow batteries, and thermal storage). Our analysis deploys 7 GW of medium-duration grid storage by 2050, (433 GWh of storage capacity).
- [x] use numba to speed up the core simulation loop
- [ ] use multiprocessing for parallel runs

### Economic modelling

- [ ] add costings for everything
- [ ] cost optimised solutions

### Net zero modelling

- [ ] add Enhanced Rock Weathering for removals
- [ ] add BECCS

### Analysis & visualisation

- [x] Look at total unmet demand instead of fraction days with unmet demand
- [ ] Also plot unmet demand as a function of day of year
- [ ] Improve visualisation of single simulation result
- [ ] Do those 40 year plots but X axis is a single year, to see if there are yearly trends 
