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

- [ ] heating degree days seasonality
- [ ] plug in different CCC and NESO FES demand scenarios

### Energy system modelling

- [x] implement storage model
- [ ] add losses from transmission and distribution
- [ ] add interconnectors
    - Maj looked at this already, found approx 14 GW capacity meeting on average 6% of demand per year
- [ ] add dispatchable low carbon generation (gas + CCS)
    - [ ] Review CB7, FES 2025, and RS report assumptions
- [ ] move to hourly time resolution

### Economic modelling

- [ ] add costings for everything
- [ ] cost optimised solutions

### Net zero modelling

- [ ] add Enhanced Rock Weathering for removals