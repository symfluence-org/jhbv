# Optional fixed-area glacier module

Glacier modeling is **off by default**. Existing `simulate`, parameter structures,
lumped/distributed models and their outputs retain their existing behavior.
`simulate_glacier` is a separate, explicitly selected model structure. It is suitable
for historical discharge/surface-balance experiments with known glacier geometry;
it does not simulate glacier retreat or constitute a climate-projection model.

## Equations and assumptions

Each elevation HRU supplies precipitation, temperature and PET, an area and a
glacier fraction. Land and ice fractions share snow forcing; glacier water bypasses
the land soil reservoir. Snowfall is `P * precip_multiplier * sfcf`; rainfall is
`P * precip_multiplier`. `tt` partitions rain/snow. A separate `melt_temp` (0–1°C)
controls snow/ice melt and refreezing. There is no ice melt below that threshold.
An optional, explicit `temperature_offset` (−1…+1 K, default zero) corrects the
forcing temperature before both phase and melt calculations. PET is supplied
independently and is not recomputed. Fix `melt_temp` when calibrating the offset:
jointly fitting TT, melt threshold and offset is non-identifiable. These broad
software bounds are not evidence of temperature bias at a particular basin.

For timestep length `dt` in days, `D = max(T - melt_temp, 0) * dt`,
`Ms = min(SWE + snowfall, cfmax * D)` and
`Mi = min(ice, ice_ddf * D * (1 - Ms / (cfmax * D)))`, with zero ice melt
when `D=0`. Snow therefore shields ice and consumes its share of a single melt
budget on the snow-exhaustion timestep. `ice_ddf >= cfmax` is required. Snow liquid
retention and refreezing follow the HBV conceptual structure.

The land fraction uses a field-capacity soil reservoir, beta recharge, moisture
limited ET, percolation and two groundwater stores. Water exceeding field capacity
enters recharge. Glacier water drains through a separate linear reservoir. Daily
outflow fractions use `k_dt = 1 - (1-k_daily)**dt` at sub-daily resolution.
Recession coefficients must be supplied in ordered form (`k0 >= k1 >= k2`);
glacier mode rejects unordered values instead of silently reassigning them.

The catchment outlet uses interval-integrated triangular routing over `maxbas`
days. Pending future discharge is an explicit storage, included in the daily
ledger: effective precipitation − ET − outlet runoff − change in all stores.
Effective precipitation includes both precipitation and snowfall corrections.

`surface_balance = snowfall + refreeze - snow_melt - ice_melt` is the change in
**solid snow + ice**, per unit HRU glacier area. `surface_mass_balance` integrates
that flux over a stated half-open timestep interval and divides by the modeled
fixed glacier area. Snow liquid, glacier runoff storage and outlet routing are
excluded. Compare only observations with compatible spatial support and dates;
an annual balance labeled by year does not establish exact September-end dates.
Do not normalize a fixed-area simulation by a changing observed glacier area.

Finite initial ice is specified in **mm water equivalent**, not metres of physical
ice thickness. Supply an evidence-based depth or document sensitivity to the
assumption. Snow is retained as snow: firn conversion, ice flow, area/elevation
change, sublimation, calving and glacier evaporation are not represented.
Glacier-water drainage can be temperature independent; its parameter is conceptual.

## Standalone API

```python
from jhbv.glacier import simulate_glacier, surface_mass_balance

# pr and pet: (time, hru), mm/timestep; temp: degC
output, state = simulate_glacier(
    pr, temp, pet, area_km2=[20., 5.], glacier_fraction=[0., 1.],
    initial_ice_mm=[0., 200_000.],
    params=dict(melt_temp=0., ice_ddf=7., glacier_k=.2),
    timestep_hours=24, use_jax=False,
)
# output.runoff: basin mm/timestep. State arrays are end-of-step.
# Exact continuation: pass initial_state=state with unchanged geometry/timestep.
balance_mwe = surface_mass_balance(output, [20., 5.], [0., 1.], 0, len(pr))
```

NumPy is the default scientific reference for numerical checks. For JAX, enable
`jax_enable_x64` **in the application** when auditing deep ice stores; subtraction
of large float32 ice inventories limits ledger precision. The library does not
change global JAX precision. `simulate_glacier_core` is the JIT/vmap entrypoint for
already-validated arrays. Its hard thresholds have piecewise gradients; smooth
gradient calibration is not supported by the SYMFLUENCE glacier worker.

## SYMFLUENCE configuration

```yaml
model:
  hydrological_model: HBV
  routing_model: none
  hbv:
    glacier_enabled: true
    glacier_forcing_file: glacier_bands.nc
    spatial_mode: lumped
    backend: jax
    timestep_hours: 24
    use_gradient_calibration: false
    default_melt_temp: 0.0
    default_ice_ddf: 7.0
    default_glacier_k: 0.2
    default_precip_multiplier: 1.0
    default_temperature_offset: 0.0
    params_to_calibrate: tt,cfmax,sfcf,fc,lp,beta,k0,k1,k2,uzl,perc,maxbas,ice_ddf,glacier_k,precip_multiplier
```

The file is absolute or relative to the domain's `data/forcing/HBV_input`.
Preprocessing validates this explicit file rather than inventing glacier geometry
or replacing elevation-band forcing with basin averages. Network/external routing
is unsupported in glacier mode. The core supports explicit `GlacierState`
restarts; legacy HBV runner restart files are incompatible and rejected.
The runner uses zero initial snow, land soil moisture `min(150, FC)` and 10 mm
in each land groundwater store. Non-default legacy initial-state settings are
rejected in glacier mode; customize them through the explicit core state API.
Outputs retain every timestep for the ledger, including sub-daily states.

Required NetCDF variables (dimensions and **exact unit attributes**):

| Variable | Dimensions | `units` |
|---|---|---|
| `pr` | time, hru | mm/timestep |
| `temp` | time, hru | degC |
| `pet` | time, hru | mm/timestep |
| `area_km2` | hru | km2 |
| `glacier_fraction` | hru | 1 |
| `initial_ice_mm` | hru | mm w.e. |

Times must be unique, increasing and regularly spaced at the configured interval.
The module never silently resamples forcing. Geometry must match the HRU axis;
areas are positive and glacier fractions lie in [0,1]. Partial glacier HRUs are
supported. For the geometry-matched land-only control set all fractions to zero
and use the same forcing/PET and routing implementation.

Runner/final-worker NetCDF output contains streamflow, every state, pending routing
water, raw runoff, effective precipitation, ET, snow/ice melt, solid surface storage,
surface balance and the **full routed** water-balance residual. `save_states` does
not suppress these diagnostics in glacier mode. The generic worker objective is
discharge only; a multi-observation objective must be explicitly constructed from
these diagnostics. Mass-balance weights are not observation uncertainties.

## Validation before interpreting a fitted model

The synthetic suite checks melt-energy sharing, snow shielding, ice exhaustion,
phase/melt separation, fractional areas, routed closure, exact continuation,
sub-daily melt budgets, backend agreement and input/output contracts. For a basin
application also assess snow cover independently (SWE is not fractional snow
cover), glacier balance, forcing corrections, multiple calibration starts,
initial-depth and objective-weight sensitivity, and temporal-resolution effects.
Use matching spatial forcing/PET in ablations. Selecting on calibration data does
not make evaluation periods untouched once they have guided development.

## Scientific precedents

- Toum et al. (2021), *The HBV.IANIGLA Hydrological Model*,
  https://journal.r-project.org/articles/RJ-2021-059/
- Stahl et al. (2008), *Coupled modelling of glacier and streamflow response to
  future climate scenarios*, https://doi.org/10.1029/2007WR005956

This is a documented conceptual HBV extension, not an exact reproduction of
HBV-EC or HBV.IANIGLA. Its equations and omissions are stated above.
