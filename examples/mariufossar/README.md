# Maríufossar · corrected JHBV glacier module

| configuration | Cal_KGE | Eval_KGE | Dry_KGE | Eval_NSE | Eval_bias_pct |
| --- | --- | --- | --- | --- | --- |
| Old HBV | 0.679 | 0.519 | 0.498 | 0.193 | -19.107 |
| Matched land control | 0.746 | 0.628 | 0.518 | 0.424 | -6.363 |
| JHBV glacier V2 | 0.835 | 0.705 | 0.623 | 0.617 | -2.333 |

The selected seed is 2026, using only calibration WY2013–16: KGE − 0.1 × annual solid glacier surface-balance RMSE (m w.e.). The flux operator permits all four complete WY2013–16 balances to be used, including the first calibration year. Seeds 42/123/2026 converge closely. DE: 200 generations, population factor 6; bounded Nelder–Mead refinement. Evaluation WY2017–21 and dry WY2022–23 are retrospective transfer checks; these periods have been inspected during development.

The opt-in module separates rain/snow TT from the 0°C melt threshold, shares each timestep’s positive-degree budget between snow and ice, retains finite ice, bypasses land soil/ET on glaciers, uses interval-integrated outlet routing, and reports every water store and flux. Surface balance excludes glacier runoff and routing reservoirs. The matched land-only control uses identical geometry, forcing, PET, threshold/routing implementation and search budget, with glacier fraction zero and its own calibration.

All 12 prespecified final gates pass. Independent native JAX runner versus NumPy differs by 2.56e-13 m³/s. Maximum full routed ledger residual is 2.15e-11 mm/timestep; solid-state/flux residual is 5.78e-11 mm. Glacier MB RMSE: calibration 0.235, retrospective WY2017–22 0.508 m w.e.

The 100/200/400 m w.e. ice-depth experiments give identical discharge: even the 100 m case retains 17.7 m w.e. in the most depleted HRU. This establishes insensitivity over this historical run, not a measured initial thickness. Objective weights 0.05/0.1/0.2 give similar discharge. Six-hour and hourly uniformly disaggregated inputs close the ledger, but evaluation KGE drops from 0.705 to 0.684/0.675; the parameters are selected for daily use, not validated for hourly forecasts.

Snow is assessed against LamaH-Ice MODIS outside glaciers with SWE thresholds 1/5/20 mm. The 5 mm proxy has RMSE 0.246/0.241/0.261 and r 0.887/0.876/0.853 (Cal/Eval/Dry). This establishes seasonal agreement, not a validated fractional snow-cover observation operator. Dry-period snow coverage has only 305 paired observations; missing/invalid observations are excluded. Snow proxies were not fitted to select the seed.

Qualifications: total precipitation multiplier 1.250; temperature correction +1 K reaches its bound; TT −1.170°C; snow and ice DDF both ≈6.158 mm/K/day (ice ≥ snow constraint active); FC ≈1451 mm is high, and k0/refreezing also approach bounds. These are effective conceptual parameters, not independent measurements of forcing bias or material properties. Spring flood peaks remain underestimated. Glacier area is fixed at 129.426 km² from the inherited RGI-derived allocation; dynamic-area references and fixed September intervals create spatial/date approximation. No firn conversion, glacier flow, sublimation, calving or area evolution is represented. This is a qualified historical diagnostic member, not climate-projection certification.


Inputs are not redistributed. See calibrate.py and ../../docs/glacier.md for reproduction.

Data source: https://essd.copernicus.org/articles/16/2741/2024/
