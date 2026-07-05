# jHBV

[![PyPI version](https://img.shields.io/pypi/v/jhbv.svg)](https://pypi.org/project/jhbv/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Dual-backend (JAX + NumPy) implementation of the HBV-96 rainfall–runoff model — usable standalone or as a SYMFLUENCE plugin.

Part of the SYMFLUENCE JAX-native model family — self-contained packages that
run standalone (NumPy fallback, no JAX required) and register automatically with
[SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE) when installed alongside it.

## Features

- **Differentiable**: automatic differentiation through the full simulation (JAX)
- **Fast**: JIT compilation via `lax.scan`; `vmap` for ensembles; GPU-capable
- **Dependency-light**: pure-NumPy fallback when JAX is not installed
- **Plugin architecture**: auto-registers with SYMFLUENCE via entry points

## Installation

```bash
pip install jhbv          # NumPy backend
pip install 'jhbv[jax]'    # with JAX (differentiable, JIT)
```

## Quickstart

```python
import numpy as np
from jhbv import simulate

# daily forcing: precipitation (mm/d), temperature (degC), PET (mm/d)
runoff, state = simulate(precip, temp, pet)           # default parameters

params = {"FC": 250.0, "BETA": 2.0, "K1": 0.1}        # override any subset
runoff, state = simulate(precip, temp, pet, params=params)

# hourly simulation (parameters stay in daily units)
runoff_h, _ = simulate(precip_h, temp_h, pet_h, timestep_hours=1)
```

## Gradient-based calibration

The JAX backend makes the full simulation differentiable end-to-end, so model
parameters can be calibrated with gradient descent:

```python
import jax
from jhbv import kge_loss, get_kge_gradient_fn

grad_fn = get_kge_gradient_fn(precip, temp, pet, observed)
value, grads = grad_fn(params)          # dKGE/dparam for every parameter
```

`nse_loss` / `kge_loss` and their gradient factories are JIT-compatible and work
with any `optax` optimizer. Within SYMFLUENCE the same interface powers the
ADAM and L-BFGS calibration options.

## Use with SYMFLUENCE

jhbv registers with [SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
through the `symfluence.plugins` entry point — installation is the integration:

```bash
pip install symfluence jhbv
```

```yaml
# config.yaml (excerpt)
model:
  hydrological_model: HBV
```

SYMFLUENCE then handles forcing preparation, calibration, evaluation, and
benchmarking for the model with no further wiring.

## Model structure

HBV-96 (Lindström et al., 1997) consists of four routines, all implemented with
smooth, differentiable formulations:

1. **Snow** — degree-day accumulation and melt with refreezing
2. **Soil moisture** — beta-function recharge and evapotranspiration reduction
3. **Response** — two-box (upper/lower zone) storage with percolation
4. **Routing** — triangular transfer-function convolution

15 calibration parameters (`jhbv.PARAM_BOUNDS`), with defaults in `jhbv.DEFAULT_PARAMS`.
Daily and sub-daily timesteps are supported; parameters are specified in daily units
and scaled internally (`timestep_hours` argument).

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## How to cite

If you use jHBV in your research, please cite the SYMFLUENCE companion papers,
which describe the design of the JAX-native model family (registry integration,
differentiability, and the calibration experiments they enable):

> Eythorsson, D., et al. (2026). The registry as social contract: Architectural patterns
> for community hydrological modeling. *Water Resources Research* (submitted).
>
> Eythorsson, D., et al. (2026). From configuration to prediction: Multi-model,
> multi-basin experiments with SYMFLUENCE. *Water Resources Research* (submitted).

Citation metadata for this package is provided in [`CITATION.cff`](CITATION.cff);
a version-specific DOI is minted via Zenodo for each GitHub release.
<!-- After the first Zenodo release, add the concept-DOI badge here. -->

## References

- Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997). Development and test of the distributed HBV-96 hydrological model. Journal of Hydrology, 201(1–4), 272–288. https://doi.org/10.1016/S0022-1694(97)00041-3
- Bergström, S. (1995). The HBV model. In V. P. Singh (Ed.), Computer Models of Watershed Hydrology (pp. 443–476). Water Resources Publications.

## License

Apache-2.0. See [LICENSE](LICENSE).
