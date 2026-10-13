# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed-area HBV glacier model; daily and sub-daily NumPy/JAX backends.

The legacy model is untouched. Geometry and forcing must be supplied explicitly.
Parameters use daily units; outputs use mm per timestep over the stated areas.
This model represents snow, finite ice and liquid routing, not glacier dynamics.
"""

from typing import NamedTuple

import numpy as np

from .parameters import DEFAULT_PARAMS, PARAM_BOUNDS

GLACIER_DEFAULTS = dict(melt_temp=0.0, ice_ddf=7.0, glacier_k=0.2, precip_multiplier=1.0, temperature_offset=0.0)
GLACIER_BOUNDS = dict(
    melt_temp=(0.0, 1.0),
    ice_ddf=(5.0, 10.0),
    glacier_k=(0.01, 0.95),
    precip_multiplier=(0.8, 1.5),
    temperature_offset=(-1.0, 1.0),
)


class GlacierState(NamedTuple):
    snow: object
    snow_water: object
    sm: object
    suz: object
    slz: object
    ice: object
    glacier_water: object
    routing_pending: object


class GlacierOutput(NamedTuple):
    runoff: object
    raw_runoff: object
    effective_precip: object
    actual_et: object
    snow_melt: object
    ice_melt: object
    surface_balance: object
    surface_storage: object
    water_balance_error: object
    states: GlacierState


def surface_mass_balance(output, area_km2, glacier_fraction, start_step, end_step):
    """Specific solid surface balance (m w.e.) over [start_step, end_step).

    Normalize by the *modeled fixed glacier area*. Liquid reservoirs and
    routing are excluded. Observations must describe the same glacier support
    and interval; do not divide a fixed-area model by a changing observed area.
    """
    a = np.asarray(area_km2) * np.asarray(glacier_fraction)
    flux = np.asarray(output.surface_balance)
    if a.shape != (flux.shape[1],) or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError("Positive glacier area with matching HRUs required")
    if not 0 <= start_step < end_step <= len(flux):
        raise ValueError("Invalid mass-balance interval")
    return float(np.sum(flux[start_step:end_step] @ a) / a.sum() / 1000.0)


def validate_parameters(params):
    """Reject invalid parameters rather than silently sorting recession rates."""
    unknown = set(params) - set(DEFAULT_PARAMS) - set(GLACIER_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown glacier parameters: {sorted(unknown)}")
    p = {**DEFAULT_PARAMS, **GLACIER_DEFAULTS, **params}
    for k, v in p.items():
        if not np.isfinite(v):
            raise ValueError(f"Nonfinite parameter {k}")
    bounds = {**PARAM_BOUNDS, **GLACIER_BOUNDS, "fc": (1.0, 3000.0), "perc": (0.0, 100.0)}
    for k, (lo, hi) in bounds.items():
        if not lo <= p[k] <= hi:
            raise ValueError(f"{k} outside [{lo}, {hi}]")
    if not p["k0"] >= p["k1"] >= p["k2"]:
        raise ValueError("Require k0 >= k1 >= k2")
    if p["ice_ddf"] < p["cfmax"]:
        raise ValueError("Require ice_ddf >= cfmax")
    if p["smoothing_enabled"]:
        raise ValueError("Glacier model uses exact thresholds; smoothing is unsupported")
    return p


def prepare_inputs(precip, temp, pet, area_km2, glacier_fraction, initial_ice_mm, timestep_hours=24):
    """Validate units/shapes and construct explicit geometry for either backend."""
    if not isinstance(timestep_hours, (int, np.integer)) or not 1 <= timestep_hours <= 24:
        raise ValueError("timestep_hours must be an integer in [1, 24]")
    arrays = [np.asarray(a, dtype=float) for a in (precip, temp, pet)]
    if arrays[0].ndim == 1:
        arrays = [a[:, None] for a in arrays]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Forcing must have matching (time, hru) shapes")
    if not all(arrays[0].shape) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Forcing must be nonempty and finite")
    if (arrays[0] < 0).any() or (arrays[2] < 0).any():
        raise ValueError("Precipitation and PET must be nonnegative mm/timestep")
    n = arrays[0].shape[1]
    area = np.asarray(area_km2, dtype=float)
    fraction = np.asarray(glacier_fraction, dtype=float)
    ice = np.broadcast_to(np.asarray(initial_ice_mm, dtype=float), (n,)).copy()
    if area.shape != (n,) or fraction.shape != (n,):
        raise ValueError("Area and glacier fraction must match the HRU axis")
    if not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("HRU areas must be finite and positive")
    if not np.isfinite(fraction).all() or ((fraction < 0) | (fraction > 1)).any():
        raise ValueError("Glacier fractions must be in [0, 1]")
    if not np.isfinite(ice).all() or (ice < 0).any():
        raise ValueError("Initial ice must be finite and nonnegative mm w.e.")
    return (*arrays, area, fraction, np.where(fraction > 0, ice, 0.0))


def _weights(maxbas, n, dt, xp):
    """Exact interval integrals of a continuous triangular unit hydrograph."""
    edges = xp.arange(n + 1) * dt
    x = xp.clip(edges / maxbas, 0.0, 1.0)
    cdf = xp.where(x <= 0.5, 2 * x * x, 1 - 2 * (1 - x) * (1 - x))
    return xp.diff(cdf)


def simulate_glacier_core(
    precip, temp, pet, area, fraction, initial_ice, params, timestep_hours=24, use_jax=True, initial_state=None
):
    """Array-only core for JIT/vmap; caller validates inputs and parameters.

    JAX gradients are piecewise derivatives; the phase and exhaustion branches
    are discontinuous. Gradient calibration is not advertised as smooth.
    Returns (GlacierOutput, final GlacierState). States in output are end-of-step.
    """
    if use_jax:
        import jax.numpy as xp
        from jax import lax
    else:
        xp = np
    p = {**DEFAULT_PARAMS, **GLACIER_DEFAULTS, **params}
    area, fraction, initial_ice = map(xp.asarray, (area, fraction, initial_ice))
    forcing = xp.stack([xp.asarray(precip), xp.asarray(temp), xp.asarray(pet)], axis=-1)
    dt = timestep_hours / 24.0
    n = int(np.ceil(7 / dt))
    weights = _weights(p["maxbas"], n, dt, xp)
    w = area / xp.sum(area)
    z = xp.zeros_like(area)
    if initial_state is None:
        land = fraction < 1
        initial_state = GlacierState(
            z,
            z,
            xp.where(land, xp.minimum(z + 150.0, p["fc"]), 0.0),
            xp.where(land, z + 10.0, 0.0),
            xp.where(land, z + 10.0, 0.0),
            initial_ice,
            z,
            xp.zeros(n),
        )
    rates = {k: 1 - (1 - p[k]) ** dt for k in ["k0", "k1", "k2", "glacier_k"]}

    def storage(s):
        land = s.sm + s.suz + s.slz
        glac = s.ice + s.glacier_water
        return (s.snow + s.snow_water + (1 - fraction) * land + fraction * glac) @ w + xp.sum(s.routing_pending)

    def step(s, met):
        temperature = met[:, 1] + p["temperature_offset"]
        rain = met[:, 0] * p["precip_multiplier"] * (temperature >= p["tt"])
        snowfall = met[:, 0] * p["precip_multiplier"] * p["sfcf"] * (temperature < p["tt"])
        snow_available = s.snow + snowfall
        positive = xp.maximum(temperature - p["melt_temp"], 0.0) * dt
        snow_potential = p["cfmax"] * positive
        snow_melt = xp.minimum(snow_available, snow_potential)
        # A single day's positive-degree budget is shared by snow and ice.
        remaining = xp.where(snow_potential > 0, 1 - snow_melt / xp.maximum(snow_potential, 1e-20), 0.0)
        ice_melt = xp.where(fraction > 0, xp.minimum(s.ice, p["ice_ddf"] * positive * remaining), 0.0)
        liquid = s.snow_water + rain + snow_melt
        refreeze = xp.minimum(liquid, p["cfr"] * p["cfmax"] * xp.maximum(p["melt_temp"] - temperature, 0.0) * dt)
        snow = snow_available - snow_melt + refreeze
        liquid -= refreeze
        water = xp.maximum(liquid - p["cwh"] * snow, 0.0)
        liquid -= water
        recharge = water * xp.minimum(s.sm / p["fc"], 1.0) ** p["beta"]
        sm = s.sm + water - recharge
        excess = xp.maximum(sm - p["fc"], 0.0)
        sm -= excess
        recharge += excess
        et = xp.minimum(met[:, 2] * xp.minimum(sm / (p["lp"] * p["fc"]), 1.0), sm)
        sm -= et
        upper = s.suz + recharge
        perc = xp.minimum(p["perc"] * dt, upper)
        upper -= perc
        lower = s.slz + perc
        q0 = rates["k0"] * xp.maximum(upper - p["uzl"], 0.0)
        q1 = rates["k1"] * upper
        land_q = q0 + q1 + rates["k2"] * lower
        uz = upper - q0 - q1
        lz = lower * (1 - rates["k2"])
        gw = s.glacier_water + water + ice_melt
        glacier_q = rates["glacier_k"] * gw
        gw -= glacier_q
        raw = ((1 - fraction) * land_q + fraction * glacier_q) @ w
        scheduled = s.routing_pending + raw * weights
        routed = scheduled[0]
        pending = xp.concatenate([scheduled[1:], xp.zeros(1)])
        new = GlacierState(
            snow,
            liquid,
            xp.where(fraction < 1, sm, 0.0),
            xp.where(fraction < 1, uz, 0.0),
            xp.where(fraction < 1, lz, 0.0),
            s.ice - ice_melt,
            xp.where(fraction > 0, gw, 0.0),
            pending,
        )
        effective = (rain + snowfall) @ w
        actual_et = ((1 - fraction) * et) @ w
        error = effective - actual_et - routed - (storage(new) - storage(s))
        out = GlacierOutput(
            routed,
            raw,
            effective,
            actual_et,
            snow_melt,
            ice_melt,
            snowfall + refreeze - snow_melt - ice_melt,
            snow + new.ice,
            error,
            new,
        )
        return new, out

    if use_jax:
        final, output = lax.scan(step, initial_state, forcing)
    else:
        state = initial_state
        outputs = []
        for met in forcing:
            state, out = step(state, met)
            outputs.append(out)
        final = state
        states = GlacierState(*(np.stack([o.states[i] for o in outputs]) for i in range(len(state))))
        output = GlacierOutput(*(np.stack([o[i] for o in outputs]) for i in range(len(out) - 1)), states)
    return output, final


def simulate_glacier(
    precip,
    temp,
    pet,
    area_km2,
    glacier_fraction,
    params=None,
    initial_ice_mm=100000.0,
    timestep_hours=24,
    use_jax=True,
    initial_state=None,
):
    """Validated opt-in entrypoint; ice depth is mm water equivalent, not metres.

    Land and glacier fractions share the HRU snow forcing. Glacier runoff bypasses
    the land soil reservoir. No glacier-area change, firn conversion, sublimation,
    calving or glacier ice flow is represented. PET applies to land only.
    """
    arrays = prepare_inputs(precip, temp, pet, area_km2, glacier_fraction, initial_ice_mm, timestep_hours)
    p = validate_parameters(params or {})
    if use_jax:
        try:
            import jax  # noqa: F401
        except ImportError:
            use_jax = False
    if initial_state is not None:
        if not isinstance(initial_state, GlacierState):
            raise ValueError("Restart requires a GlacierState")
        n = len(arrays[3])
        expected = [(n,)] * 7 + [(int(np.ceil(7 / (timestep_hours / 24.0))),)]
        if any(np.shape(a) != shape for a, shape in zip(initial_state, expected)):
            raise ValueError("Restart state dimensions do not match geometry/timestep")
        if not all(np.isfinite(a).all() and (np.asarray(a) >= 0).all() for a in initial_state):
            raise ValueError("Restart states must be finite and nonnegative")
    return simulate_glacier_core(*arrays, p, timestep_hours, use_jax, initial_state)
