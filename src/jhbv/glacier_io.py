# SPDX-License-Identifier: Apache-2.0
"""Explicit, validated NetCDF contract for the opt-in glacier runner."""

from pathlib import Path

import numpy as np

from .glacier import prepare_inputs


def load_glacier_forcing(path, timestep_hours):
    """Read (time,hru) forcing and static HRU geometry; never infer/resample units."""
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(path) as source:
        ds = source.load()
    for name in ["pr", "temp", "pet"]:
        if name not in ds or ds[name].dims != ("time", "hru"):
            raise ValueError(f"{name} must have dimensions (time, hru)")
        expected = "degC" if name == "temp" else "mm/timestep"
        if ds[name].attrs.get("units") != expected:
            raise ValueError(f"{name} units must be {expected}")
    for name, units in [("area_km2", "km2"), ("glacier_fraction", "1"), ("initial_ice_mm", "mm w.e.")]:
        if name not in ds or ds[name].dims != ("hru",):
            raise ValueError(f"{name} must have dimensions (hru,)")
        if ds[name].attrs.get("units") != units:
            raise ValueError(f"{name} units must be {units}")
    times = pd.DatetimeIndex(ds.time.values)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Forcing timestamps must be unique and increasing")
    if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(hours=timestep_hours)).all():
        raise ValueError("Forcing interval differs from timestep_hours")
    if len(np.unique(ds.hru.values)) != len(ds.hru):
        raise ValueError("HRU identifiers must be unique")
    prepare_inputs(
        ds.pr.values,
        ds.temp.values,
        ds.pet.values,
        ds.area_km2.values,
        ds.glacier_fraction.values,
        ds.initial_ice_mm.values,
        timestep_hours,
    )
    return ds


def write_glacier_output(path, ds, output, timestep_hours, time_index=None):
    """Write every state and diagnostic, including the routed water ledger."""
    import xarray as xr

    coords = dict(time=ds.time.values, hru=ds.hru.values)
    result = xr.Dataset(coords=coords)
    area = float(ds.area_km2.sum())
    result["streamflow"] = ("time", np.asarray(output.runoff) * area * 1e6 / (1000 * timestep_hours * 3600))
    result.streamflow.attrs["units"] = "m3/s"
    for name in output._fields[:-1]:
        arr = np.asarray(getattr(output, name))
        dims = ("time",) if arr.ndim == 1 else ("time", "hru")
        result[name] = (dims, arr)
        result[name].attrs["units"] = "mm" if name == "surface_storage" else "mm/timestep"
    for name in output.states._fields:
        arr = np.asarray(getattr(output.states, name))
        dims = ("time", "routing_lag") if name == "routing_pending" else ("time", "hru")
        result["state_" + name] = (dims, arr)
        result["state_" + name].attrs["units"] = "mm"
    for name in ["area_km2", "glacier_fraction", "initial_ice_mm"]:
        result[name] = ds[name]
    result.attrs.update(
        model="HBV with opt-in fixed-area glacier",
        catchment_area_km2=area,
        timestep_hours=timestep_hours,
        qualification="Fixed area; no firn conversion or glacier dynamics; PET applies to land only",
        surface_balance_definition="snowfall + refreeze - snow_melt - ice_melt; excludes liquid routing storage",
    )
    if time_index is not None:
        result = result.sel(time=time_index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(path)
    return path
