"""Analytic process, conservation, restart and backend regression tests."""

import numpy as np
import pytest

from jhbv.glacier import GlacierState, simulate_glacier, validate_parameters


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    previous = None
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
    yield request.param == "jax"
    if previous is not None:
        jax.config.update("jax_enable_x64", previous)


def forcing(n=1, temp=2.0, precip=0.0, pet=0.0):
    return np.full((n, 1), precip), np.full((n, 1), temp), np.full((n, 1), pet)


def initial(snow=0.0, ice=1000.0):
    def a(v):
        return np.array([float(v)])

    return GlacierState(a(snow), a(0), a(150), a(10), a(10), a(ice), a(0), np.zeros(7))


def test_shared_melt_energy(backend):
    # Snow uses half the day's 2 degree-days; remaining ice melt is 7 mm.
    out, final = simulate_glacier(
        *forcing(),
        [1.0],
        [1.0],
        dict(cfmax=4.0, ice_ddf=7.0, cwh=0.0),
        use_jax=backend,
        initial_state=initial(snow=4.0),
    )
    np.testing.assert_allclose(out.snow_melt, [[4.0]])
    np.testing.assert_allclose(out.ice_melt, [[7.0]])
    np.testing.assert_allclose(final.ice, [993.0])


def test_snow_shields_ice(backend):
    out, _ = simulate_glacier(
        *forcing(), [1.0], [1.0], dict(cfmax=4.0), use_jax=backend, initial_state=initial(snow=20.0)
    )
    np.testing.assert_allclose(out.ice_melt, 0)


def test_no_subzero_melt_despite_negative_phase_threshold(backend):
    out, _ = simulate_glacier(
        *forcing(temp=-1.0, precip=2.0), [1.0], [1.0], dict(tt=-2.0, melt_temp=0.0), use_jax=backend
    )
    np.testing.assert_allclose(out.ice_melt, 0)
    np.testing.assert_allclose(out.snow_melt, 0)
    np.testing.assert_allclose(out.effective_precip, [2.0])


def test_ice_exhaustion_limits_runoff(backend):
    out, final = simulate_glacier(*forcing(n=10, temp=10.0), [1.0], [1.0], initial_ice_mm=5.0, use_jax=backend)
    np.testing.assert_allclose(np.sum(out.ice_melt), 5.0)
    np.testing.assert_allclose(final.ice, 0.0)
    assert np.min(out.states.ice) >= 0


def test_routed_mass_balance_with_mixed_hru_fractions(backend):
    rng = np.random.default_rng(123)
    p = rng.gamma(2.0, 3.0, (100, 3))
    t = rng.normal(1.0, 5.0, (100, 3))
    e = np.full_like(p, 2.0)
    out, _ = simulate_glacier(
        p, t, e, [2.0, 3.0, 4.0], [0.0, 0.4, 1.0], dict(sfcf=1.2, maxbas=4.2), initial_ice_mm=10000.0, use_jax=backend
    )
    np.testing.assert_allclose(out.water_balance_error, 0.0, atol=1e-10)
    for state in out.states:
        assert np.min(state) >= -1e-12


def test_surface_balance_excludes_liquid_routing_storage(backend):
    out, _ = simulate_glacier(*forcing(n=20), [1.0], [1.0], dict(glacier_k=0.01), use_jax=backend)
    # Liquid residence changes drainage, not the surface mass-balance flux.
    np.testing.assert_allclose(out.surface_balance, -np.asarray(out.ice_melt))
    np.testing.assert_allclose(np.diff(out.surface_storage[:, 0]), out.surface_balance[1:, 0])
    assert float(np.asarray(out.states.glacier_water)[-1, 0]) > 0


def test_restart_is_exact(backend):
    p, t, e = forcing(n=20, precip=2.0)
    full, final = simulate_glacier(p, t, e, [1.0], [0.5], use_jax=backend)
    first, state = simulate_glacier(p[:8], t[:8], e[:8], [1.0], [0.5], use_jax=backend)
    last, last_state = simulate_glacier(p[8:], t[8:], e[8:], [1.0], [0.5], use_jax=backend, initial_state=state)
    np.testing.assert_allclose(np.concatenate([first.runoff, last.runoff]), full.runoff, rtol=1e-12)
    for a, b in zip(final, last_state):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_subdaily_melt_budget(backend):
    totals = []
    for hours in [24, 6, 1]:
        state = initial(snow=4.0)._replace(routing_pending=np.zeros(7 * 24 // hours))
        out, _ = simulate_glacier(
            *forcing(n=24 // hours),
            [1.0],
            [1.0],
            dict(cfmax=4.0, ice_ddf=8.0, cwh=0.0),
            timestep_hours=hours,
            use_jax=backend,
            initial_state=state,
        )
        totals.append(np.sum(out.ice_melt))
        np.testing.assert_allclose(out.water_balance_error, 0.0, atol=1e-9)
    np.testing.assert_allclose(totals, [8.0, 8.0, 8.0], atol=1e-10)


def test_surface_mass_balance_uses_fixed_area():
    from jhbv.glacier import surface_mass_balance

    p, t, e = forcing(n=3)
    out, _ = simulate_glacier(p, t, e, [2.0], [1.0], use_jax=False)
    assert surface_mass_balance(out, [2.0], [1.0], 0, 3) == pytest.approx(-0.042)
    with pytest.raises(ValueError):
        surface_mass_balance(out, [2.0], [0.0], 0, 3)


def test_numpy_jax_agree(backend):
    if not backend:
        pytest.skip("Backend comparison runs in the scoped JAX fixture")
    args = forcing(n=50, temp=1.0, precip=3.0, pet=1.0)
    a, _ = simulate_glacier(*args, [1.0], [0.3], use_jax=False)
    b, _ = simulate_glacier(*args, [1.0], [0.3], use_jax=True)
    for x, y in zip(a[:-1], b[:-1]):
        np.testing.assert_allclose(x, y, atol=1e-9)


@pytest.mark.parametrize(
    "params",
    [
        dict(k1=0.01, k2=0.1),
        dict(ice_ddf=5.0, cfmax=6.0),
        dict(melt_temp=-1.0),
        dict(fc=float("nan")),
        dict(smoothing_enabled=True),
    ],
)
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        validate_parameters(params)


@pytest.mark.parametrize(
    "area,fraction,ice",
    [([0.0], [1.0], 100.0), ([1.0], [1.1], 100.0), ([1.0], [1.0], -1.0), ([1.0, 2.0], [1.0], 100.0)],
)
def test_invalid_geometry(area, fraction, ice):
    with pytest.raises(ValueError):
        simulate_glacier(*forcing(), area, fraction, initial_ice_mm=ice, use_jax=False)


def test_config_opt_in_and_legacy_defaults():
    from jhbv.config import HBVConfig

    assert not HBVConfig().glacier_enabled
    assert "ice_ddf" not in HBVConfig().get_default_params()
    with pytest.raises(ValueError):
        HBVConfig(glacier_enabled=True)
    with pytest.raises(ValueError):
        HBVConfig(params_to_calibrate="ice_ddf")
    c = HBVConfig(glacier_enabled=True, glacier_forcing_file="bands.nc", use_gradient_calibration=False)
    assert c.get_default_params()["melt_temp"] == 0.0
    with pytest.raises(ValueError, match="custom states"):
        HBVConfig(
            glacier_enabled=True, glacier_forcing_file="bands.nc", use_gradient_calibration=False, initial_snow=100.0
        )
    with pytest.raises(ValueError, match="non-identifiable"):
        HBVConfig(
            glacier_enabled=True,
            glacier_forcing_file="bands.nc",
            use_gradient_calibration=False,
            params_to_calibrate="melt_temp,temperature_offset",
        )


def test_input_contract_and_output_roundtrip(tmp_path):
    xr = pytest.importorskip("xarray")
    pd = pytest.importorskip("pandas")
    from jhbv.glacier_io import load_glacier_forcing, write_glacier_output

    ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=2), "hru": [1]})
    for k, v, u in [("pr", 1.0, "mm/timestep"), ("temp", 2.0, "degC"), ("pet", 0.0, "mm/timestep")]:
        ds[k] = (("time", "hru"), np.full((2, 1), v))
        ds[k].attrs["units"] = u
    for k, v, u in [("area_km2", 1.0, "km2"), ("glacier_fraction", 1.0, "1"), ("initial_ice_mm", 1000.0, "mm w.e.")]:
        ds[k] = ("hru", [v])
        ds[k].attrs["units"] = u
    p = tmp_path / "input.nc"
    ds.to_netcdf(p)
    loaded = load_glacier_forcing(p, 24)
    out, _ = simulate_glacier(loaded.pr, loaded.temp, loaded.pet, [1.0], [1.0], initial_ice_mm=1000.0, use_jax=False)
    target = write_glacier_output(tmp_path / "output.nc", loaded, out, 24)
    with xr.open_dataset(target) as result:
        np.testing.assert_allclose(result.streamflow, out.runoff / 86.4)
        assert "state_routing_pending" in result and "surface_balance" in result
    ds.pr.attrs["units"] = "kg m-2 s-1"
    ds.to_netcdf(p)
    with pytest.raises(ValueError, match="units"):
        load_glacier_forcing(p, 24)


def test_legacy_config_adapter_roundtrip():
    from jhbv.config import HBVConfigAdapter

    c = HBVConfigAdapter.from_dict(
        dict(
            HBV_GLACIER_ENABLED="true",
            HBV_GLACIER_FORCING_FILE="bands.nc",
            HBV_USE_GRADIENT_CALIBRATION="false",
            HBV_PARAMS_TO_CALIBRATE="tt,ice_ddf",
            HBV_DEFAULT_ICE_DDF=8.0,
        )
    )
    assert c.glacier_enabled and not c.use_gradient_calibration
    assert HBVConfigAdapter.to_dict(c)["HBV_DEFAULT_ICE_DDF"] == 8.0
    assert not HBVConfigAdapter.from_dict({"HBV_GLACIER_ENABLED": "false"}).glacier_enabled


def test_runner_worker_and_core_match(tmp_path):
    import pandas as pd
    import xarray as xr
    from jhbv.runner import HBVRunner
    from jhbv.calibration.worker import HBVWorker
    from jhbv.glacier_io import load_glacier_forcing

    ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=3), "hru": [1]})
    for k, v, u in [("pr", 1.0, "mm/timestep"), ("temp", 2.0, "degC"), ("pet", 0.0, "mm/timestep")]:
        ds[k] = (("time", "hru"), np.full((3, 1), v))
        ds[k].attrs["units"] = u
    for k, v, u in [("area_km2", 1.0, "km2"), ("glacier_fraction", 1.0, "1"), ("initial_ice_mm", 1000.0, "mm w.e.")]:
        ds[k] = ("hru", [v])
        ds[k].attrs["units"] = u
    source = tmp_path / "bands.nc"
    ds.to_netcdf(source)
    cfg = {
        "HBV_GLACIER_ENABLED": True,
        "HBV_GLACIER_FORCING_FILE": str(source),
        "HBV_BACKEND": "numpy",
        "DOMAIN_NAME": "test",
    }
    w = HBVWorker.__new__(HBVWorker)
    w._cfg = lambda key, default=None: cfg.get(key, default)
    w._forcing = None
    w._get_forcing_dir = lambda task: tmp_path
    w.timestep_hours = 24
    w._use_jax = False
    assert w._initialize_model() and not w._use_jax
    assert w._load_forcing()
    runoff = w._run_simulation(w._forcing, {})
    assert not w.supports_native_gradients()
    w.save_output_files(runoff[1:], tmp_path / "worker", ds.time.values[1:])
    r = HBVRunner.__new__(HBVRunner)
    r._get_config_value = lambda fn, default=None, dict_key=None: cfg.get(dict_key, default)
    r._get_default_params = lambda: {}
    r._external_params = None
    r.needs_routing = False
    r._initial_state_override = None
    r.hbv_forcing_dir = tmp_path
    r.timestep_hours = 24
    r.backend = "numpy"
    r.output_dir = tmp_path / "runner"
    r.domain_name = "test"
    assert r._execute_glacier()
    with xr.open_dataset(r.output_dir / "test_hbv_output.nc") as result:
        np.testing.assert_allclose(result.runoff, runoff)
    with xr.open_dataset(tmp_path / "worker/test_hbv_output.nc") as result:
        assert len(result.time) == 2
        np.testing.assert_allclose(result.runoff, runoff[1:])
    bad = ds.copy()
    bad = bad.assign_coords(time=pd.to_datetime(["2000-01-01", "2000-01-03", "2000-01-04"]))
    bad.to_netcdf(tmp_path / "gap.nc")
    with pytest.raises(ValueError, match="interval"):
        load_glacier_forcing(tmp_path / "gap.nc", 24)
