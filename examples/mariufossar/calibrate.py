"""Reproducible fixed-area HBV glacier/control experiment on explicit inputs.

CLI: python calibrate.py --forcing forcing.nc --observations observations.csv
 --balances glacier_timeseries.csv --snow snow_cover.csv --out results
Requires scipy, pandas, xarray and JAX. See docs/glacier.md for the model contract.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from jhbv.glacier import simulate_glacier_core, simulate_glacier, surface_mass_balance
from jhbv.glacier_io import load_glacier_forcing, write_glacier_output

PERIODS = {
    "Cal": ("2012-10-01", "2016-09-30"),
    "Eval": ("2016-10-01", "2021-09-30"),
    "Dry": ("2021-10-01", "2023-09-30"),
}
NAMES = [
    "tt",
    "cfmax",
    "sfcf",
    "cfr",
    "cwh",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "uzl",
    "perc",
    "maxbas",
    "precip_multiplier",
    "temperature_offset",
    "ice_ddf",
    "glacier_k",
]
BOUNDS = [
    (-3, 3),
    (2.5, 7),
    (0.8, 1.3),
    (0, 0.1),
    (0, 0.2),
    (50, 1500),
    (0.3, 1),
    (1, 6),
    (0.05, 0.5),
    (0.01, 0.3),
    (0.0001, 0.1),
    (0, 100),
    (0, 40),
    (1, 7),
    (0.9, 1.5),
    (-1, 1),
    (5, 10),
    (0.01, 0.95),
]


def metric(q, y):
    r = np.corrcoef(q, y)[0, 1]
    alpha = q.std() / y.std()
    beta = q.mean() / y.mean()
    return dict(
        KGE=float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
        r=float(r),
        alpha=float(alpha),
        beta=float(beta),
        NSE=float(1 - np.sum((q - y) ** 2) / np.sum((y - y.mean()) ** 2)),
    )


def main():
    ap = argparse.ArgumentParser()
    for k in ["forcing", "observations", "balances", "snow", "out"]:
        ap.add_argument("--" + k, required=True, type=Path)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_glacier_forcing(args.forcing, 24)
    dates = pd.DatetimeIndex(ds.time.values)
    area = ds.area_km2.values
    frac = ds.glacier_fraction.values
    ice = ds.initial_ice_mm.values
    obs = pd.read_csv(args.observations, index_col=0, parse_dates=True).iloc[:, 0].reindex(dates).values
    cal = (dates >= PERIODS["Cal"][0]) & (dates <= PERIODS["Cal"][1])
    end = np.flatnonzero(cal)[-1] + 1
    assert np.isfinite(obs[cal]).all()
    annual = pd.read_csv(args.balances, sep=";", index_col=0)
    intervals = []
    for y, row in annual.iterrows():
        left, right = pd.Timestamp(int(y) - 1, 10, 1), pd.Timestamp(int(y), 9, 30)
        if left in dates and right in dates and np.isfinite(row.annual_net_MB):
            intervals.append(
                dict(
                    year=int(y),
                    start=dates.get_loc(left),
                    end=dates.get_loc(right) + 1,
                    observed=float(row.annual_net_MB),
                    observed_area_km2=float(row.g_area_dyn),
                )
            )
    cal_intervals = [r for r in intervals if r["start"] >= np.flatnonzero(cal)[0] and r["end"] <= end]
    assert len(cal_intervals) >= 3
    forcing = np.stack([ds.pr.values, ds.temp.values, ds.pet.values], axis=-1)
    F = jnp.asarray(forcing[:end])
    Y = jnp.asarray(obs[cal])
    ix = np.flatnonzero(cal)
    A = jnp.asarray(area)
    ICE = jnp.asarray(ice)

    def parameters(x):
        return {**dict(zip(NAMES, x)), "melt_temp": 0.0}

    @jax.jit
    def loss(x, fraction, mb_weight):
        o, _ = simulate_glacier_core(F[:, :, 0], F[:, :, 1], F[:, :, 2], A, fraction, ICE, parameters(x), 24, True)
        q = o.runoff[ix] * area.sum() / 86.4
        r = jnp.corrcoef(q, Y)[0, 1]
        alpha = jnp.std(q) / jnp.std(Y)
        beta = jnp.mean(q) / jnp.mean(Y)
        a = A * fraction
        denom = jnp.maximum(a.sum(), 1e-10)
        errors = jnp.stack(
            [
                jnp.sum(o.surface_balance[z["start"] : z["end"]] @ a) / denom / 1000 - z["observed"]
                for z in cal_intervals
            ]
        )
        return jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) + mb_weight * jnp.sqrt(jnp.mean(errors**2))

    def objective(x, fraction, weight):
        if not x[8] >= x[9] >= x[10] or x[16] < x[1]:
            return 5.0
        return float(loss(jnp.asarray(x), jnp.asarray(fraction), weight))

    protocol = dict(
        periods=PERIODS,
        primary_objective="KGE_cal - 0.1 RMSE_solid_MB_cal_mwe",
        melt_temp=0.0,
        seeds=[42, 123, 2026],
        bounds=dict(zip(NAMES, BOUNDS)),
        control="Same band geometry, forcing, PET, thresholds and routing; all glacier fractions zero; separately recalibrated on KGE_cal",
        sensitivities=dict(
            ice_depth_mwe=[100, 200, 400],
            timestep_hours=[24, 6, 1],
            mb_weights=[0.05, 0.1, 0.2],
            snow_proxy_thresholds_mm=[1, 5, 20],
        ),
        promotion_gates=dict(
            finite_nonnegative=True,
            routed_balance_mm=1e-7,
            ice_exhaustion=False,
            eval_and_dry_KGE_improve_current=True,
            holdout_MB_RMSE_mwe=0.75,
            snow_proxy_RMSE=0.30,
            snow_proxy_correlation=0.5,
        ),
        qualification="Historical diagnostic member; fixed area, assumed depth, snow cover is a threshold proxy; evaluation periods already inspected during development",
    )
    protocol_path = args.out / "protocol.json"
    if args.replay:
        rows = json.loads((args.out / "candidates.json").read_text())
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
        rows = []
        for variant in ["glacier", "land_control"]:
            fraction = frac if variant == "glacier" else np.zeros_like(frac)
            weight = 0.1 if variant == "glacier" else 0.0
            for seed in protocol["seeds"]:
                start = time.time()
                bounds = BOUNDS if variant == "glacier" else BOUNDS[:16] + [(7.0, 7.0), (0.2, 0.2)]

                def f(x):
                    return objective(x, fraction, weight)

                counter = [0]

                def progress(x, convergence):
                    counter[0] += 1
                    if counter[0] % 40 == 0:
                        print(variant, seed, "generation", counter[0], "objective", 1 - f(x), flush=True)

                de = differential_evolution(
                    f, bounds, seed=seed, popsize=6, maxiter=200, tol=1e-4, polish=False, callback=progress
                )
                local = minimize(
                    f,
                    de.x,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"maxiter": 2500, "fatol": 1e-7, "xatol": 1e-6},
                )
                x = local.x if local.fun < de.fun else de.x
                row = dict(
                    variant=variant,
                    seed=seed,
                    weight=weight,
                    objective=1 - f(x),
                    params=parameters(x.tolist()),
                    nfev=de.nfev + local.nfev,
                    seconds=time.time() - start,
                )
                rows.append(row)
                (args.out / "candidates.json").write_text(json.dumps(rows, indent=2))
                print("FINISHED", row, flush=True)
        # Freeze primary selection before any holdout diagnostics are calculated.
        selected = {
            v: max([r for r in rows if r["variant"] == v], key=lambda r: r["objective"])
            for v in ["glacier", "land_control"]
        }
        (args.out / "selection.json").write_text(json.dumps(selected, indent=2))
        winner = selected["glacier"]
        x0 = np.array([winner["params"][k] for k in NAMES])
        lo, hi = np.array(BOUNDS).T
        for weight in [0.05, 0.2]:
            rng = np.random.default_rng(42)
            x = x0.copy()
            score = objective(x, frac, weight)
            start = time.time()
            for n in range(6000):
                probability = 1 - np.log(n + 1) / np.log(6000)
                active = rng.random(len(x)) < probability
                if not active.any():
                    active[rng.integers(len(x))] = True
                trial = x.copy()
                trial[active] += 0.15 * (hi - lo)[active] * rng.normal(size=active.sum())
                trial = np.clip(trial, lo, hi)
                candidate = objective(trial, frac, weight)
                if candidate < score:
                    x, score = trial, candidate
            row = dict(
                variant="weight_sensitivity",
                seed=42,
                weight=weight,
                objective=1 - score,
                params=parameters(x.tolist()),
                seconds=time.time() - start,
            )
            rows.append(row)
            (args.out / "candidates.json").write_text(json.dumps(rows, indent=2))
    selected = json.loads((args.out / "selection.json").read_text())
    daily = {"Observed": obs}
    evaluated = []
    snow = pd.read_csv(args.snow, sep=";")
    snow_obs = pd.Series(
        snow.fsca_outside_glaciers.values / 100.0,
        index=pd.to_datetime(dict(year=snow.YYYY, month=snow.MM, day=snow.DD)),
    ).reindex(dates)
    for row in rows:
        fraction = frac if row["variant"] != "land_control" else np.zeros_like(frac)
        output, _ = simulate_glacier(
            ds.pr.values, ds.temp.values, ds.pet.values, area, fraction, row["params"], ice, use_jax=False
        )
        q = np.asarray(output.runoff) * area.sum() / 86.4
        key = f"{row['variant']}_s{row['seed']}_w{row['weight']}"
        daily[key] = q
        record = {**row, "metrics": {}, "glacier_balance": [], "snow_proxy": {}}
        for period, (left, right) in PERIODS.items():
            mask = (dates >= left) & (dates <= right)
            assert np.isfinite(obs[mask]).all()
            record["metrics"][period] = metric(q[mask], obs[mask])
        if fraction.sum() > 0:
            for z in intervals:
                value = surface_mass_balance(output, area, fraction, z["start"], z["end"])
                record["glacier_balance"].append(dict(**z, simulated=value))
        land_weights = area * (1 - frac)
        land_weights /= land_weights.sum()
        for threshold in [1, 5, 20]:
            cover = (np.asarray(output.states.snow) > threshold) @ land_weights
            record["snow_proxy"][str(threshold)] = {}
            for period, (left, right) in PERIODS.items():
                mask = (dates >= left) & (dates <= right) & snow_obs.between(0, 1).values
                y = snow_obs.values[mask]
                s = cover[mask]
                record["snow_proxy"][str(threshold)][period] = dict(
                    RMSE=float(np.sqrt(np.mean((s - y) ** 2))), r=float(np.corrcoef(s, y)[0, 1]), n=int(mask.sum())
                )
        record["max_routed_balance_error_mm"] = float(np.max(np.abs(output.water_balance_error)))
        record["minimum_ice_mwe"] = (
            float(np.min(np.asarray(output.states.ice)[:, fraction > 0]) / 1000) if fraction.sum() > 0 else None
        )
        evaluated.append(record)
        if row["variant"] in selected and row["seed"] == selected[row["variant"]]["seed"]:
            write_glacier_output(
                args.out / row["variant"] / "hbv_output.nc", ds.assign(glacier_fraction=("hru", fraction)), output, 24
            )
    pd.DataFrame(daily, index=dates).to_csv(args.out / "daily_discharge.csv", index_label="time")
    # Sensitivities use frozen primary parameters; no selection on these scores.
    primary = selected["glacier"]
    p = primary["params"]
    sens = []

    @jax.jit
    def resolution_run(f, hru_ice):
        o, _ = simulate_glacier_core(f[:, :, 0], f[:, :, 1], f[:, :, 2], A, jnp.asarray(frac), hru_ice, p, 24, True)
        return o.runoff, o.water_balance_error, o.states.ice

    for depth in [100, 200, 400]:
        q, error, remaining = map(
            np.asarray, resolution_run(jnp.asarray(forcing), jnp.where(jnp.asarray(frac) > 0, depth * 1000.0, 0.0))
        )
        sens.append(
            dict(
                kind="ice_depth",
                depth_mwe=depth,
                max_ledger_mm=float(np.max(np.abs(error))),
                minimum_ice_mwe=float(remaining[:, frac > 0].min() / 1000),
                metrics={
                    k: metric(q[(dates >= a) & (dates <= b)] * area.sum() / 86.4, obs[(dates >= a) & (dates <= b)])
                    for k, (a, b) in PERIODS.items()
                },
            )
        )
    for hours in [6, 1]:
        repeats = 24 // hours
        fine = np.repeat(forcing, repeats, axis=0)
        fine[:, :, 0] /= repeats
        fine[:, :, 2] /= repeats

        def finer(f):
            o, _ = simulate_glacier_core(f[:, :, 0], f[:, :, 1], f[:, :, 2], A, jnp.asarray(frac), ICE, p, hours, True)
            return o.runoff, o.water_balance_error

        runoff, error = map(np.asarray, jax.jit(finer)(jnp.asarray(fine)))
        q = runoff.reshape(-1, repeats).sum(axis=1) * area.sum() / 86.4
        sens.append(
            dict(
                kind="timestep",
                hours=hours,
                max_ledger_mm=float(np.max(np.abs(error))),
                qualification="Uniform disaggregation of daily P/PET and constant daily temperature; numerical sensitivity, not hourly forcing validation",
                metrics={
                    k: metric(q[(dates >= a) & (dates <= b)], obs[(dates >= a) & (dates <= b)])
                    for k, (a, b) in PERIODS.items()
                },
            )
        )
    (args.out / "results.json").write_text(
        json.dumps(
            dict(
                selected=selected,
                candidates=evaluated,
                sensitivities=sens,
                protocol=protocol,
                input_hashes={
                    str(getattr(args, k)): hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
                    for k in ["forcing", "observations", "balances", "snow"]
                },
            ),
            indent=2,
        )
    )
    print("FINAL REPLAY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
