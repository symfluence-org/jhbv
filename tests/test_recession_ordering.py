"""Tests for the k0 > k1 > k2 recession-ordering constraint.

The constraint used to be applied only on the numpy simulation path
(HBVWorker._run_simulation). The JAX loss that ADAM and L-BFGS
differentiate skipped it, so a gradient run could converge on an
unordered parameter set, report the score of the unconstrained model,
and then have the final evaluation silently re-sort the coefficients and
score a different model — a ~0.35 KGE gap between the two numbers.

These tests pin both halves: the helper orders correctly and stays
differentiable, and the two simulation paths agree on a set that
violates the ordering.
"""

import numpy as np
import pytest

from jhbv.parameters import DEFAULT_PARAMS, enforce_recession_ordering

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# k1 < k2, so the ordering is violated and sorting must change the set.
UNORDERED = {"k0": 0.0855, "k1": 0.0100, "k2": 0.0327}


def test_orders_descending():
    out = enforce_recession_ordering(dict(UNORDERED))
    assert out["k0"] == pytest.approx(0.0855)
    assert out["k1"] == pytest.approx(0.0327)
    assert out["k2"] == pytest.approx(0.0100)


def test_jax_and_numpy_backends_agree():
    npy = enforce_recession_ordering(dict(UNORDERED))
    jx = enforce_recession_ordering(dict(UNORDERED), use_jax=True)
    for key in ("k0", "k1", "k2"):
        assert float(jx[key]) == pytest.approx(float(npy[key]))


def test_already_ordered_set_is_unchanged():
    ordered = {"k0": 0.3, "k1": 0.1, "k2": 0.01}
    out = enforce_recession_ordering(dict(ordered))
    for key, value in ordered.items():
        assert float(out[key]) == pytest.approx(value)


def test_other_parameters_pass_through():
    out = enforce_recession_ordering({**UNORDERED, "fc": 250.0, "beta": 2.5})
    assert out["fc"] == 250.0
    assert out["beta"] == 2.5


def test_incomplete_parameter_set_is_returned_unchanged():
    partial = {"k0": 0.3, "fc": 250.0}
    assert enforce_recession_ordering(partial) is partial


def test_gradient_flows_through_the_sort():
    """Sorting is a permutation, so each raw input keeps a gradient.

    Gradients must reach the slot the value lands in, not the slot it was
    passed in as — that is what lets a gradient optimizer keep descending
    on the constrained objective.
    """

    def f(v):
        d = enforce_recession_ordering(
            {"k0": v[0], "k1": v[1], "k2": v[2]}, use_jax=True
        )
        return d["k0"] * 1.0 + d["k1"] * 10.0 + d["k2"] * 100.0

    grad = np.asarray(jax.grad(f)(jnp.array([0.0855, 0.0100, 0.0327])))
    # k0 stays in slot k0; raw k1 (smallest) lands in k2; raw k2 lands in k1.
    assert grad == pytest.approx([1.0, 100.0, 10.0])


def test_simulation_paths_agree_on_an_unordered_set():
    """The differentiable loss and the run path must score the same model."""
    from jhbv.model import create_initial_state, simulate, simulate_jax
    from jhbv.parameters import create_params_from_dict

    rng = np.random.default_rng(0)
    n = 800
    precip = rng.gamma(0.6, 4.0, n)
    temp = 10.0 + 12.0 * np.sin(np.arange(n) * 2 * np.pi / 365.0)
    pet = np.clip(2.5 + 2.0 * np.sin(np.arange(n) * 2 * np.pi / 365.0), 0, None)

    params = {**DEFAULT_PARAMS, **UNORDERED}
    params.pop("smoothing_enabled", None)

    # Run path: HBVWorker._run_simulation -> jhbv.model.simulate.
    run_flow, _ = simulate(
        precip, temp, pet, params=params,
        initial_state=create_initial_state(use_jax=True, timestep_hours=24),
        warmup_days=100, use_jax=True, timestep_hours=24,
    )

    # Loss path: _build_loss_fn builds parameters and calls simulate_jax
    # directly. Neither call sorts the coefficients itself — both inherit
    # the constraint from create_params_from_dict.
    loss_flow, _ = simulate_jax(
        jnp.array(precip), jnp.array(temp), jnp.array(pet),
        create_params_from_dict(dict(params), use_jax=True),
        warmup_days=100, timestep_hours=24,
    )

    assert np.allclose(np.asarray(run_flow), np.asarray(loss_flow), rtol=1e-6, atol=1e-8)
