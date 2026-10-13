"""Enabling deep-ice audit precision must not break existing network routing."""

import numpy as np
import pytest


@pytest.mark.parametrize("x64", [False, True])
def test_existing_distributed_routing_precision(x64):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jhbv.distributed import DistributedHBV
    from jhbv.network import create_synthetic_network

    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", x64)
        network = create_synthetic_network(n_nodes=3, topology="linear", use_jax=True)
        model = DistributedHBV(network, warmup_days=0, use_jax=True)
        forcing = jnp.ones((10, 3))
        q, _ = model.simulate(forcing * 2, forcing * 10, forcing)
        assert np.isfinite(q).all() and (np.asarray(q) >= 0).all()
    finally:
        jax.config.update("jax_enable_x64", previous)
