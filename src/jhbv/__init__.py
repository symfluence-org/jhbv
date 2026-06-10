# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HBV-96 Model -- Standalone Plugin Package.

A native JAX-based implementation of the HBV-96 hydrological model, enabling:
- Automatic differentiation for gradient-based calibration
- JIT compilation for fast execution
- Vectorization (vmap) for ensemble runs
- GPU acceleration when available
- Distributed modeling with graph-based Muskingum-Cunge routing

Components:
    - HBVPreProcessor: Prepares forcing data (P, T, PET)
    - HBVRunner: Executes model simulations
    - HBVPostProcessor: Extracts streamflow results
    - HBVWorker: Handles calibration with gradient support
    - DistributedHBV: Semi-distributed HBV with river network routing

Usage:
    # Standard workflow
    from jhbv import HBVPreProcessor, HBVRunner, HBVPostProcessor

    preprocessor = HBVPreProcessor(config, logger)
    preprocessor.run_preprocessing()

    runner = HBVRunner(config, logger)
    output_path = runner.run_hbv()

    # Distributed HBV with routing
    from jhbv import DistributedHBV, create_synthetic_network

    network = create_synthetic_network(n_nodes=5, topology='fishbone')
    model = DistributedHBV(network)
    outlet_flow, state = model.simulate(precip, temp, pet)

    # Gradient-based calibration
    grad_fn = model.get_gradient_function(precip, temp, pet, obs)
    gradients = grad_fn(params_array)

References:
    Lindstrom, G., Johansson, B., Persson, M., Gardelin, M., & Bergstrom, S. (1997).
    Development and test of the distributed HBV-96 hydrological model.
    Journal of Hydrology, 201(1-4), 272-288.
"""

import warnings
from typing import TYPE_CHECKING

# Flag to track if the experimental warning has been shown
_warning_shown = False


def _show_experimental_warning():
    """Show the experimental warning once when HBV components are first accessed."""
    global _warning_shown
    if not _warning_shown:
        warnings.warn(
            "HBV is an EXPERIMENTAL module. The API may change without notice. "
            "For production use, consider using SUMMA or FUSE instead.",
            category=UserWarning,
            stacklevel=4
        )
        _warning_shown = True


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # Configuration
    'HBVConfig': ('.config', 'HBVConfig'),
    'HBVConfigAdapter': ('.config', 'HBVConfigAdapter'),

    # Main components
    'HBVPreProcessor': ('.preprocessor', 'HBVPreProcessor'),
    'HBVRunner': ('.runner', 'HBVRunner'),
    'HBVPostProcessor': ('.postprocessor', 'HBVPostProcessor'),
    'HBVRoutedPostProcessor': ('.postprocessor', 'HBVRoutedPostProcessor'),
    'HBVResultExtractor': ('.extractor', 'HBVResultExtractor'),

    # Parameters (from parameters module)
    'PARAM_BOUNDS': ('.parameters', 'PARAM_BOUNDS'),
    'DEFAULT_PARAMS': ('.parameters', 'DEFAULT_PARAMS'),
    'RATE_PARAMS': ('.parameters', 'RATE_PARAMS'),
    'DURATION_PARAMS': ('.parameters', 'DURATION_PARAMS'),
    'HBVParameters': ('.parameters', 'HBVParameters'),
    'create_params_from_dict': ('.parameters', 'create_params_from_dict'),
    'scale_params_for_timestep': ('.parameters', 'scale_params_for_timestep'),
    'get_routing_buffer_length': ('.parameters', 'get_routing_buffer_length'),

    # Loss functions (from losses module)
    'nse_loss': ('.losses', 'nse_loss'),
    'kge_loss': ('.losses', 'kge_loss'),
    'get_nse_gradient_fn': ('.losses', 'get_nse_gradient_fn'),
    'get_kge_gradient_fn': ('.losses', 'get_kge_gradient_fn'),

    # Core model
    'simulate': ('.model', 'simulate'),
    'simulate_jax': ('.model', 'simulate_jax'),
    'simulate_numpy': ('.model', 'simulate_numpy'),
    'simulate_ensemble': ('.model', 'simulate_ensemble'),
    'HBVState': ('.model', 'HBVState'),
    'create_initial_state': ('.model', 'create_initial_state'),
    'step_jax': ('.model', 'step_jax'),
    'snow_routine_jax': ('.model', 'snow_routine_jax'),
    'soil_routine_jax': ('.model', 'soil_routine_jax'),
    'response_routine_jax': ('.model', 'response_routine_jax'),
    'routing_routine_jax': ('.model', 'routing_routine_jax'),
    'triangular_weights': ('.model', 'triangular_weights'),
    'jit_simulate': ('.model', 'jit_simulate'),
    'HAS_JAX': ('.model', 'HAS_JAX'),

    # Calibration
    'HBVWorker': ('.calibration.worker', 'HBVWorker'),
    'HBVParameterManager': ('.calibration.parameter_manager', 'HBVParameterManager'),
    'get_hbv_calibration_bounds': ('.calibration.parameter_manager', 'get_hbv_calibration_bounds'),

    # Distributed HBV with routing
    'DistributedHBV': ('.distributed', 'DistributedHBV'),
    'DistributedHBVState': ('.distributed', 'DistributedHBVState'),
    'DistributedHBVParams': ('.distributed', 'DistributedHBVParams'),
    'calibrate_distributed_hbv': ('.distributed', 'calibrate_distributed_hbv'),
    'calibrate_distributed_hbv_adam': ('.distributed', 'calibrate_distributed_hbv_adam'),
    'load_distributed_hbv_from_config': ('.distributed', 'load_distributed_hbv_from_config'),
    'RiverNetwork': ('.network', 'RiverNetwork'),
    'NetworkBuilder': ('.network', 'NetworkBuilder'),
    'create_synthetic_network': ('.network', 'create_synthetic_network'),
    'RoutingParams': ('.routing', 'RoutingParams'),
    'RoutingState': ('.routing', 'RoutingState'),
    'compute_muskingum_params': ('.routing', 'compute_muskingum_params'),
    'route_reach_step': ('.routing', 'route_reach_step'),
    'runoff_mm_to_cms': ('.routing', 'runoff_mm_to_cms'),

    # Regionalization
    'forward_transfer_function': ('.regionalization', 'forward_transfer_function'),
    'initialize_weights': ('.regionalization', 'initialize_weights'),
    'TransferFunctionConfig': ('.regionalization', 'TransferFunctionConfig'),
    'TransferLayer': ('.regionalization', 'TransferLayer'),

    # Optimizers
    'AdamW': ('.optimizers', 'AdamW'),
    'CosineAnnealingWarmRestarts': ('.optimizers', 'CosineAnnealingWarmRestarts'),
    'CosineDecay': ('.optimizers', 'CosineDecay'),
    'EMA': ('.optimizers', 'EMA'),
    'CalibrationResult': ('.optimizers', 'CalibrationResult'),
    'EXTENDED_PARAM_BOUNDS': ('.optimizers', 'EXTENDED_PARAM_BOUNDS'),

    # ODE-based implementation (diffrax with adjoint gradients)
    'HAS_DIFFRAX': ('.hbv_ode', 'HAS_DIFFRAX'),
    'HBVODEState': ('.hbv_ode', 'HBVODEState'),
    'AdjointMethod': ('.hbv_ode', 'AdjointMethod'),
    'hbv_dynamics': ('.hbv_ode', 'hbv_dynamics'),
    'simulate_ode': ('.hbv_ode', 'simulate_ode'),
    'simulate_ode_with_routing': ('.hbv_ode', 'simulate_ode_with_routing'),
    'nse_loss_ode': ('.hbv_ode', 'nse_loss_ode'),
    'get_nse_gradient_fn_ode': ('.hbv_ode', 'get_nse_gradient_fn_ode'),
    'compare_gradients': ('.hbv_ode', 'compare_gradients'),
    'create_forcing_interpolant': ('.hbv_ode', 'create_forcing_interpolant'),
}


def __getattr__(name: str):
    """Lazy import handler for HBV module components.

    This allows importing from the hbv module without loading all submodules
    until they are actually accessed.
    """
    if name in _LAZY_IMPORTS:
        _show_experimental_warning()
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return available attributes for tab completion."""
    return list(_LAZY_IMPORTS.keys()) + ['register']


def register() -> None:
    """Register HBV components with symfluence plugin registry."""
    from symfluence.core.registry import model_manifest
    from .calibration.optimizer import HBVModelOptimizer
    from .calibration.parameter_manager import HBVParameterManager
    from .calibration.worker import HBVWorker
    from .config import HBVConfigAdapter
    from .extractor import HBVResultExtractor
    from .postprocessor import HBVPostProcessor, HBVRoutedPostProcessor
    from .preprocessor import HBVPreProcessor
    from .runner import HBVRunner

    model_manifest(
        "HBV",
        preprocessor=HBVPreProcessor,
        runner=HBVRunner,
        runner_method='run_hbv',
        postprocessor=HBVPostProcessor,
        config_adapter=HBVConfigAdapter,
        result_extractor=HBVResultExtractor,
        optimizer=HBVModelOptimizer,
        worker=HBVWorker,
        parameter_manager=HBVParameterManager,
    )
    model_manifest(
        "HBV_routed",
        postprocessor=HBVRoutedPostProcessor,
    )


# Type hints for IDE support
if TYPE_CHECKING:
    from .calibration.parameter_manager import HBVParameterManager, get_hbv_calibration_bounds
    from .calibration.worker import HBVWorker
    from .config import HBVConfig, HBVConfigAdapter
    from .distributed import (
        DistributedHBV,
        DistributedHBVParams,
        DistributedHBVState,
        calibrate_distributed_hbv,
        calibrate_distributed_hbv_adam,
        load_distributed_hbv_from_config,
    )
    from .extractor import HBVResultExtractor
    from .hbv_ode import (
        HAS_DIFFRAX,
        AdjointMethod,
        HBVODEState,
        compare_gradients,
        create_forcing_interpolant,
        get_nse_gradient_fn_ode,
        hbv_dynamics,
        nse_loss_ode,
        simulate_ode,
        simulate_ode_with_routing,
    )
    from .losses import (
        get_kge_gradient_fn,
        get_nse_gradient_fn,
        kge_loss,
        nse_loss,
    )
    from .model import (
        HAS_JAX,
        HBVState,
        create_initial_state,
        jit_simulate,
        response_routine_jax,
        routing_routine_jax,
        simulate,
        simulate_ensemble,
        simulate_jax,
        simulate_numpy,
        snow_routine_jax,
        soil_routine_jax,
        step_jax,
        triangular_weights,
    )
    from .network import NetworkBuilder, RiverNetwork, create_synthetic_network
    from .optimizers import (
        EMA,
        EXTENDED_PARAM_BOUNDS,
        AdamW,
        CalibrationResult,
        CosineAnnealingWarmRestarts,
        CosineDecay,
    )
    from .parameters import (
        DEFAULT_PARAMS,
        DURATION_PARAMS,
        PARAM_BOUNDS,
        RATE_PARAMS,
        HBVParameters,
        create_params_from_dict,
        get_routing_buffer_length,
        scale_params_for_timestep,
    )
    from .postprocessor import HBVPostProcessor, HBVRoutedPostProcessor
    from .preprocessor import HBVPreProcessor
    from .regionalization import (
        TransferFunctionConfig,
        TransferLayer,
        forward_transfer_function,
        initialize_weights,
    )
    from .routing import (
        RoutingParams,
        RoutingState,
        compute_muskingum_params,
        route_reach_step,
        runoff_mm_to_cms,
    )
    from .runner import HBVRunner


__all__ = [
    # Plugin registration
    'register',

    # Main components
    'HBVPreProcessor',
    'HBVRunner',
    'HBVPostProcessor',
    'HBVRoutedPostProcessor',
    'HBVResultExtractor',

    # Configuration
    'HBVConfig',
    'HBVConfigAdapter',

    # Parameters (from parameters module)
    'PARAM_BOUNDS',
    'DEFAULT_PARAMS',
    'RATE_PARAMS',
    'DURATION_PARAMS',
    'HBVParameters',
    'create_params_from_dict',
    'scale_params_for_timestep',
    'get_routing_buffer_length',

    # Loss functions (from losses module)
    'nse_loss',
    'kge_loss',
    'get_nse_gradient_fn',
    'get_kge_gradient_fn',

    # Core model
    'simulate',
    'simulate_jax',
    'simulate_numpy',
    'simulate_ensemble',
    'HBVState',
    'create_initial_state',
    'step_jax',
    'snow_routine_jax',
    'soil_routine_jax',
    'response_routine_jax',
    'routing_routine_jax',
    'triangular_weights',
    'jit_simulate',
    'HAS_JAX',

    # Calibration
    'HBVWorker',
    'HBVParameterManager',
    'get_hbv_calibration_bounds',

    # Distributed HBV with routing
    'DistributedHBV',
    'DistributedHBVState',
    'DistributedHBVParams',
    'calibrate_distributed_hbv',
    'calibrate_distributed_hbv_adam',
    'load_distributed_hbv_from_config',
    'RiverNetwork',
    'NetworkBuilder',
    'create_synthetic_network',
    'RoutingParams',
    'RoutingState',
    'compute_muskingum_params',
    'route_reach_step',
    'runoff_mm_to_cms',

    # Regionalization
    'forward_transfer_function',
    'initialize_weights',
    'TransferFunctionConfig',
    'TransferLayer',

    # Optimizers
    'AdamW',
    'CosineAnnealingWarmRestarts',
    'CosineDecay',
    'EMA',
    'CalibrationResult',
    'EXTENDED_PARAM_BOUNDS',

    # ODE-based implementation (diffrax with adjoint gradients)
    'HAS_DIFFRAX',
    'HBVODEState',
    'AdjointMethod',
    'hbv_dynamics',
    'simulate_ode',
    'simulate_ode_with_routing',
    'nse_loss_ode',
    'get_nse_gradient_fn_ode',
    'compare_gradients',
    'create_forcing_interpolant',
]
