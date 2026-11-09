"""Variance-guided measurement construction for Pauli Hamiltonians."""

from .solver import solve
from .expansion import expand
from .api import design_measurements
from .config import ExpansionConfig, SamplingConfig, SolverConfig
from .covariance import CallableCovariance, Covariance, CovarianceProvider
from .measurement import MeasurementDesign, PauliHamiltonian, ShotPlan
from .state import StateCovariance

__version__ = "1.0.0"
__all__ = [
    "CallableCovariance",
    "Covariance",
    "CovarianceProvider",
    "ExpansionConfig",
    "MeasurementDesign",
    "PauliHamiltonian",
    "SamplingConfig",
    "ShotPlan",
    "SolverConfig",
    "StateCovariance",
    "design_measurements",
    "expand",
    "solve",
]
