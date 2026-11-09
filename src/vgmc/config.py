"""Immutable numerical settings and validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from collections.abc import Mapping
from typing import Any, Self

from ._validation import integer, positive


@dataclass(frozen=True)
class _Config:
    """Parse explicitly named dataclass fields at the JSON boundary."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Validate known fields and construct effective configuration.

        Args:
            values: Explicit overrides for dataclass defaults.

        Returns:
            Immutable configuration with validated scalar values.

        Raises:
            ValueError: A field name or value is outside the schema.
        """
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown {cls.__name__} fields: {sorted(unknown)}"
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        """Serialize effective dataclass settings for a run manifest."""
        return asdict(self)


@dataclass(frozen=True)
class SolverConfig(_Config):
    """Configure fixed-library coefficient and allocation optimization.

    Args:
        ridge: Diagonal design regularization in Pauli covariance units.
        relative_gap: Required numerical primal/dual relative gap.
        allocation_floor: Positive numerical floor before normalization.
        dual_safety_margin: Relative inflation of the dual score denominator.
        linear_refinements: Residual correction steps in precision solves.
        coefficient_refinements: Residual correction steps for local blocks.
        conservation_tolerance: Relative coefficient-recovery tolerance,
            scaled by ``max(1, max(abs(c)))``.
        checkpoint_interval: Accepted optimizer updates between callbacks.
        max_updates: Accepted allocation update limit, or ``None``.
        max_evaluations: Objective evaluation limit, or ``None``.
        max_backtracks: Boundary line-search step reductions.
        stall_checks: Consecutive stages without gap improvement.
        threads: BLAS thread count during one solve.

    Examples:
        >>> settings = SolverConfig(relative_gap=1e-5, max_updates=500)
        >>> settings.with_gap(1e-4).relative_gap
        0.0001
    """

    ridge: float = 1e-8
    relative_gap: float = 1e-4
    allocation_floor: float = 1e-15
    dual_safety_margin: float = 1e-8
    linear_refinements: int = 3
    coefficient_refinements: int = 2
    conservation_tolerance: float = 1e-9
    checkpoint_interval: int = 50
    max_updates: int | None = 10000
    max_evaluations: int | None = 100000
    max_backtracks: int = 30
    stall_checks: int = 3
    threads: int = 1

    def __post_init__(self) -> None:
        """Validate numerical ranges and integer control fields."""
        positive(self.ridge, "ridge", zero=True)
        positive(self.dual_safety_margin, "dual_safety_margin", zero=True)
        for name in (
            "relative_gap",
            "allocation_floor",
            "conservation_tolerance",
        ):
            if positive(getattr(self, name), name) >= 1:
                raise ValueError(f"{name} must be smaller than one")
        for name in ("linear_refinements", "coefficient_refinements"):
            integer(getattr(self, name), name)
        for name in ("checkpoint_interval", "stall_checks", "threads"):
            integer(getattr(self, name), name, 1)
        integer(self.max_backtracks, "max_backtracks", 1)
        for name in ("max_updates", "max_evaluations"):
            value = getattr(self, name)
            if value is not None:
                integer(value, name, 0 if name == "max_updates" else 1)

    def with_gap(self, gap: float) -> SolverConfig:
        """Copy the configuration with an explicit relative gap target."""
        return replace(self, relative_gap=gap)


@dataclass(frozen=True)
class ExpansionConfig(_Config):
    """Configure deterministic VGMC candidate construction and admission.

    Args:
        solver: Fixed-library numerical settings.
        inner_gap: Gap target immediately after admitting groups.
        terminal_gap: Gap target at each recorded endpoint.
        boundary_gaps: Decreasing targets used to refine unresolved admission.
        pool_gain_over_initial: Finite-pool remaining gain tolerance.
        active_threshold: Allocation threshold used for counts.
        new_mass: Allocation mass assigned to each admitted batch.
        candidate_score_margin: Relative candidate admission score margin.
        dual_safety_margin: Relative finite-pool denominator margin.
        batch_fraction: Default batch fraction of the initial library.
        batch_minimum: Default minimum batch size.
        batch_maximum: Default maximum batch size.
        linear_refinements: Score residual corrections.
        eigen_error_factor: Eigenvalue roundoff scale for score checks.
        roundoff_factor: Score arithmetic allowance scale.
    """

    solver: SolverConfig = SolverConfig()
    inner_gap: float = 1e-3
    terminal_gap: float = 1e-4
    boundary_gaps: tuple[float, ...] = (1e-5, 1e-6)
    pool_gain_over_initial: float = 1e-4
    active_threshold: float = 1e-12
    new_mass: float = 1e-3
    candidate_score_margin: float = 1e-6
    dual_safety_margin: float = 1e-9
    batch_fraction: float = 0.1
    batch_minimum: int = 8
    batch_maximum: int = 256
    linear_refinements: int = 3
    eigen_error_factor: float = 8.0
    roundoff_factor: float = 64.0

    def __post_init__(self) -> None:
        """Validate batch sizes, score settings and refinement order."""
        if not isinstance(self.solver, SolverConfig):
            raise ValueError("solver must be a SolverConfig")
        for name in (
            "inner_gap",
            "terminal_gap",
            "pool_gain_over_initial",
            "new_mass",
            "batch_fraction",
        ):
            if positive(getattr(self, name), name) >= 1:
                raise ValueError(f"{name} must be smaller than one")
        if self.terminal_gap > self.inner_gap:
            raise ValueError("terminal_gap must be at most inner_gap")
        gaps = tuple(positive(g, "boundary_gap") for g in self.boundary_gaps)
        previous = self.terminal_gap
        for gap in gaps:
            if positive(gap, "boundary_gap") >= previous:
                raise ValueError("Boundary gap targets must decrease")
            previous = gap
        object.__setattr__(self, "boundary_gaps", gaps)
        for name in (
            "active_threshold",
            "candidate_score_margin",
            "dual_safety_margin",
        ):
            positive(getattr(self, name), name, zero=True)
        for name in ("eigen_error_factor", "roundoff_factor"):
            positive(getattr(self, name), name)
        integer(self.batch_minimum, "batch_minimum", 1)
        integer(self.batch_maximum, "batch_maximum", self.batch_minimum)
        integer(self.linear_refinements, "linear_refinements")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExpansionConfig:
        """Parse a canonical JSON mapping and its nested solver settings."""
        parsed = dict(values)
        if isinstance(parsed.get("solver"), Mapping):
            parsed["solver"] = SolverConfig.from_mapping(parsed["solver"])
        return super().from_mapping(parsed)

    def audit_settings(self) -> dict[str, int | float]:
        """Return the numerical fields used by covariance score checks."""
        return {
            "linear_refinements": self.linear_refinements,
            "eigen_error_factor": self.eigen_error_factor,
            "roundoff_factor": self.roundoff_factor,
        }


@dataclass(frozen=True)
class SamplingConfig(_Config):
    """Configure E2 joint-bitstring sampling and RMSE intervals.

    Args:
        budgets: Distinct increasing integer shot budgets.
        replicates: Independent energy estimates at each budget.
        replicate_chunk: Replicates generated in one multinomial call.
        checkpoint_groups: Completed groups between committed checkpoints.
        seed: Root measurement random seed.
        bootstrap_seed: Independent bootstrap random seed.
        bootstrap_replicates: Percentile bootstrap sample count.
        epsilon: Chemical accuracy threshold in Hartree.
        conservation_tolerance: Hamiltonian coefficient recovery tolerance.
    """

    budgets: tuple[int, ...] = (10000, 20000, 50000)
    replicates: int = 300
    replicate_chunk: int = 30
    checkpoint_groups: int = 10
    seed: int = 291003
    bootstrap_seed: int = 291004
    bootstrap_replicates: int = 2000
    epsilon: float = 0.0016
    conservation_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate shot, repetition, chunk and random-stream settings."""
        budgets = tuple(integer(n, "budget", 1) for n in self.budgets)
        if not budgets or tuple(sorted(set(budgets))) != budgets:
            raise ValueError("Budgets must be distinct and increasing")
        object.__setattr__(self, "budgets", budgets)
        for name in (
            "replicates",
            "replicate_chunk",
            "checkpoint_groups",
            "bootstrap_replicates",
        ):
            integer(getattr(self, name), name, 1)
        integer(self.seed, "seed")
        integer(self.bootstrap_seed, "bootstrap_seed")
        positive(self.epsilon, "epsilon")
        positive(self.conservation_tolerance, "conservation_tolerance")


@dataclass(frozen=True)
class E1Config(_Config):
    """Select the E1 dataset universe, comparison methods and design
    options.
    """

    molecules: tuple[str, ...] = ("H2", "LiH", "BeH2", "H2O", "NH3")
    covariances: tuple[str, ...] = ("exact", "cisd")
    methods: tuple[str, ...] = ("LF", "OGM", "SI", "ICS", "SOCP", "VGMC")
    ics_iterations: int = 5
    ogm_shots: int = 1000000
    expansion: ExpansionConfig = ExpansionConfig()

    def __post_init__(self) -> None:
        """Validate declared datasets, methods and baseline iteration
        counts.
        """
        for name in ("molecules", "covariances", "methods"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct entries")
            object.__setattr__(self, name, values)
        if set(self.covariances) - {"exact", "cisd"}:
            raise ValueError("Unknown covariance source")
        if set(self.methods) - {"LF", "OGM", "SI", "ICS", "SOCP", "VGMC"}:
            raise ValueError("Unknown comparison method")
        if not isinstance(self.expansion, ExpansionConfig):
            raise ValueError("expansion must be an ExpansionConfig")
        integer(self.ics_iterations, "ics_iterations", 1)
        integer(self.ogm_shots, "ogm_shots", 1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> E1Config:
        """Parse the E1 configuration and its nested expansion settings."""
        parsed = dict(values)
        if isinstance(parsed.get("expansion"), Mapping):
            parsed["expansion"] = ExpansionConfig.from_mapping(
                parsed["expansion"]
            )
        return super().from_mapping(parsed)


@dataclass(frozen=True)
class MoleculeLimits(_Config):
    """Set E3 batch size, admitted-group budget and recorded count grid."""

    batch: int
    budget: int
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the integer count grid against the group budget."""
        integer(self.batch, "batch", 1)
        integer(self.budget, "budget")
        targets = tuple(integer(k, "target") for k in self.targets)
        if (
            not targets
            or targets[0] != 0
            or tuple(sorted(set(targets))) != targets
        ):
            raise ValueError("Targets must start at zero and increase")
        if targets[-1] > self.budget:
            raise ValueError("Target count exceeds the admitted-group budget")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class E3Config(_Config):
    """Declare E3 growth arms, per-molecule limits and random references."""

    molecules: Mapping[str, MoleculeLimits]
    arms: Mapping[str, Mapping[str, str]]
    workers: int = 1
    numerics: ExpansionConfig = ExpansionConfig()
    random_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)

    def __post_init__(self) -> None:
        """Freeze nested mappings and validate the four arm definitions."""
        from types import MappingProxyType

        expected = {
            "A": dict(growth_covariance="full", growth_response="current"),
            "B": dict(growth_covariance="full", growth_response="frozen_mu0"),
            "C": dict(growth_covariance="diagonal", growth_response="current"),
            "D": dict(
                growth_covariance="diagonal", growth_response="frozen_mu0"
            ),
        }
        if dict(self.arms) != expected:
            raise ValueError(
                "E3 arm definitions differ from the implemented arms"
            )
        if not self.molecules or any(
            not isinstance(v, MoleculeLimits) for v in self.molecules.values()
        ):
            raise ValueError("E3 requires validated molecule limits")
        if not isinstance(self.numerics, ExpansionConfig):
            raise ValueError("numerics must be an ExpansionConfig")
        seeds = tuple(
            integer(seed, "random seed") for seed in self.random_seeds
        )
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Random reference seeds must be distinct")
        integer(self.workers, "workers", 1)
        object.__setattr__(
            self, "molecules", MappingProxyType(dict(self.molecules))
        )
        object.__setattr__(
            self,
            "arms",
            MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in expected.items()}
            ),
        )
        object.__setattr__(self, "random_seeds", seeds)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> E3Config:
        """Parse canonical E3 JSON and every nested numerical option."""
        parsed = dict(values)
        if isinstance(parsed.get("numerics"), Mapping):
            parsed["numerics"] = ExpansionConfig.from_mapping(
                parsed["numerics"]
            )
        if isinstance(parsed.get("molecules"), Mapping):
            parsed["molecules"] = {
                k: MoleculeLimits.from_mapping(v)
                for k, v in parsed["molecules"].items()
            }
        return super().from_mapping(parsed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize immutable nested E3 configuration without mutable
        aliases.
        """
        return dict(
            molecules={k: v.to_dict() for k, v in self.molecules.items()},
            arms={k: dict(v) for k, v in self.arms.items()},
            workers=self.workers,
            numerics=self.numerics.to_dict(),
            random_seeds=list(self.random_seeds),
        )
