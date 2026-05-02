"""
run_simulation.py

Defines how to run a single run.
In practice, we simulate up to max_steps, or stop early when a benchmark condition is satisfied.
"""

from dataclasses import dataclass
from typing import Optional, List, Set
import numpy as np

from .ranking import RankingFunction
from .formula_generation import FormulaGenerator


@dataclass
class RunConfig:
    n: int
    x: int
    p: float
    q: float | None
    q_mode: str
    max_steps: int
    seed: int | None
    window_size: int = 100
    true_world: Optional[int] = None
    max_retries: int = 10000
    stop_on_absorption: bool = False
    capture_trace: bool = False
    init_mode: str = "flat"
    valuemaxall: int = 0
    valuemin_star: int = 0
    valuemax_star: int = 0
    use_marked_worlds: bool = False
    nb_marked_worlds: int | None = None


@dataclass
class RunResult:
    true_world: int
    beliefs: List[Set[int]]  # Bel(rf_t)
    hits: List[bool]  # Bel(rf_t) == {w*}
    absorption_time: Optional[int]  # first t such that window_size consecutive beliefs are {w*}
    first_hit_time: Optional[int]  # first t such that belief is {w*}
    entrenchment_degree: int  # rf_last(w') - rf_last(w*)
    max_nontrue_formula_frequency: float | None  # max proportion of formulas containing any non-true world
    true_world_formula_frequency: float | None  # proportion of formulas containing the true world
    formulas: Optional[List[List[int]]] = None
    ranks: Optional[List[List[int]]] = None
    initial_ranks: Optional[List[int]] = None


def simulate_run(config: RunConfig) -> RunResult:
    """Simulate one run under the given config."""
    rng = np.random.default_rng(config.seed)
    num_worlds = 2 ** config.n

    if config.true_world is None:
        true_world = int(rng.integers(0, num_worlds))
    else:
        if not (0 <= config.true_world < num_worlds):
            raise ValueError("true_world must be in [0, 2^n - 1].")
        true_world = config.true_world

    if config.init_mode not in {"flat", "random"}:
        raise ValueError("init_mode must be 'flat' or 'random'.")
    if config.valuemaxall < 0:
        raise ValueError("valuemaxall must be >= 0.")
    if not (0 <= config.valuemin_star <= config.valuemax_star <= config.valuemaxall):
        raise ValueError("Require 0 <= valuemin_star <= valuemax_star <= valuemaxall.")

    if config.init_mode == "flat":
        rf = RankingFunction.flat(num_worlds)
    else:
        ranks = rng.integers(0, config.valuemaxall + 1, size=num_worlds, dtype=np.int64)
        # Normalize other worlds first to ensure at least one world has rank 0.
        ranks -= int(ranks.min())
        desired_true = int(rng.integers(config.valuemin_star, config.valuemax_star + 1))
        if desired_true > 0:
            other_zeros = np.flatnonzero((ranks == 0) & (np.arange(num_worlds) != true_world))
            if len(other_zeros) == 0:
                # Force some non-true world to 0 to keep minimum at 0.
                other_indices = np.arange(num_worlds) != true_world
                min_other = ranks[other_indices].min()
                ranks[other_indices] -= int(min_other)
        ranks[true_world] = desired_true
        rf = RankingFunction(ranks=ranks)

    marked_worlds: np.ndarray | None = None
    if config.use_marked_worlds:
        if config.nb_marked_worlds is None:
            raise ValueError("nb_marked_worlds must be set when use_marked_worlds is True.")
        max_marked = num_worlds - 1
        if not (1 <= config.nb_marked_worlds <= max_marked):
            raise ValueError("nb_marked_worlds must be in [1, 2^n - 1].")
        if config.q_mode != "random":
            if config.q is None or not (0.0 < config.q < config.p):
                raise ValueError("Require 0 < q < p when using marked worlds.")
        candidates = np.array([w for w in range(num_worlds) if w != true_world], dtype=np.int64)
        marked_worlds = rng.choice(
            candidates, size=config.nb_marked_worlds, replace=False
        ).astype(np.int64)

    q_value = config.q
    q_mode = config.q_mode
    if config.q_mode == "random":
        q_value = float(rng.uniform(low=np.nextafter(0.0, 1.0), high=config.p))
        q_mode = "fixed"

    gen = FormulaGenerator(
        p=config.p,
        q=q_value,
        true_world=true_world,
        num_worlds=num_worlds,
        q_mode=q_mode,
        max_retries=config.max_retries,
        marked_worlds=marked_worlds,
    )

    beliefs: List[Set[int]] = []
    hits: List[bool] = []
    formulas: List[List[int]] | None = [] if config.capture_trace else None
    ranks: List[List[int]] | None = [] if config.capture_trace else None
    initial_ranks = rf.ranks.tolist() if config.capture_trace else None
    absorption_time = None
    first_hit_time = None
    formula_counts = np.zeros(num_worlds, dtype=np.int64)

    consecutive_singleton = 0

    for t in range(config.max_steps):
        # Generate formula
        F = gen.generate(rng)
        formula_counts[F] += 1

        # Apply improvement
        rf.improve(F, x=config.x)

        bel = rf.belief()
        beliefs.append(bel)

        is_hit = bel == {true_world}
        hits.append(is_hit)

        if config.capture_trace:
            assert formulas is not None
            assert ranks is not None
            formulas.append([int(w) for w in F])
            ranks.append([int(v) for v in rf.ranks.tolist()])

        # first hit time
        if first_hit_time is None and is_hit:
            first_hit_time = t

        # absorption time (window_size consecutive)
        if is_hit:
            consecutive_singleton += 1
            if consecutive_singleton >= config.window_size and absorption_time is None:
                absorption_time = t - config.window_size + 1
                if config.stop_on_absorption:
                    break
        else:
            consecutive_singleton = 0

    true_rank = int(rf.ranks[true_world])
    non_true_mask = np.arange(num_worlds) != true_world
    min_non_true = int(np.min(rf.ranks[non_true_mask]))
    entrenchment_degree = min_non_true - true_rank
    steps_simulated = len(beliefs)
    if steps_simulated == 0 or not np.any(non_true_mask):
        max_nontrue_formula_frequency = None
    else:
        max_nontrue_formula_frequency = float(
            np.max(formula_counts[non_true_mask]) / steps_simulated
        )
    true_world_formula_frequency = (
        float(formula_counts[true_world] / steps_simulated) if steps_simulated > 0 else None
    )

    return RunResult(
        true_world=true_world,
        beliefs=beliefs,
        hits=hits,
        absorption_time=absorption_time,
        first_hit_time=first_hit_time,
        entrenchment_degree=entrenchment_degree,
        max_nontrue_formula_frequency=max_nontrue_formula_frequency,
        true_world_formula_frequency=true_world_formula_frequency,
        formulas=formulas,
        ranks=ranks,
        initial_ranks=initial_ranks,
    )
