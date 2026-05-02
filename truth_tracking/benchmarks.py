"""
benchmarks.py

Runs many simulations and aggregates results into benchmark statistics.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from .run_simulation import RunConfig, simulate_run, RunResult


@dataclass
class BenchmarkResult:
    df: pd.DataFrame
    mean_absorption_time: Optional[float]
    mean_first_hit_time: Optional[float]
    mean_tail_frequency: Optional[float]
    mean_entrenchment_degree: Optional[float]
    mean_max_nontrue_formula_frequency: Optional[float]
    mean_true_world_formula_frequency: Optional[float]
    frequency_curve: Optional[np.ndarray]
    tail_frequency_curve: Optional[np.ndarray]


def run_benchmark(
    *,
    n: int,
    x: int,
    p: float,
    q: float | None,
    q_mode: str,
    use_marked_worlds: bool,
    nb_marked_worlds: int | None,
    num_runs: int,
    max_steps: int,
    window_size: int,
    seed: int | None,
    frequency_window: int,
    stop_on_absorption: bool,
    progress: bool,
    init_mode: str,
    valuemaxall: int,
    valuemin_star: int,
    valuemax_star: int,
) -> BenchmarkResult:
    """Simulate many runs and compute benchmark summaries."""
    results: List[RunResult] = []

    if seed is None:
        seeds = [None] * num_runs
    else:
        seeds = [seed + k for k in range(num_runs)]

    for k in range(num_runs):
        cfg = RunConfig(
            n=n,
            x=x,
            p=p,
            q=q,
            q_mode=q_mode,
            use_marked_worlds=use_marked_worlds,
            nb_marked_worlds=nb_marked_worlds,
            max_steps=max_steps,
            window_size=window_size,
            seed=seeds[k],
            stop_on_absorption=stop_on_absorption,
            init_mode=init_mode,
            valuemaxall=valuemaxall,
            valuemin_star=valuemin_star,
            valuemax_star=valuemax_star,
        )
        result = simulate_run(cfg)
        results.append(result)
        if progress:
            steps_simulated = len(result.hits)
            window = min(frequency_window, steps_simulated) if steps_simulated > 0 else 0
            tail_freq = float(np.mean(result.hits[-window:])) if window > 0 else None
            print(
                f"run {k + 1}/{num_runs} | steps={steps_simulated} | "
                f"first_hit={result.first_hit_time} | absorption={result.absorption_time} | "
                f"tail_freq={tail_freq} | entrenchment={result.entrenchment_degree} | "
                f"max_nontrue_freq={result.max_nontrue_formula_frequency} | "
                f"true_world_freq={result.true_world_formula_frequency}"
            )

    data = []
    tail_freqs = []
    for i, r in enumerate(results):
        steps_simulated = len(r.hits)
        window = min(frequency_window, steps_simulated) if steps_simulated > 0 else 0
        tail_freq = float(np.mean(r.hits[-window:])) if window > 0 else None
        tail_freqs.append(tail_freq)

        data.append(
            {
                "run": i,
                "true_world": r.true_world,
                "absorption_time": r.absorption_time,
                "first_hit_time": r.first_hit_time,
                "entrenchment_degree": r.entrenchment_degree,
                "max_nontrue_formula_frequency": r.max_nontrue_formula_frequency,
                "true_world_formula_frequency": r.true_world_formula_frequency,
                "steps_simulated": steps_simulated,
                "tail_frequency": tail_freq,
            }
        )

    df = pd.DataFrame(data)

    absorption_times = df["absorption_time"].dropna()
    first_hit_times = df["first_hit_time"].dropna()
    tail_freq_values = df["tail_frequency"].dropna()
    entrenchment_values = df["entrenchment_degree"].dropna()
    max_formula_values = df["max_nontrue_formula_frequency"].dropna()
    true_world_formula_values = df["true_world_formula_frequency"].dropna()

    mean_absorption_time = float(absorption_times.mean()) if len(absorption_times) > 0 else None
    mean_first_hit_time = float(first_hit_times.mean()) if len(first_hit_times) > 0 else None
    mean_tail_frequency = float(tail_freq_values.mean()) if len(tail_freq_values) > 0 else None
    mean_entrenchment_degree = (
        float(entrenchment_values.mean()) if len(entrenchment_values) > 0 else None
    )
    mean_max_nontrue_formula_frequency = (
        float(max_formula_values.mean()) if len(max_formula_values) > 0 else None
    )
    mean_true_world_formula_frequency = (
        float(true_world_formula_values.mean()) if len(true_world_formula_values) > 0 else None
    )

    frequency_curve = convergence_frequency_curve(results, max_steps) if results else None
    tail_frequency_curve = (
        tail_frequency_curve_over_time(results, max_steps, frequency_window) if results else None
    )

    return BenchmarkResult(
        df=df,
        mean_absorption_time=mean_absorption_time,
        mean_first_hit_time=mean_first_hit_time,
        mean_tail_frequency=mean_tail_frequency,
        mean_entrenchment_degree=mean_entrenchment_degree,
        mean_max_nontrue_formula_frequency=mean_max_nontrue_formula_frequency,
        mean_true_world_formula_frequency=mean_true_world_formula_frequency,
        frequency_curve=frequency_curve,
        tail_frequency_curve=tail_frequency_curve,
    )


def convergence_frequency_curve(results: List[RunResult], max_steps: int) -> np.ndarray:
    """
    Compute average frequency curve:
    freq[t] = avg over runs of ( fraction of steps <= t where Bel=={w*} ).
    """
    curves = []
    for r in results:
        hits = np.array(r.hits, dtype=np.float64)
        if len(hits) < max_steps:
            hits = np.pad(hits, (0, max_steps - len(hits)), constant_values=0.0)
        else:
            hits = hits[:max_steps]

        cumsum = np.cumsum(hits)
        curve = cumsum / (np.arange(max_steps) + 1)
        curves.append(curve)

    return np.mean(np.stack(curves), axis=0)


def tail_frequency_curve_over_time(
    results: List[RunResult], max_steps: int, window: int
) -> np.ndarray:
    """
    Compute average tail-frequency curve:
    tail[t] = avg over runs of ( fraction of hits in the last 'window' steps ending at t ).
    """
    curves = []
    for r in results:
        hits = np.array(r.hits, dtype=np.float64)
        if len(hits) < max_steps:
            hits = np.pad(hits, (0, max_steps - len(hits)), constant_values=0.0)
        else:
            hits = hits[:max_steps]

        tail = np.empty(max_steps, dtype=np.float64)
        cumsum = np.cumsum(hits)
        for t in range(max_steps):
            start = max(0, t - window + 1)
            window_sum = cumsum[t] - (cumsum[start - 1] if start > 0 else 0.0)
            tail[t] = window_sum / (t - start + 1)
        curves.append(tail)

    return np.mean(np.stack(curves), axis=0)
