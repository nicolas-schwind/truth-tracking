"""
cli.py

Command-line interface for running simulations and benchmarks.
"""

from enum import Enum
from pathlib import Path
from datetime import datetime
import typer
import numpy as np
import click
from rich.console import Console
from rich.table import Table
import pandas as pd
import matplotlib.pyplot as plt

from .benchmarks import run_benchmark
from .plotting import (
    plot_frequency_curve,
    plot_trace_ranks,
    plot_surface_from_points,
    plot_scatter_from_points,
    plot_wireframe_from_points,
    plot_contour_from_points,
    plot_heatmap_from_points,
    plot_contour2d_from_points,
    _interpolate_grid,
    _polyfit_grid,
    plot_surface_from_grid,
    plot_wireframe_from_grid,
    plot_contour_from_grid,
    plot_scatter_from_grid,
    plot_surface_overlay,
    plot_diffpq_line,
)
from .run_simulation import RunConfig, simulate_run
from .worlds import world_to_bits

# Note for future commands: include a full, user-friendly description plus
# multiple diverse usage examples in the docstring so `--help` stays consistent.
app = typer.Typer(
    add_completion=False,
    help=(
        "Truth tracking CLI.\n\n"
        "This tool simulates runs of ranking functions updated by stochastic formulas, "
        "and provides benchmarks, frequency curves, and detailed traces.\n\n"
        "Key ideas:\n"
        "- A world is an integer in [0, 2^n - 1].\n"
        "- Bel(rf) is the set of worlds with rank 0.\n"
        "- Each formula F is a non-empty set of worlds.\n"
        "- The true world w* is included with probability p.\n"
        "- Non-true worlds use q, fixed or random per-run/per-world depending on q_mode.\n\n"
        "Defaults (if not provided):\n"
        "- n=4, x=1, p=0.5, q_mode=random, max_steps=1000 (simulate), seed=random.\n"
        "- q is required only when q_mode=fixed (or for marked worlds); otherwise it is sampled.\n\n"
        "Examples:\n"
        "  truth-tracking simulate --n 4 --x 1 --p 0.5 --q 0.2 --nb-runs 100\n"
        "  truth-tracking simulate --q-mode fixed --q 0.2 --stop-on-absorption\n"
        "  truth-tracking single --n 4 --q 0.2 --max-steps 1000\n"
        "  truth-tracking trace --n 4 --q 0.2 --max-steps 30 --plot-output --print-ranks\n"
        "  truth-tracking batchpq --n 6 --nb-runs 1000 --max-steps 10000\n"
        "  truth-tracking batchpq-plot batchpq_20260204_130100_n6_x1_nbr1000_ms10000_ws50_fw50_seednone_stop0_initflat_fast0.csv\n"
        "  truth-tracking batchpq-plotdiff batchpq_20260204_130100_n6_x1_nbr1000_ms10000_ws50_fw50_seednone_stop0_initflat_fast0.csv --diffpq 0.03 0.08\n"
        "  truth-tracking batchpmaxstep --bound-max-steps 200 --fast\n"
    ),
)
console = Console()


class QMode(str, Enum):
    random = "random"
    fixed = "fixed"
    uniform = "uniform"


def _validate_q(
    p: float,
    q: float | None,
    q_mode: QMode,
    *,
    use_marked_worlds: bool = False,
    nb_marked_worlds: int | None = None,
    n: int | None = None,
) -> None:
    if not (0.0 < p < 1.0):
        raise typer.BadParameter("Require 0 < p < 1.")
    if q_mode == QMode.fixed:
        if q is None or not (0.0 < q < p):
            raise typer.BadParameter("Require 0 < q < p when --q-mode fixed.")
    if use_marked_worlds:
        if n is None:
            raise typer.BadParameter("n is required when using marked worlds.")
        max_marked = (2 ** n) - 1
        if nb_marked_worlds is None or not (1 <= nb_marked_worlds <= max_marked):
            raise typer.BadParameter(
                "Require 1 <= nb_marked_worlds <= 2^n - 1 when using marked worlds."
            )
        if q_mode != QMode.random:
            if q is None or not (0.0 < q < p):
                raise typer.BadParameter("Require 0 < q < p when using marked worlds.")


def _resolve_q_mode(ctx: typer.Context, q_mode: QMode, q: float | None) -> QMode:
    if q is None:
        return q_mode
    if q_mode == QMode.fixed:
        return q_mode
    source = ctx.get_parameter_source("q_mode")
    if source == click.core.ParameterSource.DEFAULT:
        return QMode.fixed
    raise typer.BadParameter("Cannot combine --q with --q-mode other than fixed.")


class WorldFormat(str, Enum):
    bits = "bits"
    integer = "int"


def _label_world(world: int, n: int, fmt: WorldFormat) -> str:
    if fmt == WorldFormat.bits:
        return world_to_bits(world, n)
    return str(world)


def _format_world_set(worlds: list[int], n: int, fmt: WorldFormat) -> str:
    return "{" + ", ".join(_label_world(w, n, fmt) for w in worlds) + "}"


def _format_ranks(ranks: list[int], n: int, fmt: WorldFormat) -> str:
    groups: dict[int, list[str]] = {}
    for w, r in enumerate(ranks):
        groups.setdefault(r, []).append(_label_world(w, n, fmt))
    parts = []
    for r in sorted(groups.keys()):
        worlds = "/".join(groups[r])
        parts.append(f"{r}:{worlds}")
    return " ".join(parts)


def _print_options(title: str, options: list[tuple[str, object]]) -> None:
    table = Table(title=title)
    table.add_column("Option", style="bold")
    table.add_column("Value")
    for name, value in options:
        table.add_row(f"--{name.replace('_', '-')}", str(value))
    console.print(table)


def _batch_output_path(
    *,
    n: int,
    x: int,
    nb_runs: int,
    max_steps: int,
    window_size: int,
    frequency_window: int,
    seed: int | None,
    stop_on_absorption: bool,
    init_mode: InitMode,
    valuemaxall: int,
    valuemin_star: int,
    valuemax_star: int,
    fast: bool,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_label = "none" if seed is None else str(seed)
    parts = [
        f"batchpq_{timestamp}",
        f"n{n}",
        f"x{x}",
        f"nbr{nb_runs}",
        f"ms{max_steps}",
        f"ws{window_size}",
        f"fw{frequency_window}",
        f"seed{seed_label}",
        f"stop{int(stop_on_absorption)}",
        f"init{init_mode.value}",
        f"fast{int(fast)}",
    ]
    if init_mode == InitMode.random:
        parts.extend(
            [
                f"vmax{valuemaxall}",
                f"vmin{valuemin_star}",
                f"vmaxs{valuemax_star}",
            ]
        )
    filename = "_".join(parts) + ".csv"
    return Path.cwd() / filename


def _batchpmaxstep_output_path(
    *,
    n: int,
    x: int,
    nb_runs: int,
    bound_max_steps: int,
    window_size: int,
    frequency_window: int,
    seed: int | None,
    stop_on_absorption: bool,
    init_mode: InitMode,
    valuemaxall: int,
    valuemin_star: int,
    valuemax_star: int,
    fast: bool,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_label = "none" if seed is None else str(seed)
    parts = [
        f"batchpmaxstep_{timestamp}",
        f"n{n}",
        f"x{x}",
        f"nbr{nb_runs}",
        f"bms{bound_max_steps}",
        f"ws{window_size}",
        f"fw{frequency_window}",
        f"seed{seed_label}",
        f"stop{int(stop_on_absorption)}",
        f"init{init_mode.value}",
        f"fast{int(fast)}",
    ]
    if init_mode == InitMode.random:
        parts.extend(
            [
                f"vmax{valuemaxall}",
                f"vmin{valuemin_star}",
                f"vmaxs{valuemax_star}",
            ]
        )
    filename = "_".join(parts) + ".csv"
    return Path.cwd() / filename


class InitMode(str, Enum):
    flat = "flat"
    random = "random"


class Plot3DMode(str, Enum):
    heatmap = "heatmap"
    scatter = "scatter"
    surface = "surface"
    wireframe = "wireframe"
    contour = "contour"
    contour2d = "contour2d"


class SmoothMode(str, Enum):
    grid_linear = "grid-linear"
    grid_cubic = "grid-cubic"
    poly2 = "poly2"
    poly3 = "poly3"
    overlay_linear = "overlay-linear"
    overlay_cubic = "overlay-cubic"


@app.command()
def simulate(
    ctx: typer.Context,
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    p: float = typer.Option(
        0.5,
        help="Probability that the true world w* is included in each formula.",
        rich_help_panel="Core options",
    ),
    q_mode: QMode = typer.Option(
        QMode.random,
        help=(
            "How to choose q for non-true worlds: random = per-run random q in (0,p), "
            "fixed = constant q, uniform = per-world random in (0,p)."
        ),
        rich_help_panel="Core options",
    ),
    q: float | None = typer.Option(
        None,
        help=(
            "Fixed probability for non-true worlds (required when q_mode=fixed or using marked worlds; "
            "passing --q implicitly sets --q-mode fixed)."
        ),
        rich_help_panel="Core options",
    ),
    nb_runs: int = typer.Option(
        100,
        help="Number of independent runs in the benchmark.",
        rich_help_panel="Run control",
    ),
    max_steps: int = typer.Option(
        1000,
        help="Maximum number of update steps per run before stopping.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute mean tail frequency in the benchmark.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed base for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop a run early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    progress: bool = typer.Option(
        False,
        help="Print one progress line per run (steps, first hit, absorption time).",
        rich_help_panel="Run control",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
    fast: bool = typer.Option(
        False,
        help="Use coarse steps of 0.1 for p and q instead of 0.01.",
        rich_help_panel="Run control",
    ),
):
    """
    Run a benchmark at a single (p, q) setting and summarize the results.

    This command runs many independent simulations with the same parameters, then prints
    summary statistics (means, counts, etc.) to the console. It does not write a CSV.

    Defaults (if not provided):
    - n=4, x=1, p=0.5, q_mode=random, nb_runs=100, max_steps=1000,
      window_size=50, frequency_window=50, seed=random, stop_on_absorption=false,
      progress=false.

    Examples:
    - truth-tracking simulate --n 4 --x 1 --p 0.5 --q 0.2 --nb-runs 100
      Fixed q=0.2, 100 runs, summary printed to the console.
    - truth-tracking simulate --q-mode random --p 0.6 --nb-runs 500
      Random q per run in (0, p), 500 runs for smoother averages.
    - truth-tracking simulate --q-mode fixed --q 0.15 --stop-on-absorption
      Early-stop once absorption is detected (a full window of Bel={w*}).
    - truth-tracking simulate --seed 123 --progress true
      Reproducible runs with per-run progress lines.
    """
    q_mode = _resolve_q_mode(ctx, q_mode, q)
    _validate_q(
        p,
        q,
        q_mode,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        n=n,
    )

    result = run_benchmark(
        n=n,
        x=x,
        p=p,
        q=q,
        q_mode=q_mode.value,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        num_runs=nb_runs,
        max_steps=max_steps,
        window_size=window_size,
        frequency_window=frequency_window,
        seed=seed,
        stop_on_absorption=stop_on_absorption,
        progress=progress,
        init_mode=init_mode.value,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
    )

    _print_options(
        "Simulate Options",
        [
            ("n", n),
            ("x", x),
            ("p", p),
            ("q_mode", q_mode.value),
            ("q", q),
            ("nb_runs", nb_runs),
            ("max_steps", max_steps),
            ("window_size", window_size),
            ("frequency_window", frequency_window),
            ("seed", seed),
            ("init_mode", init_mode.value),
            ("valuemaxall", valuemaxall),
            ("valuemin_star", valuemin_star),
            ("valuemax_star", valuemax_star),
            ("stop_on_absorption", stop_on_absorption),
            ("progress", progress),
            ("use_marked_worlds", use_marked_worlds),
            ("nb_marked_worlds", nb_marked_worlds),
        ],
    )

    table = Table(title="Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Mean absorption time", str(result.mean_absorption_time))
    table.add_row("Mean first-hit time", str(result.mean_first_hit_time))
    table.add_row("Mean tail frequency", str(result.mean_tail_frequency))
    table.add_row("Mean entrenchment degree", str(result.mean_entrenchment_degree))
    table.add_row(
        "Mean max non-true formula frequency",
        str(result.mean_max_nontrue_formula_frequency),
    )
    table.add_row(
        "Mean true-world formula frequency",
        str(result.mean_true_world_formula_frequency),
    )
    table.add_row("Runs reaching absorption", str(result.df["absorption_time"].notna().sum()))
    table.add_row("Runs reaching first hit", str(result.df["first_hit_time"].notna().sum()))

    console.print(table)

    if result.df["absorption_time"].notna().any():
        qtable = Table(title="Absorption time quantiles")
        qtable.add_column("Quantile")
        qtable.add_column("Value")
        quants = result.df["absorption_time"].dropna().quantile([0.1, 0.5, 0.9])
        for qv, v in quants.items():
            qtable.add_row(str(qv), str(float(v)))
        console.print(qtable)


@app.command(name="batchpq")
def batchpq(
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    nb_runs: int = typer.Option(
        100,
        help="Number of independent runs in the benchmark.",
        rich_help_panel="Run control",
    ),
    max_steps: int = typer.Option(
        1000,
        help="Maximum number of update steps per run before stopping.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute mean tail frequency in the benchmark.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed base for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop a run early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    progress: bool = typer.Option(
        False,
        help="Print one progress line per (p, q) pair.",
        rich_help_panel="Run control",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
    fast: bool = typer.Option(
        False,
        help="Use coarse steps of 0.1 for p and q instead of 0.01.",
        rich_help_panel="Run control",
    ),
):
    """
    Run a grid of (p, q) benchmarks and save results to a CSV file.

    This command sweeps over p and q (with q < p), runs a benchmark at each pair,
    and writes one CSV row per (p, q). The output file is saved in the current
    working directory with a `batchpq_...csv` filename.

    Examples:
    - truth-tracking batchpq
      Run the default grid and write `batchpq_YYYYMMDD_HHMMSS_...csv` in cwd.
    - truth-tracking batchpq --n 6 --nb-runs 1000 --max-steps 10000
      Larger world count and longer runs for smoother estimates.
    - truth-tracking batchpq --fast
      Coarse grid with 0.1 steps for p and q (much faster, less resolution).
    - truth-tracking batchpq --use-marked-worlds --nb-marked-worlds 10
      Restrict non-true worlds to a marked subset across the grid.
    """
    if use_marked_worlds:
        if nb_marked_worlds is None:
            raise typer.BadParameter(
                "nb_marked_worlds must be set when use_marked_worlds is True."
            )
        max_marked = (2 ** n) - 1
        if not (1 <= nb_marked_worlds <= max_marked):
            raise typer.BadParameter("Require 1 <= nb_marked_worlds <= 2^n - 1.")

    _print_options(
        "BatchPQ Options",
        [
            ("n", n),
            ("x", x),
            ("nb_runs", nb_runs),
            ("max_steps", max_steps),
            ("window_size", window_size),
            ("frequency_window", frequency_window),
            ("seed", seed),
            ("init_mode", init_mode.value),
            ("valuemaxall", valuemaxall),
            ("valuemin_star", valuemin_star),
            ("valuemax_star", valuemax_star),
            ("stop_on_absorption", stop_on_absorption),
            ("progress", progress),
            ("use_marked_worlds", use_marked_worlds),
            ("nb_marked_worlds", nb_marked_worlds),
            ("fast", fast),
        ],
    )

    output_path = _batch_output_path(
        n=n,
        x=x,
        nb_runs=nb_runs,
        max_steps=max_steps,
        window_size=window_size,
        frequency_window=frequency_window,
        seed=seed,
        stop_on_absorption=stop_on_absorption,
        init_mode=init_mode,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
        fast=fast,
    )

    output_path.write_text(
        "p,q,mean_absorption_time,mean_first_hit_time,mean_tail_frequency,"
        "mean_entrenchment_degree,mean_max_nontrue_formula_frequency,"
        "mean_true_world_formula_frequency,runs_reaching_absorption,runs_reaching_first_hit\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as f:
        step = 10 if fast else 1
        for p_int in range(2 * step, 100, step):
            p = p_int / 100.0
            for q_int in range(step, p_int, step):
                q = q_int / 100.0
                result = run_benchmark(
                    n=n,
                    x=x,
                    p=p,
                    q=q,
                    q_mode="fixed",
                    use_marked_worlds=use_marked_worlds,
                    nb_marked_worlds=nb_marked_worlds,
                    num_runs=nb_runs,
                    max_steps=max_steps,
                    window_size=window_size,
                    frequency_window=frequency_window,
                    seed=seed,
                    stop_on_absorption=stop_on_absorption,
                    progress=False,
                    init_mode=init_mode.value,
                    valuemaxall=valuemaxall,
                    valuemin_star=valuemin_star,
                    valuemax_star=valuemax_star,
                )
                runs_absorption = int(result.df["absorption_time"].notna().sum())
                runs_first_hit = int(result.df["first_hit_time"].notna().sum())
                if progress:
                    console.print(
                        f"batchpq p={p:.2f} q={q:.2f} | "
                        f"mean_absorption={result.mean_absorption_time} | "
                        f"mean_first_hit={result.mean_first_hit_time} | "
                        f"mean_tail={result.mean_tail_frequency} | "
                        f"mean_entrenchment={result.mean_entrenchment_degree} | "
                        f"mean_max_nontrue_freq={result.mean_max_nontrue_formula_frequency} | "
                        f"mean_true_world_freq={result.mean_true_world_formula_frequency} | "
                        f"runs_absorption={runs_absorption} | "
                        f"runs_first_hit={runs_first_hit}"
                    )
                row = ",".join(
                    [
                        f"{p:.2f}",
                        f"{q:.2f}",
                        ""
                        if result.mean_absorption_time is None
                        else str(result.mean_absorption_time),
                        ""
                        if result.mean_first_hit_time is None
                        else str(result.mean_first_hit_time),
                        ""
                        if result.mean_tail_frequency is None
                        else str(result.mean_tail_frequency),
                        ""
                        if result.mean_entrenchment_degree is None
                        else str(result.mean_entrenchment_degree),
                        ""
                        if result.mean_max_nontrue_formula_frequency is None
                        else str(result.mean_max_nontrue_formula_frequency),
                        ""
                        if result.mean_true_world_formula_frequency is None
                        else str(result.mean_true_world_formula_frequency),
                        str(runs_absorption),
                        str(runs_first_hit),
                    ]
                )
                f.write(f"{row}\n")

    console.print(f"Saved batchpq results to {output_path}")


@app.command(name="batchpmaxstep")
def batchpmaxstep(
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    bound_max_steps: int = typer.Option(
        ...,
        help="Upper bound for max-steps sweep (must be >= 10).",
        rich_help_panel="Run control",
    ),
    nb_runs: int = typer.Option(
        100,
        help="Number of independent runs in the benchmark.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute mean tail frequency in the benchmark.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed base for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop a run early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    progress: bool = typer.Option(
        False,
        help="Print one progress line per (p, max-steps) pair.",
        rich_help_panel="Run control",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
    fast: bool = typer.Option(
        False,
        help="Use coarse steps of 0.1 for p instead of 0.01.",
        rich_help_panel="Run control",
    ),
):
    """
    Run a grid of (p, max-steps) benchmarks with q sampled randomly per run.

    This command sweeps p in {0.02, ..., 0.99} and max-steps in {10, ..., bound-max-steps}.
    For each pair (p, max-steps), it runs a benchmark with q-mode random, then writes
    one CSV row. Output is saved in the current working directory with a
    `batchpmaxstep_...csv` filename.

    Constraints:
    - --bound-max-steps is required and must be >= 10.
    - --q and --q-mode are not available here; q is randomized per run in (0, p).
    - --p and --max-steps are not available here; they are swept by design.

    Examples:
    - truth-tracking batchpmaxstep --bound-max-steps 200
      Run the full p grid (0.02..0.99) with max-steps 10..200.
    - truth-tracking batchpmaxstep --bound-max-steps 500 --nb-runs 1000
      More runs for smoother averages across a larger max-steps range.
    - truth-tracking batchpmaxstep --bound-max-steps 200 --fast
      Coarse p grid (step 0.1), full max-steps range.
    - truth-tracking batchpmaxstep --bound-max-steps 200 --use-marked-worlds --nb-marked-worlds 10
      Restrict non-true worlds to a marked subset across the sweep.
    """
    if bound_max_steps < 10:
        raise typer.BadParameter("--bound-max-steps must be >= 10.")
    if use_marked_worlds:
        if nb_marked_worlds is None:
            raise typer.BadParameter(
                "nb_marked_worlds must be set when use_marked_worlds is True."
            )
        max_marked = (2 ** n) - 1
        if not (1 <= nb_marked_worlds <= max_marked):
            raise typer.BadParameter("Require 1 <= nb_marked_worlds <= 2^n - 1.")

    _print_options(
        "BatchPMaxStep Options",
        [
            ("n", n),
            ("x", x),
            ("bound_max_steps", bound_max_steps),
            ("nb_runs", nb_runs),
            ("window_size", window_size),
            ("frequency_window", frequency_window),
            ("seed", seed),
            ("init_mode", init_mode.value),
            ("valuemaxall", valuemaxall),
            ("valuemin_star", valuemin_star),
            ("valuemax_star", valuemax_star),
            ("stop_on_absorption", stop_on_absorption),
            ("progress", progress),
            ("use_marked_worlds", use_marked_worlds),
            ("nb_marked_worlds", nb_marked_worlds),
            ("fast", fast),
        ],
    )

    output_path = _batchpmaxstep_output_path(
        n=n,
        x=x,
        nb_runs=nb_runs,
        bound_max_steps=bound_max_steps,
        window_size=window_size,
        frequency_window=frequency_window,
        seed=seed,
        stop_on_absorption=stop_on_absorption,
        init_mode=init_mode,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
        fast=fast,
    )

    output_path.write_text(
        "p,max_steps,mean_absorption_time,mean_first_hit_time,mean_tail_frequency,"
        "mean_entrenchment_degree,mean_max_nontrue_formula_frequency,"
        "mean_true_world_formula_frequency,runs_reaching_absorption,runs_reaching_first_hit\n",
        encoding="utf-8",
    )

    with output_path.open("a", encoding="utf-8") as f:
        p_step = 10 if fast else 1
        for p_int in range(2, 100, p_step):
            p = p_int / 100.0
            if seed is None:
                seeds = [None] * nb_runs
            else:
                seeds = [seed + k for k in range(nb_runs)]

            sum_tail = np.zeros(bound_max_steps, dtype=np.float64)
            sum_entrench = np.zeros(bound_max_steps, dtype=np.float64)
            sum_max_nontrue = np.zeros(bound_max_steps, dtype=np.float64)
            sum_true_world = np.zeros(bound_max_steps, dtype=np.float64)
            count_tail = np.zeros(bound_max_steps, dtype=np.int64)
            count_entrench = np.zeros(bound_max_steps, dtype=np.int64)
            count_max_nontrue = np.zeros(bound_max_steps, dtype=np.int64)
            count_true_world = np.zeros(bound_max_steps, dtype=np.int64)

            abs_count_diff = np.zeros(bound_max_steps + 2, dtype=np.int64)
            abs_sum_diff = np.zeros(bound_max_steps + 2, dtype=np.float64)
            first_count_diff = np.zeros(bound_max_steps + 2, dtype=np.int64)
            first_sum_diff = np.zeros(bound_max_steps + 2, dtype=np.float64)

            for k in range(nb_runs):
                cfg = RunConfig(
                    n=n,
                    x=x,
                    p=p,
                    q=None,
                    q_mode="random",
                    use_marked_worlds=use_marked_worlds,
                    nb_marked_worlds=nb_marked_worlds,
                    max_steps=bound_max_steps,
                    window_size=window_size,
                    seed=seeds[k],
                    stop_on_absorption=stop_on_absorption,
                    init_mode=init_mode.value,
                    valuemaxall=valuemaxall,
                    valuemin_star=valuemin_star,
                    valuemax_star=valuemax_star,
                    capture_trace=True,
                )
                result = simulate_run(cfg)
                if result.ranks is None or result.formulas is None:
                    raise ValueError("capture_trace=True required for batchpmaxstep.")

                steps_simulated = len(result.hits)
                if steps_simulated == 0:
                    continue

                hits = np.array(result.hits, dtype=np.float64)
                cumsum = np.cumsum(hits)
                tail = np.empty(steps_simulated, dtype=np.float64)
                for t in range(steps_simulated):
                    start = max(0, t - frequency_window + 1)
                    window_sum = cumsum[t] - (cumsum[start - 1] if start > 0 else 0.0)
                    tail[t] = window_sum / (t - start + 1)

                ranks_arr = np.array(result.ranks, dtype=np.int64)
                true_world = result.true_world
                non_true_mask = np.ones(ranks_arr.shape[1], dtype=bool)
                non_true_mask[true_world] = False
                min_non_true = ranks_arr[:, non_true_mask].min(axis=1)
                true_rank = ranks_arr[:, true_world]
                entrench = (min_non_true - true_rank).astype(np.float64)

                num_worlds = ranks_arr.shape[1]
                counts = np.zeros(num_worlds, dtype=np.int64)
                max_nontrue_counts = np.empty(steps_simulated, dtype=np.int64)
                true_counts = np.empty(steps_simulated, dtype=np.int64)
                max_nontrue = 0
                true_count = 0
                for i, formula in enumerate(result.formulas):
                    for w in formula:
                        counts[w] += 1
                        if w == true_world:
                            true_count += 1
                        else:
                            if counts[w] > max_nontrue:
                                max_nontrue = counts[w]
                    max_nontrue_counts[i] = max_nontrue
                    true_counts[i] = true_count
                step_index = np.arange(1, steps_simulated + 1, dtype=np.float64)
                max_nontrue_freq = max_nontrue_counts / step_index
                true_world_freq = true_counts / step_index

                def _pad_last(values: np.ndarray) -> np.ndarray:
                    if steps_simulated >= bound_max_steps:
                        return values[:bound_max_steps]
                    return np.pad(
                        values,
                        (0, bound_max_steps - steps_simulated),
                        constant_values=values[-1],
                    )

                tail_full = _pad_last(tail)
                entrench_full = _pad_last(entrench)
                max_nontrue_full = _pad_last(max_nontrue_freq)
                true_world_full = _pad_last(true_world_freq)

                sum_tail += tail_full
                sum_entrench += entrench_full
                sum_max_nontrue += max_nontrue_full
                sum_true_world += true_world_full
                count_tail += np.isfinite(tail_full)
                count_entrench += np.isfinite(entrench_full)
                count_max_nontrue += np.isfinite(max_nontrue_full)
                count_true_world += np.isfinite(true_world_full)

                if result.absorption_time is not None:
                    start = result.absorption_time + 1
                    if start <= bound_max_steps:
                        abs_count_diff[start] += 1
                        abs_count_diff[bound_max_steps + 1] -= 1
                        abs_sum_diff[start] += result.absorption_time
                        abs_sum_diff[bound_max_steps + 1] -= result.absorption_time
                if result.first_hit_time is not None:
                    start = result.first_hit_time + 1
                    if start <= bound_max_steps:
                        first_count_diff[start] += 1
                        first_count_diff[bound_max_steps + 1] -= 1
                        first_sum_diff[start] += result.first_hit_time
                        first_sum_diff[bound_max_steps + 1] -= result.first_hit_time

            abs_counts = np.cumsum(abs_count_diff)
            abs_sums = np.cumsum(abs_sum_diff)
            first_counts = np.cumsum(first_count_diff)
            first_sums = np.cumsum(first_sum_diff)

            for max_steps in range(10, bound_max_steps + 1):
                idx = max_steps - 1
                runs_absorption = int(abs_counts[max_steps])
                runs_first_hit = int(first_counts[max_steps])
                mean_absorption = (
                    float(abs_sums[max_steps] / abs_counts[max_steps])
                    if abs_counts[max_steps] > 0
                    else None
                )
                mean_first_hit = (
                    float(first_sums[max_steps] / first_counts[max_steps])
                    if first_counts[max_steps] > 0
                    else None
                )
                mean_tail = (
                    float(sum_tail[idx] / count_tail[idx]) if count_tail[idx] > 0 else None
                )
                mean_entrench = (
                    float(sum_entrench[idx] / count_entrench[idx])
                    if count_entrench[idx] > 0
                    else None
                )
                mean_max_nontrue = (
                    float(sum_max_nontrue[idx] / count_max_nontrue[idx])
                    if count_max_nontrue[idx] > 0
                    else None
                )
                mean_true_world = (
                    float(sum_true_world[idx] / count_true_world[idx])
                    if count_true_world[idx] > 0
                    else None
                )
                if progress:
                    console.print(
                        f"batchpmaxstep p={p:.2f} max_steps={max_steps} | "
                        f"mean_absorption={mean_absorption} | "
                        f"mean_first_hit={mean_first_hit} | "
                        f"mean_tail={mean_tail} | "
                        f"mean_entrenchment={mean_entrench} | "
                        f"mean_max_nontrue_freq={mean_max_nontrue} | "
                        f"mean_true_world_freq={mean_true_world} | "
                        f"runs_absorption={runs_absorption} | "
                        f"runs_first_hit={runs_first_hit}"
                    )
                row = ",".join(
                    [
                        f"{p:.2f}",
                        str(max_steps),
                        "" if mean_absorption is None else str(mean_absorption),
                        "" if mean_first_hit is None else str(mean_first_hit),
                        "" if mean_tail is None else str(mean_tail),
                        "" if mean_entrench is None else str(mean_entrench),
                        "" if mean_max_nontrue is None else str(mean_max_nontrue),
                        "" if mean_true_world is None else str(mean_true_world),
                        str(runs_absorption),
                        str(runs_first_hit),
                    ]
                )
                f.write(f"{row}\n")

    console.print(f"Saved batchpmaxstep results to {output_path}")


@app.command()
def single(
    ctx: typer.Context,
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    p: float = typer.Option(
        0.5,
        help="Probability that the true world w* is included in each formula.",
        rich_help_panel="Core options",
    ),
    q_mode: QMode = typer.Option(
        QMode.random,
        help=(
            "How to choose q for non-true worlds: random = per-run random q in (0,p), "
            "fixed = constant q, uniform = per-world random in (0,p)."
        ),
        rich_help_panel="Core options",
    ),
    q: float | None = typer.Option(
        None,
        help=(
            "Fixed probability for non-true worlds (required when q_mode=fixed or using marked worlds; "
            "passing --q implicitly sets --q-mode fixed)."
        ),
        rich_help_panel="Core options",
    ),
    max_steps: int = typer.Option(
        1000,
        help="Maximum number of update steps before stopping.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute mean tail frequency for the run.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
):
    """
    Simulate one run and print its summary metrics.

    This command runs a single simulation and reports absorption time, first-hit time,
    tail frequency, and other metrics to the console. It is the quickest way to inspect
    behavior for a specific parameter setting.

    Defaults (if not provided):
    - n=4, x=1, p=0.5, q_mode=random, max_steps=1000, window_size=50,
      seed=random, stop_on_absorption=false.

    Examples:
    - truth-tracking single --n 4 --p 0.5 --q 0.2
      Fixed q=0.2, one run, summary printed to the console.
    - truth-tracking single --q-mode uniform --p 0.6
      Per-world random q in (0, p) for each non-true world.
    - truth-tracking single --seed 42 --stop-on-absorption false
      Reproducible run, continue after absorption is detected.
    - truth-tracking single --use-marked-worlds --nb-marked-worlds 7 --q 0.15
      Restrict non-true worlds to a marked subset with fixed q.
    """
    q_mode = _resolve_q_mode(ctx, q_mode, q)
    _validate_q(
        p,
        q,
        q_mode,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        n=n,
    )
    cfg = RunConfig(
        n=n,
        x=x,
        p=p,
        q=q,
        q_mode=q_mode.value,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        max_steps=max_steps,
        seed=seed,
        window_size=window_size,
        stop_on_absorption=stop_on_absorption,
        init_mode=init_mode.value,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
    )
    result = simulate_run(cfg)

    _print_options(
        "Single Options",
        [
            ("n", n),
            ("x", x),
            ("p", p),
            ("q_mode", q_mode.value),
            ("q", q),
            ("max_steps", max_steps),
            ("window_size", window_size),
            ("frequency_window", frequency_window),
            ("seed", seed),
            ("init_mode", init_mode.value),
            ("valuemaxall", valuemaxall),
            ("valuemin_star", valuemin_star),
            ("valuemax_star", valuemax_star),
            ("stop_on_absorption", stop_on_absorption),
            ("use_marked_worlds", use_marked_worlds),
            ("nb_marked_worlds", nb_marked_worlds),
        ],
    )

    steps_simulated = len(result.hits)
    window = min(frequency_window, steps_simulated) if steps_simulated > 0 else 0
    tail_freq = float(np.mean(result.hits[-window:])) if window > 0 else None
    summary = Table(title="Single Run Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("True world w*", str(result.true_world))
    summary.add_row("Absorption time", str(result.absorption_time))
    summary.add_row("First hit time", str(result.first_hit_time))
    summary.add_row("Tail frequency", str(tail_freq))
    summary.add_row("Entrenchment degree", str(result.entrenchment_degree))
    summary.add_row(
        "Max non-true formula frequency",
        str(result.max_nontrue_formula_frequency),
    )
    summary.add_row(
        "True-world formula frequency",
        str(result.true_world_formula_frequency),
    )
    summary.add_row("Steps simulated", str(len(result.beliefs)))
    console.print(summary)


@app.command()
def frequency(
    ctx: typer.Context,
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    p: float = typer.Option(
        0.5,
        help="Probability that the true world w* is included in each formula.",
        rich_help_panel="Core options",
    ),
    q_mode: QMode = typer.Option(
        QMode.random,
        help=(
            "How to choose q for non-true worlds: random = per-run random q in (0,p), "
            "fixed = constant q, uniform = per-world random in (0,p)."
        ),
        rich_help_panel="Core options",
    ),
    q: float | None = typer.Option(
        None,
        help=(
            "Fixed probability for non-true worlds (required when q_mode=fixed or using marked worlds; "
            "passing --q implicitly sets --q-mode fixed)."
        ),
        rich_help_panel="Core options",
    ),
    num_runs: int = typer.Option(
        100,
        help="Number of independent runs in the benchmark.",
        rich_help_panel="Run control",
    ),
    max_steps: int = typer.Option(
        1000,
        help="Maximum number of update steps per run before stopping.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute mean tail frequency in the benchmark.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed base for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    out_csv: Path | None = typer.Option(
        None,
        help="Optional path to save the frequency curve as CSV.",
        rich_help_panel="Run control",
    ),
    plot: bool = typer.Option(
        False, help="Plot the frequency curve. (deprecated)", rich_help_panel="Run control"
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop a run early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    progress: bool = typer.Option(
        False,
        help="Print one progress line per run (steps, first hit, absorption time).",
        rich_help_panel="Run control",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
):
    """
    Deprecated. Compute the average convergence-frequency curve across runs.

    This command is kept for legacy workflows. Prefer `simulate` unless you specifically
    need the full frequency curve as a CSV or plot.

    Defaults (if not provided):
    - n=4, x=1, p=0.5, q_mode=random, num_runs=100, max_steps=1000,
      window_size=50, frequency_window=50, seed=random, stop_on_absorption=false,
      progress=false.

    Examples:
    - truth-tracking frequency --q 0.2 --plot
      Plot the average frequency curve over 1000 steps (default run count).
    - truth-tracking frequency --out-csv results.csv
      Save the curve as `t,frequency` for external plotting.
    - truth-tracking frequency --num-runs 1000 --max-steps 5000
      Smoother curve using more runs and longer trajectories.
    - truth-tracking frequency --q-mode fixed --q 0.2
      Fixed q=0.2 for all non-true worlds.

    Interpretation:
    - The curve at time t is the average, across runs, of the fraction of steps
      up to t where Bel(rf) = {w*}. It is a running frequency, not an instantaneous rate.
    - "Final average frequency" is the curve value at the last simulated step.
    - "Mean tail frequency" averages the last `frequency_window` steps per run.
    """
    console.print(
        "[yellow]Warning:[/yellow] The 'frequency' command is deprecated. "
        "Use 'simulate' instead unless you specifically need the curve output."
    )
    q_mode = _resolve_q_mode(ctx, q_mode, q)
    _validate_q(
        p,
        q,
        q_mode,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        n=n,
    )
    result = run_benchmark(
        n=n,
        x=x,
        p=p,
        q=q,
        q_mode=q_mode.value,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        num_runs=num_runs,
        max_steps=max_steps,
        window_size=window_size,
        frequency_window=frequency_window,
        seed=seed,
        stop_on_absorption=stop_on_absorption,
        progress=progress,
        init_mode=init_mode.value,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
    )

    if result.frequency_curve is None:
        console.print("No frequency curve produced.")
        return

    console.print("\n[bold cyan]Frequency Curve Summary[/bold cyan]")
    console.print(f"Final average frequency = {float(result.frequency_curve[-1]):.4f}")
    console.print(f"Mean tail frequency = {result.mean_tail_frequency}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("t,frequency\n", encoding="utf-8")
        with out_csv.open("a", encoding="utf-8") as f:
            for t, v in enumerate(result.frequency_curve):
                f.write(f"{t},{float(v)}\n")
        console.print(f"Saved curve to {out_csv}")

    if plot:
        final_avg = float(result.frequency_curve[-1]) if result.frequency_curve is not None else None
        if q_mode == QMode.fixed:
            q_label = f"{q}"
        else:
            q_label = "per-world random"
        legend_title = (
            f"n={n}, x={x}, p={p}, q={q_label}\n"
            f"runs={num_runs}, max_steps={max_steps}, "
            f"abs_window={window_size}, tail_window={frequency_window}"
        )
        plot_frequency_curve(
            result.frequency_curve,
            final_avg=final_avg,
            tail_avg=result.mean_tail_frequency,
            tail_curve=result.tail_frequency_curve,
            tail_window=frequency_window,
            legend_title=legend_title,
        )


@app.command()
def trace(
    ctx: typer.Context,
    n: int = typer.Option(
        4,
        help="Number of propositional variables; total worlds = 2^n.",
        rich_help_panel="Core options",
    ),
    x: int = typer.Option(
        1,
        help="Improvement operator strength; subtract x from worlds in F.",
        rich_help_panel="Core options",
    ),
    p: float = typer.Option(
        0.5,
        help="Probability that the true world w* is included in each formula.",
        rich_help_panel="Core options",
    ),
    q_mode: QMode = typer.Option(
        QMode.random,
        help=(
            "How to choose q for non-true worlds: random = per-run random q in (0,p), "
            "fixed = constant q, uniform = per-world random in (0,p)."
        ),
        rich_help_panel="Core options",
    ),
    q: float | None = typer.Option(
        None,
        help=(
            "Fixed probability for non-true worlds (required when q_mode=fixed or using marked worlds; "
            "passing --q implicitly sets --q-mode fixed)."
        ),
        rich_help_panel="Core options",
    ),
    max_steps: int = typer.Option(
        1000,
        help="Maximum number of update steps before stopping.",
        rich_help_panel="Run control",
    ),
    window_size: int = typer.Option(
        50,
        help="Absorption detection window: number of consecutive steps with Bel={w*}.",
        rich_help_panel="Run control",
    ),
    frequency_window: int = typer.Option(
        50,
        help="Tail window length used to compute tail frequency for the run.",
        rich_help_panel="Run control",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed for reproducibility (default: random).",
        rich_help_panel="Run control",
    ),
    init_mode: InitMode = typer.Option(
        InitMode.flat,
        help="Initial ranking: flat (all zeros) or random.",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemaxall: int = typer.Option(
        0,
        help="Max rank for random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemin_star: int = typer.Option(
        0,
        help="Min rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    valuemax_star: int = typer.Option(
        0,
        help="Max rank for true world in random init (used when init_mode=random).",
        rich_help_panel="Initialization (depends on --init-mode)",
    ),
    stop_on_absorption: bool = typer.Option(
        False,
        help="Stop early once absorption is detected (a full window of Bel={w*}).",
        rich_help_panel="Run control",
    ),
    out_csv: Path | None = typer.Option(
        None,
        help="Optional path to save the trace as CSV.",
        rich_help_panel="Run control",
    ),
    world_format: WorldFormat = typer.Option(
        WorldFormat.integer,
        help="World display format: int or bits.",
        rich_help_panel="Run control",
    ),
    print_steps: bool = typer.Option(
        False, help="Print each step to the console.", rich_help_panel="Run control"
    ),
    print_ranks: bool = typer.Option(
        False, help="Include full ranks in console output.", rich_help_panel="Run control"
    ),
    plot_output: bool = typer.Option(
        False,
        help="Save the plot to a file (uses --plot-output-path or a default path).",
        rich_help_panel="Plot output (depends on --plot-output)",
    ),
    plot_output_path: Path | None = typer.Option(
        None,
        help="File path to save the plot (png, pdf, etc.).",
        rich_help_panel="Plot output (depends on --plot-output)",
    ),
    use_marked_worlds: bool = typer.Option(
        False,
        help="Restrict non-true worlds to a marked subset.",
        rich_help_panel="Marked worlds (rare)",
    ),
    nb_marked_worlds: int | None = typer.Option(
        None,
        help="Number of marked non-true worlds (requires --use-marked-worlds).",
        rich_help_panel="Marked worlds (rare)",
    ),
):
    """
    Run a single trace and display formula, belief, and ranking evolution.

    This command captures the full trajectory of a single run. It can print each step,
    export the trace to CSV, and optionally save a plot of the rank evolution.

    Defaults (if not provided):
    - n=4, x=1, p=0.5, q_mode=random, max_steps=1000, window_size=50,
      seed=random, stop_on_absorption=false, world_format=int,
      print_steps=false, print_ranks=false.

    Examples:
    - truth-tracking trace --n 4 --q 0.2 --max-steps 30 --print-steps
      Show every step for a short run.
    - truth-tracking trace --world-format bits --print-ranks
      Display worlds as bitstrings and include full ranks.
    - truth-tracking trace --out-csv trace.csv --plot-output
      Save the trace to CSV and write a plot image.
    - truth-tracking trace --plot-output-path my_trace.png
      Save the plot to a custom filename (implies --plot-output).
    - truth-tracking trace --q-mode fixed --q 0.2 --seed 123
      Fixed q with a reproducible run.
    """
    q_mode = _resolve_q_mode(ctx, q_mode, q)
    if plot_output_path is not None and not plot_output:
        plot_output = True
    if plot_output and plot_output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_output_path = Path.cwd() / f"trace_plot_{timestamp}.png"
    _validate_q(
        p,
        q,
        q_mode,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        n=n,
    )
    cfg = RunConfig(
        n=n,
        x=x,
        p=p,
        q=q,
        q_mode=q_mode.value,
        use_marked_worlds=use_marked_worlds,
        nb_marked_worlds=nb_marked_worlds,
        max_steps=max_steps,
        seed=seed,
        window_size=window_size,
        stop_on_absorption=stop_on_absorption,
        capture_trace=True,
        init_mode=init_mode.value,
        valuemaxall=valuemaxall,
        valuemin_star=valuemin_star,
        valuemax_star=valuemax_star,
    )
    result = simulate_run(cfg)

    if result.formulas is None or result.ranks is None:
        console.print("Trace not captured.")
        return

    _print_options(
        "Trace Options",
        [
            ("n", n),
            ("x", x),
            ("p", p),
            ("q_mode", q_mode.value),
            ("q", q),
            ("max_steps", max_steps),
            ("window_size", window_size),
            ("frequency_window", frequency_window),
            ("seed", seed),
            ("init_mode", init_mode.value),
            ("valuemaxall", valuemaxall),
            ("valuemin_star", valuemin_star),
            ("valuemax_star", valuemax_star),
            ("stop_on_absorption", stop_on_absorption),
            ("out_csv", out_csv),
            ("world_format", world_format.value),
            ("print_steps", print_steps),
            ("print_ranks", print_ranks),
            ("plot_output", plot_output),
            ("plot_output_path", plot_output_path),
            ("use_marked_worlds", use_marked_worlds),
            ("nb_marked_worlds", nb_marked_worlds),
        ],
    )

    steps_simulated = len(result.hits)
    window = min(frequency_window, steps_simulated) if steps_simulated > 0 else 0
    tail_freq = float(np.mean(result.hits[-window:])) if window > 0 else None
    summary = Table(title="Trace Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("True world w*", _label_world(result.true_world, n, world_format))
    summary.add_row("Absorption time", str(result.absorption_time))
    summary.add_row("First hit time", str(result.first_hit_time))
    summary.add_row("Tail frequency", str(tail_freq))
    summary.add_row("Entrenchment degree", str(result.entrenchment_degree))
    summary.add_row(
        "Max non-true formula frequency",
        str(result.max_nontrue_formula_frequency),
    )
    summary.add_row(
        "True-world formula frequency",
        str(result.true_world_formula_frequency),
    )
    summary.add_row("Steps simulated", str(len(result.beliefs)))
    console.print(summary)

    if print_steps:
        if result.initial_ranks is None:
            initial_ranks_list = [0] * (2 ** n)
        else:
            initial_ranks_list = result.initial_ranks
        initial_belief = _format_world_set(
            [i for i, v in enumerate(initial_ranks_list) if v == 0], n, world_format
        )
        initial_ranks = _format_ranks(initial_ranks_list, n, world_format)
        console.print(f"t=-1 | F={{}} | Bel={initial_belief} | ranks={initial_ranks}")
        for t, (formula, belief, ranks) in enumerate(
            zip(result.formulas, result.beliefs, result.ranks)
        ):
            formula_str = _format_world_set(formula, n, world_format)
            belief_str = _format_world_set(sorted(list(belief)), n, world_format)
            if print_ranks:
                ranks_str = _format_ranks(ranks, n, world_format)
                console.print(f"t={t} | F={formula_str} | Bel={belief_str} | ranks={ranks_str}")
            else:
                console.print(f"t={t} | F={formula_str} | Bel={belief_str}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("t,formula,belief,ranks,tail_frequency\n", encoding="utf-8")
        with out_csv.open("a", encoding="utf-8") as f:
            for t, (formula, belief, ranks) in enumerate(
                zip(result.formulas, result.beliefs, result.ranks)
            ):
                formula_str = _format_world_set(formula, n, world_format)
                belief_str = _format_world_set(sorted(list(belief)), n, world_format)
                ranks_str = _format_ranks(ranks, n, world_format)
                if t == len(result.hits) - 1:
                    tail_value = tail_freq
                else:
                    tail_value = ""
            f.write(f"{t},{formula_str},{belief_str},{ranks_str},{tail_value}\n")
        console.print(f"Saved trace to {out_csv}")

    if plot_output:
        if result.ranks is None:
            console.print("No ranks captured for plotting.")
        else:
            initial = result.initial_ranks or ([0] * (2 ** n))
            ranks_over_time = [initial] + result.ranks
            plot_trace_ranks(
                ranks_over_time,
                result.true_world,
                show=False,
                save_path=str(plot_output_path),
            )
            console.print(f"Saved plot to {plot_output_path}")


@app.command(name="batchpq-plot")
def batchpq_plot(
    csv_file: Path = typer.Argument(..., help="Batch CSV file name or path."),
    plot_mode: Plot3DMode = typer.Option(
        Plot3DMode.heatmap,
        help="Plot style: heatmap, scatter, surface, wireframe, contour, or contour2d.",
        rich_help_panel="Plot options",
    ),
    smooth: bool = typer.Option(
        False,
        help="Smooth discrete data into a continuous surface (3D modes only).",
        rich_help_panel="Plot options",
    ),
    smooth_mode: SmoothMode = typer.Option(
        SmoothMode.grid_linear,
        help="Smoothing method for 3D modes.",
        rich_help_panel="Plot options",
    ),
) -> None:
    """
    Generate plots for each metric column in a batch CSV (excluding p and q).

    Input should be a CSV produced by `batchpq`. For each metric column, this command
    writes one plot image alongside the CSV, using the CSV stem as a prefix.

    Examples:
    - truth-tracking batchpq-plot batchpq_20260204_130100_n6_x1_nbr1000_ms10000_ws50_fw50_seednone_stop0_initflat_fast0.csv
      Default heatmaps for every metric column.
    - truth-tracking batchpq-plot batchpq_...csv --plot-mode surface --smooth
      Smooth a surface for each metric (requires a 3D plot mode).
    - truth-tracking batchpq-plot batchpq_...csv --plot-mode contour2d
      2D contour plots for each metric.
    - truth-tracking batchpq-plot batchpq_...csv --plot-mode scatter
      Scatter plots of metric values over (p, q).
    """
    if not csv_file.exists():
        raise typer.BadParameter(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {"p", "q"}
    if not required_cols.issubset(df.columns):
        raise typer.BadParameter("CSV must contain 'p' and 'q' columns.")

    metric_cols = [c for c in df.columns if c not in {"p", "q"}]
    if not metric_cols:
        console.print("No metric columns found to plot.")
        return

    p_vals = df["p"].to_numpy(dtype=float)
    q_vals = df["q"].to_numpy(dtype=float)
    valid_pair = q_vals < p_vals
    if not np.all(valid_pair):
        console.print(
            "[yellow]Warning:[/yellow] Dropping rows with q >= p for plotting."
        )
    if smooth and plot_mode in {Plot3DMode.heatmap, Plot3DMode.contour2d}:
        raise typer.BadParameter("--smooth requires a 3D plot-mode.")

    base = csv_file.parent / csv_file.stem

    for col in metric_cols:
        z_vals = df[col].to_numpy(dtype=float)
        mask = np.isfinite(z_vals) & valid_pair
        if not np.any(mask):
            console.print(f"Skipping {col}: no finite values.")
            continue
        safe_name = col.replace(" ", "_")
        out_path = Path(f"{base}_{safe_name}.png")
        p_use = p_vals[mask]
        q_use = q_vals[mask]
        z_use = z_vals[mask]
        if smooth:
            if smooth_mode == SmoothMode.grid_linear:
                P, Q, Z = _interpolate_grid(p_use, q_use, z_use, method="linear")
            elif smooth_mode == SmoothMode.grid_cubic:
                P, Q, Z = _interpolate_grid(p_use, q_use, z_use, method="cubic")
            elif smooth_mode == SmoothMode.poly2:
                P, Q, Z = _polyfit_grid(p_use, q_use, z_use, degree=2)
            elif smooth_mode == SmoothMode.poly3:
                P, Q, Z = _polyfit_grid(p_use, q_use, z_use, degree=3)
            elif smooth_mode == SmoothMode.overlay_linear:
                P, Q, Z = _interpolate_grid(p_use, q_use, z_use, method="linear")
                plot_surface_overlay(
                    p_use,
                    q_use,
                    z_use,
                    P,
                    Q,
                    Z,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
                console.print(f"Saved plot to {out_path}")
                continue
            else:
                P, Q, Z = _interpolate_grid(p_use, q_use, z_use, method="cubic")

            if plot_mode == Plot3DMode.surface:
                plot_surface_from_grid(
                    P, Q, Z, title=f"{col} vs p,q", z_label=col, save_path=str(out_path)
                )
            elif plot_mode == Plot3DMode.wireframe:
                plot_wireframe_from_grid(
                    P, Q, Z, title=f"{col} vs p,q", z_label=col, save_path=str(out_path)
                )
            elif plot_mode == Plot3DMode.contour:
                plot_contour_from_grid(
                    P, Q, Z, title=f"{col} vs p,q", z_label=col, save_path=str(out_path)
                )
            else:
                plot_scatter_from_grid(
                    P, Q, Z, title=f"{col} vs p,q", z_label=col, save_path=str(out_path)
                )
        else:
            if plot_mode == Plot3DMode.surface:
                plot_surface_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
            elif plot_mode == Plot3DMode.wireframe:
                plot_wireframe_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
            elif plot_mode == Plot3DMode.contour:
                plot_contour_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
            elif plot_mode == Plot3DMode.heatmap:
                plot_heatmap_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
            elif plot_mode == Plot3DMode.contour2d:
                plot_contour2d_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
            else:
                plot_scatter_from_points(
                    p_use,
                    q_use,
                    z_use,
                    title=f"{col} vs p,q",
                    z_label=col,
                    save_path=str(out_path),
                )
        console.print(f"Saved plot to {out_path}")


@app.command(name="batchpq-plotdiff", context_settings={"allow_extra_args": True})
def batchpq_plotdiff(
    ctx: typer.Context,
    csv_file: Path = typer.Argument(..., help="Batch CSV file name or path."),
    diffpq: float | None = typer.Option(
        None,
        help=(
            "One or more p - q values (0.01..0.98 in steps of 0.01). "
            "Provide multiple values separated by spaces."
        ),
        rich_help_panel="Plot options",
    ),
    namex: str | None = typer.Option(
        None,
        help="Optional x-axis label (overrides default 'p').",
        rich_help_panel="Plot options",
    ),
    namey: str | None = typer.Option(
        None,
        help="Optional y-axis label (overrides default metric column name).",
        rich_help_panel="Plot options",
    ),
) -> None:
    """
    Plot metric columns against p for rows where (p - q) equals specific values.

    This command reads a `batchpq` CSV and, for each metric column, plots multiple
    curves of the metric vs p at fixed (p - q) offsets. Output files are written next
    to the CSV with a `batchpq-plotdiff_...` suffix indicating the chosen offsets.

    Examples:
    - truth-tracking batchpq-plotdiff batchpq_...csv --diffpq 0.03
      Plot a single (p - q) slice for each metric column.
    - truth-tracking batchpq-plotdiff batchpq_...csv --diffpq 0.01 0.05 0.10
      Plot several slices, each in a different color.
    - truth-tracking batchpq-plotdiff batchpq_...csv --diffpq 0.03 0.08 0.20
      Compare small, medium, and large offsets.
    - truth-tracking batchpq-plotdiff batchpq_...csv --diffpq 0.03 0.08 0.20 0.30
      Up to 10 values are allowed; extra values can also be passed positionally.
    """
    extra_values: list[float] = []
    for arg in ctx.args:
        if arg.startswith("-"):
            raise typer.BadParameter(f"Unexpected argument: {arg}")
        extra_values.append(float(arg))
    values = ([] if diffpq is None else [diffpq]) + extra_values
    if not values:
        raise typer.BadParameter("--diffpq requires at least one value.")
    if len(values) > 10:
        raise typer.BadParameter("--diffpq accepts at most 10 values.")
    diff_values = sorted({round(v, 2) for v in values})
    for v in diff_values:
        if not (0.01 <= v <= 0.98):
            raise typer.BadParameter("--diffpq values must be in [0.01, 0.98].")
        if not np.isclose(v * 100, round(v * 100)):
            raise typer.BadParameter("--diffpq values must be in steps of 0.01.")

    if not csv_file.exists():
        raise typer.BadParameter(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    if not {"p", "q"}.issubset(df.columns):
        raise typer.BadParameter("CSV must contain 'p' and 'q' columns.")

    diffs = (df["p"] - df["q"]).round(2)

    metric_cols = [c for c in df.columns if c not in {"p", "q"}]
    if not metric_cols:
        console.print("No metric columns found to plot.")
        return

    base = csv_file.parent / csv_file.stem
    suffix = "batchpq-plotdiff_" + "-".join(f"{v:.2f}" for v in diff_values)

    for col in metric_cols:
        safe_name = col.replace(" ", "_")
        out_path = Path(f"{base}_{safe_name}_{suffix}.png")
        fig, ax = plt.subplots()
        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(len(diff_values) - 1, 1)) for i in range(len(diff_values))]
        any_curve = False
        for v, color in zip(diff_values, colors):
            filtered = df[diffs == v].sort_values("p")
            if filtered.empty:
                continue
            p_vals = filtered["p"].to_numpy(dtype=float)
            y_vals = filtered[col].to_numpy(dtype=float)
            mask = np.isfinite(y_vals)
            if not np.any(mask):
                continue
            ax.plot(
                p_vals[mask],
                y_vals[mask],
                marker="o",
                markersize=3,
                linewidth=1.0,
                color=color,
                label=f"{v:.2f}",
            )
            any_curve = True
        if not any_curve:
            console.print(f"Skipping {col}: no finite values.")
            plt.close(fig)
            continue
        ax.set_xlabel(namex if namex is not None else "p")
        ax.set_ylabel(namey if namey is not None else col)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(title="p - q", fontsize="small")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        console.print(f"Saved plot to {out_path}")


@app.command(name="batchpmaxstep-plot", context_settings={"allow_extra_args": True})
def batchpmaxstep_plot(
    ctx: typer.Context,
    csv_file: Path = typer.Argument(..., help="Batchpmaxstep CSV file name or path."),
    p: float | None = typer.Option(
        None,
        help=(
            "One or more p values (0.02..0.99 in steps of 0.01). "
            "Provide multiple values separated by spaces."
        ),
        rich_help_panel="Plot options",
    ),
    bound_max_steps: int | None = typer.Option(
        None,
        help=(
            "Optional upper bound for max_steps on the x-axis (must be >= 10). "
            "If omitted, uses the full available range in the CSV."
        ),
        rich_help_panel="Plot options",
    ),
    namex: str | None = typer.Option(
        None,
        help="Optional x-axis label (overrides default 'max_steps').",
        rich_help_panel="Plot options",
    ),
    namey: str | None = typer.Option(
        None,
        help="Optional y-axis label (overrides default 'mean_entrenchment_degree').",
        rich_help_panel="Plot options",
    ),
    linewidth: float = typer.Option(
        0.5,
        help="Line width for plot curves.",
        rich_help_panel="Plot options",
    ),
) -> None:
    """
    Plot mean entrenchment degree against max-steps for selected p values.

    This command reads a `batchpmaxstep` CSV and, for each chosen p value, plots
    mean_entrenchment_degree vs max_steps. The resulting PNG is saved next to the CSV.

    Examples:
    - truth-tracking batchpmaxstep-plot batchpmaxstep_...csv --p 0.10
      Plot a single curve for p=0.10.
    - truth-tracking batchpmaxstep-plot batchpmaxstep_...csv --p 0.10 0.30 0.50
      Plot multiple curves, one per p value.
    - truth-tracking batchpmaxstep-plot batchpmaxstep_...csv --p 0.02 0.08 0.20
      Compare small, medium, and larger p settings.
    - truth-tracking batchpmaxstep-plot batchpmaxstep_...csv --p 0.10 0.30 --bound-max-steps 200
      Truncate curves at max_steps <= 200 (if available).
    """
    extra_values: list[float] = []
    for arg in ctx.args:
        if arg.startswith("-"):
            raise typer.BadParameter(f"Unexpected argument: {arg}")
        extra_values.append(float(arg))
    values = ([] if p is None else [p]) + extra_values
    if not values:
        raise typer.BadParameter("--p requires at least one value.")
    if len(values) > 10:
        raise typer.BadParameter("--p accepts at most 10 values.")
    p_values = sorted({round(v, 2) for v in values})
    for v in p_values:
        if not (0.02 <= v <= 0.99):
            raise typer.BadParameter("--p values must be in [0.02, 0.99].")
        if not np.isclose(v * 100, round(v * 100)):
            raise typer.BadParameter("--p values must be in steps of 0.01.")
    if bound_max_steps is not None and bound_max_steps < 10:
        raise typer.BadParameter("--bound-max-steps must be >= 10.")

    if not csv_file.exists():
        raise typer.BadParameter(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    if not {"p", "max_steps", "mean_entrenchment_degree"}.issubset(df.columns):
        raise typer.BadParameter(
            "CSV must contain 'p', 'max_steps', and 'mean_entrenchment_degree' columns."
        )

    base = csv_file.parent / csv_file.stem
    suffix = "batchpmaxstep-plot_" + "-".join(f"{v:.2f}" for v in p_values)
    if bound_max_steps is not None:
        suffix += f"_bms{bound_max_steps}"
    out_path = Path(f"{base}_{suffix}.png")

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(p_values) - 1, 1)) for i in range(len(p_values))]
    any_curve = False
    for v, color in zip(p_values, colors):
        filtered = df[np.isclose(df["p"], v)]
        if bound_max_steps is not None:
            filtered = filtered[filtered["max_steps"] <= bound_max_steps]
        filtered = filtered.sort_values("max_steps")
        if filtered.empty:
            continue
        x_vals = filtered["max_steps"].to_numpy(dtype=float)
        y_vals = filtered["mean_entrenchment_degree"].to_numpy(dtype=float)
        mask = np.isfinite(y_vals)
        if not np.any(mask):
            continue
        point_size = max(0.1, linewidth * 10.0)
        ax.plot(
            x_vals[mask],
            y_vals[mask],
            marker="o",
            markersize=point_size,
            linewidth=linewidth,
            color=color,
            label=f"{v:.2f}",
        )
        any_curve = True
    if not any_curve:
        console.print("No finite values found for selected p values.")
        plt.close(fig)
        return
    ax.set_xlabel(namex if namex is not None else "max_steps")
    ax.set_ylabel(namey if namey is not None else "mean_entrenchment_degree")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(title="p", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    console.print(f"Saved plot to {out_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
