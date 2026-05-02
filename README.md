# Truth Tracking (Ranking Functions)

This project simulates belief dynamics for an agent whose epistemic state is a **ranking function** over a finite world set.

At each step, the simulator:
1. Samples a non-empty formula `F` over worlds.
2. Applies an improvement operator `f_x` to lower ranks of worlds in `F`.
3. Normalizes ranks so the minimum rank is 0.
4. Tracks whether the believed set is exactly the true world `{w*}`.

The software is designed for reproducible experiments and reviewer-friendly benchmarking.

## What You Can Do With This Software

- Run a **single simulation** and inspect key metrics.
- Run **many simulations** and compute aggregate benchmark statistics.
- Sweep an entire `(p, q)` grid and export results to CSV.
- Sweep `(p, max_steps)` (with random `q`) and export results to CSV.
- Generate publication-ready plots from batch CSVs.
- Trace one run step-by-step (formula, belief, ranks), export CSV, and save a rank heatmap.

## Core Concepts and Metrics

- Worlds: integers in `[0, 2^n - 1]`.
- Belief set `Bel(rf)`: worlds with rank 0.
- Formula generation:
  - true world included with probability `p`
  - non-true worlds included with probability `q` (fixed/uniform/random modes)
- Improvement update `f_x`:
  - subtract `x` from ranks of worlds in `F`
  - normalize by subtracting global minimum rank

Main reported metrics:
- `absorption_time`: first time index of a window of consecutive steps where `Bel = {w*}`
- `first_hit_time`: first time index where `Bel = {w*}`
- `tail_frequency`: mean hit frequency in the trailing window
- `entrenchment_degree`: `min_{w != w*} rf(w) - rf(w*)` at run end
- `max_nontrue_formula_frequency`
- `true_world_formula_frequency`

## Requirements

- Python `>= 3.12`
- Poetry (recommended)

Dependencies are managed in `pyproject.toml`.

## Installation

### Option A (recommended): Poetry

```bash
poetry install
```

Optional: keep the virtual environment in the project directory.

```bash
poetry config virtualenvs.in-project true
poetry install
```

### Option B: pip editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## How to Run

After installation, invoke the CLI as:

```bash
poetry run truth-tracking --help
```

If installed via `pip -e .`, you can also run:

```bash
truth-tracking --help
```

## Quick Start

Run one benchmark configuration:

```bash
poetry run truth-tracking simulate --n 6 --x 1 --p 0.6 --q 0.2 --q-mode fixed --nb-runs 200 --max-steps 5000
```

Run one single trajectory:

```bash
poetry run truth-tracking single --n 6 --x 1 --p 0.6 --q 0.2 --q-mode fixed --max-steps 2000 --seed 42
```

Create a `(p, q)` batch CSV:

```bash
poetry run truth-tracking batchpq --n 6 --nb-runs 1000 --max-steps 100000 --stop-on-absorption
```

Plot all metrics from that CSV:

```bash
poetry run truth-tracking batchpq-plot batchpq_YYYYMMDD_HHMMSS_...csv --plot-mode heatmap
```

## Command Reference

### `simulate`
Runs many independent simulations at one parameter point and prints aggregate metrics.

Example:

```bash
poetry run truth-tracking simulate --n 4 --x 1 --p 0.5 --q 0.2 --q-mode fixed --nb-runs 100
```

### `single`
Runs exactly one simulation and prints run-level summary metrics.

Example:

```bash
poetry run truth-tracking single --n 4 --p 0.5 --q 0.2 --q-mode fixed --max-steps 1000
```

### `trace`
Runs one simulation with full trace capture (formulas, beliefs, ranks), with optional CSV and plot output.

Examples:

```bash
poetry run truth-tracking trace --n 6 --max-steps 30 --print-steps --print-ranks
poetry run truth-tracking trace --n 6 --max-steps 30 --out-csv trace.csv --plot-output
```

### `batchpq`
Sweeps a grid of `(p, q)` values (`q < p`) and writes one CSV row per pair.

Example:

```bash
poetry run truth-tracking batchpq --n 6 --nb-runs 1000 --max-steps 100000 --fast
```

### `batchpq-plot`
Reads a `batchpq` CSV and writes one plot per metric column.

Example:

```bash
poetry run truth-tracking batchpq-plot batchpq_...csv --plot-mode contour2d
```

### `batchpq-plotdiff`
Plots metric vs `p` for selected fixed values of `(p - q)`.

Example:

```bash
poetry run truth-tracking batchpq-plotdiff batchpq_...csv --diffpq 0.05 0.10 0.15 0.20
```

### `batchpmaxstep`
Sweeps `(p, max_steps)` with random `q` per run and writes one CSV row per pair.

Example:

```bash
poetry run truth-tracking batchpmaxstep --n 6 --nb-runs 1000 --bound-max-steps 1000
```

### `batchpmaxstep-plot`
Plots `mean_entrenchment_degree` vs `max_steps` for selected `p` values from a `batchpmaxstep` CSV.

Example:

```bash
poetry run truth-tracking batchpmaxstep-plot batchpmaxstep_...csv --p 0.10 0.20 0.30 --bound-max-steps 500
```

### `frequency` (deprecated)
Legacy command for convergence-frequency curve output. Prefer `simulate` for new workflows.

## Reproducibility Notes

- Use `--seed` for deterministic pseudo-random behavior.
- For multi-run commands, run seeds are derived as `seed + k`.
- `--stop-on-absorption` can reduce runtime and changes trajectory length.
- Runtime grows quickly with `n` because the number of worlds is `2^n`.

## Output Files

The CLI writes outputs in the current working directory by default:

- `batchpq_...csv`
- `batchpmaxstep_...csv`
- `..._batchpq-plot*.png`
- `..._batchpmaxstep-plot*.png`
- `trace_plot_...png` (if `trace --plot-output`)

An `example-results-plots/` folder is included with sample CSV/PNG outputs.

## Build / Compile

To build distributable artifacts (wheel + source distribution):

```bash
poetry build
```

Artifacts are generated in `dist/`.

## Project Structure

- `truth_tracking/cli.py`: Typer CLI entry point.
- `truth_tracking/run_simulation.py`: single-run simulation engine.
- `truth_tracking/benchmarks.py`: multi-run aggregation.
- `truth_tracking/formula_generation.py`: formula sampling logic.
- `truth_tracking/ranking.py`: ranking function and improvement operator.
- `truth_tracking/plotting.py`: plotting utilities.

## Reviewer Checklist (Suggested)

1. Install dependencies and run `truth-tracking --help`.
2. Run one `single` and one `simulate` command with a fixed seed.
3. Run a small `batchpq --fast`, then `batchpq-plot`.
4. Inspect generated CSV schema and PNG outputs.
