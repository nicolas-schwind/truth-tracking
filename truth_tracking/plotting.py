"""
plotting.py

Plot convergence frequency curves.
"""

from typing import Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_frequency_curve(
    freq: np.ndarray,
    title: str = "Convergence Frequency Curve",
    final_avg: float | None = None,
    tail_avg: float | None = None,
    tail_curve: np.ndarray | None = None,
    tail_window: int | None = None,
    legend_title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    fig = plt.figure()
    plt.plot(freq, label="running avg frequency")
    if tail_curve is not None:
        if tail_window is None:
            tail_label = "running tail frequency"
        else:
            tail_label = f"running tail frequency (window={tail_window})"
        plt.plot(tail_curve, label=tail_label)
    if final_avg is not None:
        plt.axhline(final_avg, color="green", linestyle="--", linewidth=1.0, label="final avg")
    if tail_avg is not None:
        plt.axhline(tail_avg, color="orange", linestyle="--", linewidth=1.0, label="mean tail")
    plt.xlabel("time step t")
    plt.ylabel("avg frequency of Bel(rf_t) == {w*} up to time t")
    plt.title(title)
    plt.grid(True)
    if legend_title is None:
        plt.legend()
    else:
        plt.legend(title=legend_title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_trace_ranks(
    ranks_over_time: Sequence[Sequence[int]],
    true_world: int,
    title: str = "Ranking Evolution (Trace)",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot a heatmap of ranks over time.
    ranks_over_time is a list of rank vectors (one per time step).
    """
    data = np.array(ranks_over_time, dtype=np.float64)
    # shape: (time, world)
    fig, ax = plt.subplots()
    im = ax.imshow(data.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("time step t")
    ax.set_ylabel("world index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("rank")

    if 0 <= true_world < data.shape[1]:
        ax.axhline(true_world, color="white", linewidth=1.0, alpha=0.8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_trace_preorder(
    ranks_over_time: Sequence[Sequence[Tuple[str, int]]],
    formulas_over_time: Sequence[Sequence[str]],
    true_world_label: str,
    title: str = "Total Preorder per Step",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Render each step as a total preorder text line with the true world highlighted.
    """
    lines = []
    for t, (ranks, formula) in enumerate(zip(ranks_over_time, formulas_over_time)):
        groups: dict[int, list[str]] = {}
        for label, r in ranks:
            groups.setdefault(r, []).append(label)

        preorder_parts = []
        for r in sorted(groups.keys()):
            worlds = groups[r]
            world_strs = []
            for w in worlds:
                if w == true_world_label:
                    world_strs.append(f"*{w}")
                else:
                    world_strs.append(w)
            preorder_parts.append(f"{r}:{'/'.join(world_strs)}")

        formula_str = "{" + ", ".join(formula) + "}" if formula else "{}"
        lines.append(f"t={t - 1} | F={formula_str} | preorder={' < '.join(preorder_parts)}")

    fig, ax = plt.subplots(figsize=(12, max(4, 0.25 * len(lines))))
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_surface_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(p_vals, q_vals, z_vals, cmap="viridis", linewidth=0.2, antialiased=True)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_scatter_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p_vals, q_vals, z_vals, c=z_vals, cmap="viridis", s=12, alpha=0.9)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_wireframe_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        p_vals,
        q_vals,
        z_vals,
        linewidth=0.4,
        antialiased=True,
        color="none",
        edgecolor="gray",
    )
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_contour_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.tricontourf(p_vals, q_vals, z_vals, levels=20, cmap="viridis")
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _interpolate_grid(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    method: str,
    grid_size: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_grid = np.linspace(p_vals.min(), p_vals.max(), grid_size)
    q_grid = np.linspace(q_vals.min(), q_vals.max(), grid_size)
    P, Q = np.meshgrid(p_grid, q_grid)
    tri = mtri.Triangulation(p_vals, q_vals)
    if method == "linear":
        interpolator = mtri.LinearTriInterpolator(tri, z_vals)
    elif method == "cubic":
        interpolator = mtri.CubicTriInterpolator(tri, z_vals)
    else:
        raise ValueError("Unknown interpolation method.")
    Z = interpolator(P, Q)
    return P, Q, Z


def _polyfit_grid(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    degree: int,
    grid_size: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    A = np.column_stack([(p_vals ** i) * (q_vals ** j) for i, j in terms])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
    p_grid = np.linspace(p_vals.min(), p_vals.max(), grid_size)
    q_grid = np.linspace(q_vals.min(), q_vals.max(), grid_size)
    P, Q = np.meshgrid(p_grid, q_grid)
    Z = np.zeros_like(P, dtype=float)
    for (i, j), c in zip(terms, coeffs):
        Z += c * (P ** i) * (Q ** j)
    return P, Q, Z


def plot_surface_from_grid(
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
    alpha: float = 1.0,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, Q, Z, cmap="viridis", linewidth=0.2, alpha=alpha)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_wireframe_from_grid(
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(P, Q, Z, color="gray", linewidth=0.4)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_contour_from_grid(
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.contourf(P, Q, Z, levels=20, cmap="viridis")
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_scatter_from_grid(
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P.ravel(), Q.ravel(), Z.ravel(), c=Z.ravel(), cmap="viridis", s=8, alpha=0.9)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_surface_overlay(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p_vals, q_vals, z_vals, c="black", s=8, alpha=0.6)
    ax.plot_surface(P, Q, Z, cmap="viridis", linewidth=0.2, alpha=0.7)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_heatmap_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots()
    sc = ax.scatter(p_vals, q_vals, c=z_vals, cmap="viridis", s=20)
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(z_label)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_contour2d_from_points(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    z_vals: np.ndarray,
    *,
    title: str,
    z_label: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots()
    contour = ax.tricontourf(p_vals, q_vals, z_vals, levels=20, cmap="viridis")
    ax.set_xlabel("p")
    ax.set_ylabel("q", rotation=0, labelpad=10)
    ax.set_title(title)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(z_label)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_diffpq_line(
    p_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    title: str,
    y_label: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(p_vals, y_vals, marker="o", markersize=3, linewidth=1.0)
    ax.set_xlabel("p")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
