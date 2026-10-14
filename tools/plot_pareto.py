#!/usr/bin/env python3
"""Plot Pareto-style curves: Time/Cost vs Performance across LLMZero runs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def load_timing_log(run_path: Path) -> dict:
    with open(run_path / "timing_log.json") as f:
        return json.load(f)


def load_token_usage(run_path: Path) -> dict:
    with open(run_path / "token_usage.json") as f:
        return json.load(f)


def load_node_metrics(run_path: Path, node_id: int) -> dict | None:
    p = run_path / f"node_{node_id}" / "states" / "best_validation_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def infer_metrics(run_path: Path, timing_log: dict) -> tuple[str, str]:
    """Infer val and test metric names from the first node with valid metrics."""
    for node_info in timing_log["nodes"]:
        metrics = load_node_metrics(run_path, node_info["node_id"])
        if metrics and metrics.get("selected_validation_metric_name"):
            val_metric = metrics["selected_validation_metric_name"]
            test_metric = val_metric.replace("__dev", "")
            return val_metric, test_metric
    raise ValueError(f"Could not infer metric names from any node in {run_path}")


def get_test_score(metrics: dict | None, val_metric: str, test_metric: str) -> float | None:
    if metrics is None or metrics.get("best_validation_metrics") is None:
        return None
    bvm = metrics["best_validation_metrics"]
    if test_metric in bvm:
        return bvm[test_metric]
    return None


def get_val_score(metrics: dict | None) -> float | None:
    if metrics is None:
        return None
    return metrics.get("best_validation_score")


def compute_curves(
    run_path: Path,
    val_metric: str | None,
    test_metric: str | None,
    api_input_cost: float,
    api_output_cost: float,
    gpu_cost_per_hour: float,
) -> dict:
    """Compute accumulated time, API cost, total cost, and best-so-far test performance."""
    timing_log = load_timing_log(run_path)
    token_usage = load_token_usage(run_path)

    if val_metric is None or test_metric is None:
        val_metric, test_metric = infer_metrics(run_path, timing_log)

    nodes = sorted(timing_log["nodes"], key=lambda n: n["node_id"])  # chronological MCTS order
    num_nodes = len(nodes)

    # Compute per-node API cost (averaged)
    total_input_tokens = token_usage["total"]["total_input_tokens"]
    total_output_tokens = token_usage["total"]["total_output_tokens"]
    total_api_cost = (total_input_tokens / 1e6) * api_input_cost + (total_output_tokens / 1e6) * api_output_cost
    per_node_api_cost = total_api_cost / num_nodes

    acc_times = []
    acc_api_costs = []
    acc_total_costs = []
    best_test_scores = []

    acc_time = 0.0
    acc_api = 0.0
    best_val = -float("inf")
    best_test = None

    for i, node_info in enumerate(nodes):
        node_id = node_info["node_id"]
        exec_time = node_info["execution_time_seconds"]

        acc_time += exec_time
        acc_api += per_node_api_cost
        gpu_cost = (acc_time / 3600.0) * gpu_cost_per_hour

        metrics = load_node_metrics(run_path, node_id)
        val_score = get_val_score(metrics)
        test_score = get_test_score(metrics, val_metric, test_metric)

        # Strict > so that ties keep the earliest node's test score
        if val_score is not None and val_score > best_val:
            best_val = val_score
            if test_score is not None:
                best_test = test_score

        acc_times.append(acc_time / 3600.0)  # convert to hours
        acc_api_costs.append(acc_api)
        acc_total_costs.append(acc_api + gpu_cost)
        best_test_scores.append(best_test)

    return {
        "acc_times": acc_times,
        "acc_api_costs": acc_api_costs,
        "acc_total_costs": acc_total_costs,
        "best_test_scores": best_test_scores,
        "val_metric": val_metric,
        "test_metric": test_metric,
    }


def smooth_curve(xs, ys, smoothing_factor=0.5, num_points=300):
    """Generate a smoothed spline curve. smoothing_factor in (0,1]: higher = smoother."""
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    if len(xs) < 4:
        return xs, ys
    # Scale smoothing_factor to a reasonable s value relative to data variance
    y_range = ys.max() - ys.min()
    if y_range == 0:
        return xs, ys
    s = smoothing_factor * len(xs) * (y_range**2)
    spline = UnivariateSpline(xs, ys, k=3, s=s)
    x_smooth = np.linspace(xs[0], xs[-1], num_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def _plot_row(axes, all_curves, run_names, colors, smoothing_factor: float):
    """Plot one row of 3 panels (time/api/total vs performance)."""
    configs = [
        ("acc_times", "Accumulated Time (hours)", "Time vs Performance"),
        ("acc_api_costs", "Accumulated API Cost ($)", "API Cost vs Performance"),
        ("acc_total_costs", "Accumulated Total Cost ($)", "Total Cost (API + GPU) vs Performance"),
    ]

    for ax, (x_key, x_label, title) in zip(axes, configs):
        for i, (curves, name) in enumerate(zip(all_curves, run_names)):
            xs = curves[x_key]
            ys = curves["best_test_scores"]
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not valid:
                continue
            vx, vy = zip(*valid)
            color = colors[i]
            if smoothing_factor > 0:
                sx, sy = smooth_curve(vx, vy, smoothing_factor=smoothing_factor)
                ax.plot(sx, sy, color=color, linewidth=2, label=name)
                ax.scatter(vx, vy, color=color, s=20, zorder=5)
            else:
                ax.plot(vx, vy, marker="o", markersize=4, label=name, color=color, linewidth=2)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Performance (Test Metric)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)


def plot_pareto(
    all_curves: list[dict],
    run_names: list[str],
    output_path: Path,
    api_input_cost: float,
    api_output_cost: float,
    gpu_cost_per_hour: float,
    smoothing_factor: float = 0.3,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(run_names), 10)))

    # Top row: raw (no smoothing)
    _plot_row(axes[0], all_curves, run_names, colors, smoothing_factor=0.0)
    for ax in axes[0]:
        t = ax.get_title()
        ax.set_title(t, fontsize=13)
    axes[0][0].annotate(
        "Raw",
        xy=(0, 0.5),
        xytext=(-0.35, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        va="center",
        rotation=90,
    )

    # Bottom row: smoothed
    _plot_row(axes[1], all_curves, run_names, colors, smoothing_factor=smoothing_factor)
    axes[1][0].annotate(
        "Smoothed",
        xy=(0, 0.5),
        xytext=(-0.35, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        va="center",
        rotation=90,
    )

    # Add info as subtitle text
    val_metric = all_curves[0]["val_metric"]
    test_metric = all_curves[0]["test_metric"]
    info_lines = (
        f"Val Metric: {val_metric}  |  Test Metric: {test_metric}\n"
        f"API Cost: ${api_input_cost}/MTok input, ${api_output_cost}/MTok output  |  "
        f"GPU Cost: ${gpu_cost_per_hour:.2f}/hr"
    )
    fig.text(0.5, -0.01, info_lines, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto curves for LLMZero runs")
    parser.add_argument("--run-paths", nargs="+", required=True, help="Path(s) to run directories")
    parser.add_argument(
        "--run-names", nargs="+", default=None, help="Display names for each run (default: folder name)"
    )
    parser.add_argument("--val-metric", default=None, help="Validation metric name (default: auto-infer)")
    parser.add_argument("--test-metric", default=None, help="Test metric name (default: auto-infer from val metric)")
    parser.add_argument(
        "--api-input-cost", type=float, default=5.0, help="API input cost per MTok in $ (default: 5.0)"
    )
    parser.add_argument(
        "--api-output-cost", type=float, default=25.0, help="API output cost per MTok in $ (default: 25.0)"
    )
    parser.add_argument(
        "--gpu-cost", type=float, default=4 * 21.957642, help="GPU cost per hour in $ (default: 4*21.957642)"
    )
    parser.add_argument("--output", default=None, help="Output file path (default: <first_run>/pareto_curves.png)")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.3,
        metavar="FACTOR",
        help="Smoothing factor for the smoothed row in (0,1]: higher=smoother (default: 0.3)",
    )
    args = parser.parse_args()

    run_paths = [Path(p) for p in args.run_paths]
    run_names = args.run_names or [p.name for p in run_paths]

    if len(run_names) != len(run_paths):
        parser.error("Number of --run-names must match number of --run-paths")

    output_path = Path(args.output) if args.output else run_paths[0] / "pareto_curves.png"

    all_curves = []
    for rp in run_paths:
        curves = compute_curves(
            rp, args.val_metric, args.test_metric, args.api_input_cost, args.api_output_cost, args.gpu_cost
        )
        all_curves.append(curves)

    plot_pareto(
        all_curves, run_names, output_path, args.api_input_cost, args.api_output_cost, args.gpu_cost, args.smooth
    )


if __name__ == "__main__":
    main()
