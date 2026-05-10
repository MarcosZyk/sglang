from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = '../result_128G_9'

METHOD_ORDER = [
    "result-enable-artesia-lru",
    "result-enable-artesia-mru",
    "result-disable-artesia-lru",
    #"result-disable-artesia-mru",
]

METHOD_LABELS = {
    "result-enable-artesia-lru": "enable-artesia-lru",
    "result-enable-artesia-mru": "enable-artesia-mru",
    "result-disable-artesia-lru": "disable-artesia-lru",
    #"result-disable-artesia-mru": "disable-artesia-mru",
}

METHOD_COLORS = {
    "result-enable-artesia-lru": "#1f77b4",
    "result-enable-artesia-mru": "#ff7f0e",
    "result-disable-artesia-lru": "#2ca02c",
    #"result-disable-artesia-mru": "#d62728",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_summary_root = script_dir.parent / "result_128G_11" / "summary"

    parser = argparse.ArgumentParser(
        description="Plot CDF curves from per-method summary CSV files.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=default_summary_root,
        help="Directory containing per-method summary CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to write PNG plots. Defaults to <summary-root>/plots.",
    )
    return parser.parse_args()


def load_summary_rows(summary_csv: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prefill_total_time = float(row["prefill_total_time"])
            decode_total_time = float(row["decode_total_time"])
            rows.append(
                {
                    "agent_request_id": row["agent_request_id"],
                    "prefill_total_time": prefill_total_time,
                    "decode_total_time": decode_total_time,
                    "end_to_end_time": prefill_total_time + decode_total_time,
                    "local_cache_ratio": float(row["local_cache_ratio"]),
                    "global_cache_ratio": float(row["global_cache_ratio"]),
                }
            )
    return rows


def empirical_cdf(values: list[float]) -> tuple[list[float], list[float]]:
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return [], []
    cumulative = [(index + 1) / n for index in range(n)]
    return sorted_values, cumulative


def interpolated_cdf_curve(
    values: list[float],
    interpolation_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = empirical_cdf(values)
    if not x_values:
        return np.array([]), np.array([])

    unique_x: list[float] = []
    unique_y: list[float] = []
    for x_value, y_value in zip(x_values, y_values):
        if unique_x and x_value == unique_x[-1]:
            unique_y[-1] = y_value
        else:
            unique_x.append(x_value)
            unique_y.append(y_value)

    if len(unique_x) == 1:
        return np.array(unique_x), np.array(unique_y)

    dense_x = np.linspace(unique_x[0], unique_x[-1], interpolation_points)
    dense_y = np.interp(dense_x, unique_x, unique_y)
    return dense_x, dense_y


def method_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem in METHOD_ORDER:
        return (METHOD_ORDER.index(stem), stem)
    return (len(METHOD_ORDER), stem)


def plot_values(ax, method_name: str, values: list[float]) -> bool:
    x_values, y_values = interpolated_cdf_curve(values)
    if len(x_values) == 0:
        return False
    ax.plot(
        x_values,
        y_values,
        label=METHOD_LABELS.get(method_name, method_name),
        color=METHOD_COLORS.get(method_name),
        linewidth=2.0,
    )
    return True


def style_metric_axis(ax, metric_title: str, x_label: str) -> None:
    ax.set_title(metric_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_metric(
    metric_key: str,
    metric_title: str,
    x_label: str,
    summary_files: list[Path],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for summary_csv in summary_files:
        method_name = summary_csv.stem
        rows = load_summary_rows(summary_csv)
        values = [float(row[metric_key]) for row in rows]
        plot_values(ax, method_name, values)

    style_metric_axis(ax, metric_title, x_label)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_decode_time(
    metric_key: str,
    metric_title: str,
    x_label: str,
    summary_files: list[Path],
    output_path: Path,
) -> None:
    rows_by_method = [
        (summary_csv.stem, load_summary_rows(summary_csv))
        for summary_csv in summary_files
    ]
    decode_times = sorted(
        {
            float(row["decode_total_time"])
            for _, rows in rows_by_method
            for row in rows
        }
    )
    if not decode_times:
        return

    num_plots = len(decode_times)
    ncols = 1 if num_plots == 1 else min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    fig_width = 8 if num_plots == 1 else 6 * ncols
    fig_height = 5 if num_plots == 1 else 4.5 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.ravel()

    for ax, decode_time in zip(flat_axes, decode_times):
        for method_name, rows in rows_by_method:
            values = [
                float(row[metric_key])
                for row in rows
                if float(row["decode_total_time"]) == decode_time
            ]
            plot_values(ax, method_name, values)
        style_metric_axis(
            ax,
            f"{metric_title}\nDecode Time = {decode_time:.6g}s",
            x_label,
        )
        ax.legend()

    for ax in flat_axes[num_plots:]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_root = args.summary_root.resolve()
    output_root = (args.output_root or (summary_root / "plots")).resolve()

    if not summary_root.exists():
        raise FileNotFoundError(f"summary root does not exist: {summary_root}")
    if not summary_root.is_dir():
        raise NotADirectoryError(f"summary root is not a directory: {summary_root}")

    summary_files = sorted(summary_root.glob("*.csv"), key=method_sort_key)
    if not summary_files:
        raise FileNotFoundError(f"no summary csv files found under {summary_root}")

    plot_metric_by_decode_time(
        metric_key="prefill_total_time",
        metric_title="CDF of Total Prefill Time",
        x_label="Prefill Total Time (s)",
        summary_files=summary_files,
        output_path=output_root / "prefill_total_time_cdf.png",
    )
    plot_metric_by_decode_time(
        metric_key="end_to_end_time",
        metric_title="CDF of End-to-End Time",
        x_label="End-to-End Time (s)",
        summary_files=summary_files,
        output_path=output_root / "end_to_end_time_cdf.png",
    )
    plot_metric(
        metric_key="local_cache_ratio",
        metric_title="CDF of Local Cache Ratio",
        x_label="Local Cache Ratio",
        summary_files=summary_files,
        output_path=output_root / "local_cache_ratio_cdf.png",
    )
    plot_metric(
        metric_key="global_cache_ratio",
        metric_title="CDF of Global Cache Ratio",
        x_label="Global Cache Ratio",
        summary_files=summary_files,
        output_path=output_root / "global_cache_ratio_cdf.png",
    )

    print(f"processed {len(summary_files)} summary files from {summary_root}")
    print(f"wrote plots to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
