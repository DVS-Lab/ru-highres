"""Publication-oriented plots for dataset-resolution summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

AXES = {
    "lr": ("voxel_size_lr_mm", "Left-right voxel size (mm)"),
    "ap": ("voxel_size_ap_mm", "Anterior-posterior voxel size (mm)"),
    "is": ("voxel_size_is_mm", "Inferior-superior voxel size (mm)"),
}


def write_plots(
    *,
    summary_csv: Path,
    figures_dir: Path = Path("figures"),
    bin_width_mm: float = 0.1,
    main_min_mm: float = 0.5,
    main_max_mm: float = 6.0,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    outputs: list[Path] = []
    if summary.empty:
        return outputs

    for axis_name, (column, label) in AXES.items():
        outputs.extend(
            _histogram(
                summary[column].dropna().to_numpy(dtype=float),
                xlabel=label,
                title=f"OpenNeuro BOLD voxel size: {axis_name.upper()} axis",
                output_stem=figures_dir / f"hist_voxel_size_{axis_name}",
                bin_width_mm=bin_width_mm,
                main_min_mm=main_min_mm,
                main_max_mm=main_max_mm,
            )
        )
    outputs.extend(
        _histogram(
            summary[[column for column, _label in AXES.values()]]
            .to_numpy(dtype=float)
            .ravel(),
            xlabel="Voxel size (mm)",
            title="OpenNeuro BOLD voxel sizes across anatomical dimensions",
            output_stem=figures_dir / "hist_voxel_size_all_dimensions",
            bin_width_mm=bin_width_mm,
            main_min_mm=main_min_mm,
            main_max_mm=main_max_mm,
        )
    )
    outputs.extend(_common_triplet_plot(summary, figures_dir / "common_resolution_triplets"))
    outputs.extend(_resolutions_per_dataset_plot(summary, figures_dir / "resolutions_per_dataset"))
    return outputs


def _histogram(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str,
    output_stem: Path,
    bin_width_mm: float,
    main_min_mm: float,
    main_max_mm: float,
) -> list[Path]:
    values = values[np.isfinite(values)]
    in_range = values[(values >= main_min_mm) & (values <= main_max_mm)]
    bins = np.arange(main_min_mm, main_max_mm + bin_width_mm, bin_width_mm)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.hist(in_range, bins=bins.tolist(), color="#2d6cdf", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dataset-resolution observations")
    ax.grid(axis="y", alpha=0.25)

    stats = _summary_stats(values)
    outside = int(values.size - in_range.size)
    annotation = f"N={values.size}\nMedian={stats['median']:.3g} mm\nIQR={stats['iqr']:.3g} mm"
    if outside:
        annotation += f"\nOutside {main_min_mm:g}-{main_max_mm:g} mm: {outside}"
    ax.text(
        0.98,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0"},
    )

    return _save_figure(fig, output_stem)


def _common_triplet_plot(summary: pd.DataFrame, output_stem: Path) -> list[Path]:
    triplet_labels = (
        summary["voxel_size_lr_mm"].astype(str)
        + " x "
        + summary["voxel_size_ap_mm"].astype(str)
        + " x "
        + summary["voxel_size_is_mm"].astype(str)
    )
    counts = triplet_labels.value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    counts.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_xlabel("Dataset-resolution observations")
    ax.set_ylabel("Resolution triplet (LR x AP x IS mm)")
    ax.set_title("Most common OpenNeuro BOLD resolution triplets")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, output_stem)


def _resolutions_per_dataset_plot(summary: pd.DataFrame, output_stem: Path) -> list[Path]:
    counts = summary.groupby("dataset_accession").size()
    max_count = int(counts.max()) if not counts.empty else 1
    bins = np.arange(0.5, max_count + 1.5, 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.hist(counts.to_numpy(), bins=bins.tolist(), color="#e76f51", edgecolor="white")
    ax.set_xlabel("Unique resolution triplets per dataset")
    ax.set_ylabel("Datasets")
    ax.set_title("Number of raw BOLD resolutions per OpenNeuro dataset")
    ax.set_xticks(np.arange(1, max_count + 1))
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_stem)


def _save_figure(fig: Figure, output_stem: Path) -> list[Path]:
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in outputs:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return outputs


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "iqr": float("nan")}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "iqr": float(q3 - q1)}
