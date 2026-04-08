"""Publication-style SVG progress plots for epoch metrics."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

logger = logging.getLogger(__name__)

SPLIT_NAMES = {"train", "val", "test"}
PLOT_EXCLUDED_PREFIXES = ("LR/",)
SERIES_COLORS = {
    "train": "#1f77b4",
    "val": "#d62728",
    "test": "#2ca02c",
    "default": "#111827",
}


def save_metric_plots(series: Dict[str, Dict[int, float]], output_dir: Path) -> None:
    """Render publication-style metric plots from scalar history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.svg"):
        existing.unlink()

    for base_tag, series_by_split in iter_plot_groups(series):
        if series_by_split:
            save_metric_plot(output_dir / f"{slugify(base_tag)}.svg", base_tag, series_by_split)


def iter_plot_groups(
    series: Dict[str, Dict[int, float]],
) -> Iterable[tuple[str, Dict[str, Dict[int, float]]]]:
    """Yield metric groups keyed by the base tag with optional split suffixes."""
    grouped: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(dict)
    for tag, points in series.items():
        if tag.startswith(PLOT_EXCLUDED_PREFIXES):
            continue
        base_tag, split = split_metric_tag(tag)
        grouped[base_tag][split or "default"] = {int(epoch): float(value) for epoch, value in points.items()}

    for base_tag in sorted(grouped):
        yield base_tag, grouped[base_tag]


def save_metric_plot(
    path: Path,
    base_tag: str,
    series_by_split: Dict[str, Dict[int, float]],
) -> None:
    """Render a single metric group as an SVG line plot."""
    y_label, scale = axis_spec(base_tag)
    plotted = 0

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=300, constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for split_name in sorted(series_by_split):
            points = series_by_split[split_name]
            if not points:
                continue

            epochs = sorted(points)
            values = [points[epoch] * scale for epoch in epochs]
            sparse = len(epochs) <= 20 or max(epochs) - min(epochs) > max(len(epochs) * 2, 1)
            color = SERIES_COLORS.get(split_name, SERIES_COLORS["default"])
            label = split_name if split_name != "default" else humanize_metric_name(base_tag)

            ax.plot(
                epochs,
                values,
                color=color,
                linewidth=1.8,
                marker="o" if sparse else None,
                markersize=3.2 if sparse else 0.0,
                markerfacecolor="white" if sparse else color,
                markeredgewidth=0.9 if sparse else 0.0,
                label=label,
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return

        ax.set_xlabel("Epoch (index)")
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", width=0.8, length=3)

        if plotted > 1 or "default" not in series_by_split:
            legend = ax.legend(frameon=False, loc="best", handlelength=2.2)
            for line in legend.get_lines():
                line.set_linewidth(1.8)

        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)


def split_metric_tag(tag: str) -> tuple[str, Optional[str]]:
    """Split a metric tag into its base name and split suffix if present."""
    parts = tag.split("/")
    if len(parts) > 1 and parts[-1] in SPLIT_NAMES:
        return "/".join(parts[:-1]), parts[-1]
    return tag, None


def axis_spec(metric_key: str) -> tuple[str, float]:
    """Return axis label and scale factor for a metric tag."""
    normalized = metric_key.lower()

    if "psnr" in normalized:
        return "PSNR (dB)", 1.0
    if "ssim" in normalized:
        return "SSIM (index)", 1.0
    if "accuracy" in normalized:
        return "Accuracy (%)", 100.0
    if "precision" in normalized:
        return "Precision (%)", 100.0
    if "recall" in normalized:
        return "Recall (%)", 100.0
    if re.search(r"(^|[/_])f1($|[/_])", normalized):
        return "F1 (%)", 100.0
    if "sensitivity" in normalized:
        return "Sensitivity (%)", 100.0
    if "specificity" in normalized:
        return "Specificity (%)", 100.0
    if "auc" in normalized:
        return "AUC (index)", 1.0
    if "probe_acc" in normalized:
        return "Probe accuracy (%)", 100.0
    if "loss_d" in normalized:
        return "Discriminator loss (a.u.)", 1.0
    if "g_adv" in normalized or "adversarial" in normalized:
        return "Generator adversarial loss (a.u.)", 1.0
    if "perceptual" in normalized:
        return "Perceptual loss (a.u.)", 1.0
    if normalized == "kl" or normalized.endswith("/kl") or normalized.endswith("_kl"):
        return "KL divergence (a.u.)", 1.0
    if "recon" in normalized:
        return "Reconstruction loss (a.u.)", 1.0
    if "loss" in normalized:
        return "Loss (a.u.)", 1.0
    return f"{humanize_metric_name(metric_key)} (a.u.)", 1.0


def format_tick(value: float, _: int) -> str:
    """Compact numeric tick formatter without scientific notation."""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def slugify(value: str) -> str:
    """Convert a metric tag into a filesystem-friendly stem."""
    stem = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return stem or "metric"


def humanize_metric_name(metric_key: str) -> str:
    """Convert an internal metric tag to a human-readable label."""
    text = metric_key.replace("/", " ").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Metric"
