"""Tables and figures for E1, E2 and E3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["LF", "OGM", "SI", "ICS", "SOCP", "VGMC"]
MOLECULES = ["LiH", "BeH2", "H2O", "NH3"]
LABELS = {
    "H2": r"H$_2$",
    "LiH": "LiH",
    "BeH2": r"BeH$_2$",
    "H2O": r"H$_2$O",
    "NH3": r"NH$_3$",
}
STYLES = {
    "VGMC": ("#0072B2", "-", "D"),
    "Frozen": ("#D55E00", "--", "o"),
    "Diag": ("#009E73", "-.", "s"),
    "Frozen+Diag": ("#CC79A7", ":", "^"),
    "Random": ("#666666", "--", "o"),
}


def style() -> None:
    """Set the shared figure typography and vector export settings."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """Export a figure to PDF and PNG."""
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", bbox_inches="tight")


def table_e1(frame: pd.DataFrame, covariance: str) -> pd.DataFrame:
    """Arrange E1 variance coefficients in manuscript method order."""
    subset = frame[frame.covariance == covariance]
    table = subset.pivot(index="molecule", columns="method", values="V_Ha2")
    molecules = [m for m in ["H2", *MOLECULES] if m in table.index]
    return table.reindex(
        index=molecules, columns=[m for m in METHODS if m in table.columns]
    ).rename_axis(columns=None)


def plot_e2(
    frame: pd.DataFrame,
    *,
    covariance: str = "exact",
    zoom: bool = True,
    molecules: list[str] | None = None,
) -> plt.Figure:
    """Plot empirical energy RMSE with percentile bootstrap intervals."""
    style()
    molecules = molecules or [m for m in MOLECULES if m in set(frame.molecule)]
    figure, axes = plt.subplots(
        1, len(molecules), figsize=(2.7 * len(molecules), 2.7), squeeze=False
    )
    colors = {"ICS": "#D55E00", "SOCP": "#0072B2", "VGMC": "#009E73"}
    for ax, molecule in zip(axes[0], molecules, strict=True):
        subset = frame[
            (frame.molecule == molecule) & (frame.covariance == covariance)
        ]
        thresholds = (
            subset.accuracy_threshold.unique()
            if "accuracy_threshold" in subset
            else np.array([0.0016])
        )
        if len(thresholds) != 1:
            raise ValueError(
                "Each panel requires one RMSE reference threshold"
            )
        threshold = float(thresholds[0]) * 1000
        for method, color in colors.items():
            rows = subset[subset.method == method].sort_values("budget")
            if rows.empty:
                continue
            ci = np.array(rows.rmse_ci95.tolist()) * 1000
            ax.plot(
                rows.budget,
                rows.rmse * 1000,
                marker="o",
                ms=3,
                lw=1.2,
                color=color,
                label=method,
            )
            ax.fill_between(
                rows.budget.to_numpy(),
                ci[:, 0],
                ci[:, 1],
                color=color,
                alpha=0.12,
                lw=0,
            )
        ax.axhline(
            threshold,
            color="#333333",
            ls=":",
            lw=1.2,
            label="Chemical accuracy",
        )
        ax.set_xscale("log")
        if zoom:
            ax.set_ylim(0.9, 3.0)
            ax.axhspan(0.9, threshold, color="#eeeeee", zorder=-2)
            relevant = subset[
                (subset.rmse * 1000 <= 3.5) & (subset.rmse * 1000 >= 0.7)
            ]
            if not relevant.empty:
                ax.set_xlim(
                    relevant.budget.min() / 1.2, relevant.budget.max() * 1.2
                )
        else:
            ax.set_yscale("log")
        ax.set_title(LABELS[molecule])
        ax.set_xlabel("Shots")
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Energy RMSE (mHa)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def summarize_e3(frame: pd.DataFrame) -> pd.DataFrame:
    """Collect the final recorded point for each molecule, method and seed."""
    return (
        frame.sort_values("added")
        .groupby(["molecule", "method", "seed"], dropna=False)
        .tail(1)
        .sort_values(["molecule", "method", "seed"])
    )


def plot_e3(
    frame: pd.DataFrame, *, molecules: list[str] | None = None
) -> plt.Figure:
    """Plot variance reduction and active fractions against added-group
    count.
    """
    style()
    molecules = molecules or [m for m in MOLECULES if m in set(frame.molecule)]
    figure, axes = plt.subplots(
        2,
        len(molecules),
        figsize=(2.7 * len(molecules), 4.7),
        squeeze=False,
        sharex="col",
    )
    for column, molecule in enumerate(molecules):
        subset = frame[frame.molecule == molecule]
        for method, (color, linestyle, marker) in STYLES.items():
            selected = subset[subset.method == method].sort_values("added")
            if selected.empty:
                continue
            for row, metric in enumerate(("gain_pct", "active_pct")):
                ax = axes[row, column]
                if method == "Random":
                    summary = selected.groupby("added")[metric].agg(
                        ["median", "min", "max"]
                    )
                    ax.plot(
                        summary.index,
                        summary["median"],
                        color=color,
                        ls=linestyle,
                        marker=marker,
                        ms=3,
                        lw=1.1,
                        label=method,
                    )
                    ax.fill_between(
                        summary.index.to_numpy(),
                        summary["min"].to_numpy(),
                        summary["max"].to_numpy(),
                        color=color,
                        alpha=0.13,
                        lw=0,
                    )
                else:
                    ax.plot(
                        selected.added,
                        selected[metric],
                        color=color,
                        ls=linestyle,
                        marker=marker,
                        markevery=max(1, len(selected) // 8),
                        ms=3,
                        lw=1.1,
                        label=method,
                    )
                ax.grid(alpha=0.15)
        axes[0, column].set_title(LABELS[molecule])
        axes[1, column].set_xlabel("Added groups")
        axes[1, column].set_ylim(-2, 102)
    axes[0, 0].set_ylabel("Variance reduction (%)")
    axes[1, 0].set_ylabel("Active added groups (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure
