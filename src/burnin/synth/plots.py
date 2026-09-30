"""Figures for the synthetic dataset: healthy vs each defect trajectory, and lot-to-lot variation.

Static matplotlib figures. Colour does one job each: slot-1 blue is the lot reference (median and
1-99 % band of healthy parts), slot-2 orange is "defective example", neutral ink is an individual
healthy part. Text never wears a series colour. Every figure says SYNTHETIC DATA in its title.

Documentation figures only: this module reads the labels file to colour the defective parts and
is not on the screening path (CONVENTIONS.md rule 3). Reading hours are taken from the data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

from burnin.schema import pivot_hours

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# Built-in types in display order; any other type found in the labels gets its own panel after.
BUILTIN_ORDER = (
    "baseline_outlier",
    "accelerating_drift",
    "excess_linear_drift",
    "step_change",
    "catastrophic",
    "multivariate",
)
TITLES = {"catastrophic": "catastrophic (fails static limit)"}


def _title(defect_type: str) -> str:
    return TITLES.get(defect_type, defect_type.replace("_", " "))


def _load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    meas = pd.read_csv(data_dir / "measurements.csv")
    labels = pd.read_csv(data_dir / "labels.csv", keep_default_na=False)
    with open(data_dir / "limits.yaml", encoding="utf-8") as fh:
        limits = yaml.safe_load(fh)["part_numbers"]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    return meas, labels, limits, manifest


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _ratio_ticks(axis, lim: tuple[float, float]) -> None:
    """Clean ×-ratio ticks on a log axis, only inside the current limits (no autoscale change)."""
    candidates = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 20, 30]
    axis.set_ticks([t for t in candidates if lim[0] <= t <= lim[1]])
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _ratio_to_lot_median(meas: pd.DataFrame, labels: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Each part's trajectory divided by its lot's healthy median at the same hour.

    Returns one row per part with hour columns plus the label columns.
    """
    wide = pivot_hours(meas.loc[meas["parameter"] == parameter]).reset_index()
    hours = [c for c in wide.columns if isinstance(c, int)]
    merged = wide.merge(labels, on=["lot_id", "part_id"], how="left")
    med = (
        merged.loc[~merged["is_defective"]]
        .groupby("lot_id")[hours]
        .median()
        .reindex(merged["lot_id"])
        .to_numpy()
    )
    merged[hours] = merged[hours].to_numpy() / med
    return merged


def plot_example_trajectories(
    data_dir: str | Path,
    out_png: str | Path,
    *,
    parameter: str = "IDDQ",
    n_examples: int = 3,
    n_healthy: int = 40,
) -> tuple[Path, list[str]]:
    """Small multiples: healthy population vs example parts of every defect type in the labels.

    Returns the output path and the list of defect types that got a panel.
    """
    data_dir, out_png = Path(data_dir), Path(out_png)
    meas, labels, _limits, manifest = _load(data_dir)
    df = _ratio_to_lot_median(meas, labels, parameter)
    hours = [c for c in df.columns if isinstance(c, int)]
    healthy = df.loc[~df["is_defective"]]
    band = healthy[hours].quantile([0.01, 0.5, 0.99])
    counts = labels.loc[labels["is_defective"], "defect_type"].value_counts()
    types = [t for t in BUILTIN_ORDER if counts.get(t, 0) > 0]
    types += sorted(t for t in counts.index if t not in BUILTIN_ORDER)

    ncols = 3
    nrows = max(1, math.ceil(len(types) / ncols))
    height = 4.0 * nrows + 0.6  # inches; the header (title + legend) takes a fixed 0.62 in
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, height), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(types) :]:
        ax.set_axis_off()
    for ax, dtype in zip(axes.flat, types, strict=False):
        _style(ax)
        if dtype == "multivariate":
            _multivariate_panel(ax, meas, labels, parameter, int(counts[dtype]), hours[0])
            continue
        sample = healthy.sample(min(n_healthy, len(healthy)), random_state=0)
        ax.plot(hours, sample[hours].to_numpy().T, color=MUTED, linewidth=0.6, alpha=0.35)
        ax.fill_between(hours, band.loc[0.01], band.loc[0.99], color=BLUE, alpha=0.10, lw=0)
        ax.plot(hours, band.loc[0.5], color=BLUE, linewidth=2)

        ex = df.loc[
            (df["defect_type"] == dtype) & df["affected_parameters"].str.contains(parameter)
        ].sort_values(["onset_hour", "lot_id", "part_id"])
        if dtype == "step_change":  # one example per onset hour, early and late
            ex = ex.groupby("onset_hour", sort=True).head(1)
        ex = ex.dropna(subset=hours).head(n_examples)
        if ex.empty:
            ax.text(
                0.5,
                0.9,
                f"no complete example carries this defect on {parameter}",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
                color=INK2,
            )
        for _, row in ex.iterrows():
            y = row[hours].to_numpy(dtype=float)
            ax.plot(
                hours,
                y,
                color=ORANGE,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            if dtype == "step_change" and int(row["onset_hour"]) in hours:
                k = hours.index(int(row["onset_hour"]))
                ax.annotate(
                    f"step at {hours[k]} h",
                    (hours[k], y[k]),
                    xytext=(6, -12),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=INK2,
                )
        ax.set_yscale("log")
        _ratio_ticks(ax.yaxis, ax.get_ylim())
        ax.set_xticks(hours)
        ax.set_xlabel("burn-in hour")
        ax.set_ylabel(f"{parameter} / lot median")
        ax.set_title(
            f"{_title(dtype)}  (n = {counts.get(dtype, 0)})", loc="left", color=INK, fontsize=11
        )

    handles = [
        Line2D([], [], color=MUTED, linewidth=1, label="healthy parts (sample)"),
        Patch(facecolor=BLUE, alpha=0.10, label="healthy 1-99 % band"),
        Line2D([], [], color=BLUE, linewidth=2, label="healthy median"),
        Line2D([], [], color=ORANGE, linewidth=2, marker="o", label="defective example"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 1.0 - 0.47 / height),
        labelcolor=INK2,
    )
    fig.suptitle(
        f"SYNTHETIC DATA — {parameter} relative to each lot's healthy median "
        f"(seed {manifest['seed']}, {manifest['lots']} lots, {manifest['parts']} parts)",
        color=INK,
        fontsize=13,
        x=0.5,
        y=1.0 - 0.13 / height,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.62 / height))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png, types


def _multivariate_panel(
    ax: plt.Axes, meas: pd.DataFrame, labels: pd.DataFrame, parameter: str, n: int, h0: int
) -> None:
    """First-reading scatter of the correlated pair: each inside +/-2.5 sigma, jointly off-ridge."""
    other = "ILEAK" if parameter == "IDDQ" else "IDDQ"
    a = _ratio_to_lot_median(meas, labels, parameter)[["lot_id", "part_id", h0, "defect_type"]]
    b = _ratio_to_lot_median(meas, labels, other)[["lot_id", "part_id", h0]]
    d = a.merge(b, on=["lot_id", "part_id"], suffixes=("_a", "_b")).dropna()
    ka, kb = f"{h0}_a", f"{h0}_b"
    healthy = d.loc[d["defect_type"] == "healthy"]
    mv = d.loc[d["defect_type"] == "multivariate"]
    hs = healthy.sample(min(3000, len(healthy)), random_state=0)
    ax.scatter(hs[ka], hs[kb], s=6, color=MUTED, alpha=0.35, linewidths=0)
    ax.scatter(mv[ka], mv[kb], s=42, color=ORANGE, edgecolors=SURFACE, linewidths=1.5)
    # marginal +/- 2.5 sigma_robust box (log space) from the healthy population
    for key, axis in ((ka, "x"), (kb, "y")):
        logs = np.log(healthy[key].to_numpy())
        sigma = 1.4826 * np.median(np.abs(logs - np.median(logs)))
        lo, hi = np.exp(np.median(logs) - 2.5 * sigma), np.exp(np.median(logs) + 2.5 * sigma)
        if axis == "x":
            ax.axvline(lo, color=INK2, linewidth=1)
            ax.axvline(hi, color=INK2, linewidth=1)
        else:
            ax.axhline(lo, color=INK2, linewidth=1)
            ax.axhline(hi, color=INK2, linewidth=1)
    ax.text(
        0.02,
        0.97,
        "lines: marginal ±2.5σ (robust)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK2,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _ratio_ticks(ax.xaxis, ax.get_xlim())
    _ratio_ticks(ax.yaxis, ax.get_ylim())
    ax.set_xlabel(f"{parameter} / lot median at {h0} h")
    ax.set_ylabel(f"{other} / lot median at {h0} h")
    ax.set_title(
        f"multivariate  (n = {n}): {parameter} vs {other} at {h0} h",
        loc="left",
        color=INK,
        fontsize=11,
    )


def plot_lot_overview(
    data_dir: str | Path,
    out_png: str | Path,
    *,
    parameter: str = "IDDQ",
    hour: int | None = None,
    n_lots: int = 25,
) -> Path:
    """Left: one large lot in absolute units with the datasheet limit. Right: lot-to-lot spread.

    `hour` defaults to the first stressed reading in the data (24 h on the standard schedule).
    """
    data_dir, out_png = Path(data_dir), Path(out_png)
    meas, labels, limits, manifest = _load(data_dir)
    hours = sorted(int(h) for h in meas["hour"].unique())
    if hour is None:
        hour = hours[1] if len(hours) > 1 else hours[0]
    if hour not in hours:
        raise ValueError(f"hour {hour} is not one of the dataset's readings {hours}")
    h0 = hours[0]
    sub = meas.loc[(meas["parameter"] == parameter)].merge(labels, on=["lot_id", "part_id"])
    pn = sorted(sub["part_number"].unique())[0]
    sub = sub.loc[sub["part_number"] == pn]
    unit = sub["unit"].iloc[0]
    limit = limits.get(pn, {}).get(parameter, {}).get("max")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.6]}
    )
    _style(ax1)
    _style(ax2)

    # --- left: the largest lot at `hour`
    sizes = sub.loc[sub["hour"] == h0].groupby("lot_id").size()
    lot_id = sizes.idxmax()
    lot = sub.loc[(sub["lot_id"] == lot_id) & (sub["hour"] == hour)].dropna(subset=["value"])
    lot = lot.assign(x=np.random.default_rng(0).uniform(-0.3, 0.3, size=len(lot)))
    on_param = lot["affected_parameters"].astype(str).str.contains(parameter)
    good = lot.loc[~lot["is_defective"]]
    here = lot.loc[lot["is_defective"] & on_param]
    elsewhere = lot.loc[lot["is_defective"] & ~on_param]
    ax1.scatter(good["x"], good["value"], s=9, color=MUTED, alpha=0.5, linewidths=0)
    ax1.scatter(
        elsewhere["x"],
        elsewhere["value"],
        s=34,
        facecolors=SURFACE,
        edgecolors=INK2,
        linewidths=1.2,
        zorder=3,
    )
    ax1.scatter(
        here["x"], here["value"], s=46, color=ORANGE, edgecolors=SURFACE, linewidths=1.5, zorder=4
    )
    # One label per defect type (at its highest point, with a count) so labels do not collide.
    for dtype, grp in here.groupby("defect_type"):
        top = grp.loc[grp["value"].idxmax()]
        text = str(dtype).replace("_", " ") + (f" (×{len(grp)})" if len(grp) > 1 else "")
        ax1.annotate(
            text,
            (top["x"], top["value"]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=INK2,
            va="center",
        )
    if len(good):
        med = float(good["value"].median())
        ax1.axhline(med, color=BLUE, linewidth=2)
        ax1.text(
            1.38,
            med,
            f"lot median {med:.3g} {unit}",
            color=INK2,
            fontsize=9,
            va="bottom",
            ha="right",
        )
    if limit is not None:
        ax1.axhline(limit, color=INK2, linewidth=1)
        ax1.text(
            1.38,
            limit,
            f"datasheet max {limit:g} {unit}",
            color=INK2,
            fontsize=9,
            va="bottom",
            ha="right",
        )
    ax1.set_yscale("log")
    ax1.set_xlim(-0.6, 1.4)
    ax1.set_xticks([])
    ax1.set_ylabel(f"{parameter} at {hour} h ({unit})")
    ax1.set_title(
        f"lot {lot_id} ({pn}, n = {sizes.max()}) at {hour} h", loc="left", color=INK, fontsize=11
    )

    # --- right: first n_lots lots of this part number at the first reading
    lots0 = sub.loc[sub["hour"] == h0].dropna(subset=["value"])
    first = sorted(lots0["lot_id"].unique())[:n_lots]
    data = [lots0.loc[lots0["lot_id"] == lid, "value"].to_numpy() for lid in first]
    bp = ax2.boxplot(
        data,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markerfacecolor": MUTED,
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
    )
    for box in bp["boxes"]:
        box.set(facecolor=SURFACE, edgecolor=BLUE, linewidth=1.3)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=MUTED, linewidth=1)
    for line in bp["medians"]:
        line.set(color=INK, linewidth=1.5)
    if limit is not None:
        ax2.axhline(limit, color=INK2, linewidth=1)
        ax2.text(
            len(first) + 0.4,
            limit,
            f"datasheet max {limit:g} {unit}",
            color=INK2,
            fontsize=9,
            va="bottom",
            ha="right",
        )
    ax2.set_yscale("log")
    ax2.set_xticks(range(1, len(first) + 1))
    ax2.set_xticklabels(
        [f"{lid}\nn={len(v)}" for lid, v in zip(first, data, strict=True)],
        fontsize=7.5,
        rotation=90,
    )
    ax2.set_ylabel(f"{parameter} at {h0} h ({unit})")
    ax2.set_title(
        f"lot-to-lot variation and lot sizes: first {len(first)} {pn} lots at {h0} h",
        loc="left",
        color=INK,
        fontsize=11,
    )

    handles = [
        Line2D([], [], color=MUTED, marker="o", linestyle="none", label="healthy part"),
        Line2D(
            [],
            [],
            color=ORANGE,
            marker="o",
            linestyle="none",
            markersize=8,
            label=f"defective on {parameter} (labelled)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=SURFACE,
            markeredgecolor=INK2,
            label="defective on another parameter",
        ),
        Line2D([], [], color=BLUE, linewidth=2, label="lot median (healthy)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.93),
        labelcolor=INK2,
    )
    fig.suptitle(
        f"SYNTHETIC DATA — {parameter}: anomalous relative to the lot, inside the datasheet limit "
        f"(seed {manifest['seed']})",
        color=INK,
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png
