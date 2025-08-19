#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 17:32:55 2025

@author: abolfazl
"""

"""
MMDetection Log Analyzer

What it does
------------
- Parse MMDetection training logs (per-iteration lines) with a regex.
- Aggregate metrics by epoch (mean per epoch).
- Save a single epoch-level CSV.
- Plot epoch-wise curves (one metric per figure).
- Optional rolling (smoothing) on epoch curves.

"""

from pathlib import Path
import re
import os
from typing import Iterable, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt


LOG_LINE_RE = re.compile(
    r"Epoch \[(\d+)\]\[(\d+)/(\d+)\]\s*lr:\s*([0-9.eE+-]+).*?"
    r"loss_rpn_cls:\s*([0-9.]+),\s*"
    r"loss_rpn_bbox:\s*([0-9.]+),\s*"
    r"loss_cls:\s*([0-9.]+),\s*"
    r"acc:\s*([0-9.]+),\s*"
    r"loss_bbox:\s*([0-9.]+),\s*"
    r"loss:\s*([0-9.]+),\s*"
    r"grad_norm:\s*([0-9.]+)",
    re.IGNORECASE
)

def _parse_iterations(log_path: Path) -> pd.DataFrame:
    """Parse the raw log into a per-iteration DataFrame. We won't plot iterations;
    this is only used to compute epoch means."""
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            it_in_epoch = int(m.group(2))
            iters_per_epoch = int(m.group(3))
            lr = float(m.group(4))
            loss_rpn_cls = float(m.group(5))
            loss_rpn_bbox = float(m.group(6))
            loss_cls = float(m.group(7))
            acc = float(m.group(8))
            loss_bbox = float(m.group(9))
            loss_total = float(m.group(10))
            grad_norm = float(m.group(11))


            rows.append({
                "epoch": epoch,
                "iter_in_epoch": it_in_epoch,
                "iters_per_epoch": iters_per_epoch,
                "lr": lr,
                "acc": acc,
                "loss_rpn_cls": loss_rpn_cls,
                "loss_rpn_bbox": loss_rpn_bbox,
                "loss_cls": loss_cls,
                "loss_bbox": loss_bbox,
                "loss_total": loss_total,
                "grad_norm": grad_norm,
            })
    if not rows:
        raise ValueError("No training iterations found. Check the log format/regex.")
    df = pd.DataFrame(rows).sort_values(["epoch", "iter_in_epoch"]).reset_index(drop=True)
    return df


def _epoch_means(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Aggregate iteration rows into per-epoch mean metrics."""
    exist = [m for m in metrics if m in df.columns]
    if not exist:
        raise ValueError("None of the requested metrics exist in parsed DataFrame.")
    return df.groupby("epoch", as_index=False)[exist].mean(numeric_only=True)


def _maybe_add_rolling(epoch_df: pd.DataFrame, metrics: Iterable[str], window: int) -> pd.DataFrame:
    """Optionally add centered rolling mean columns (e.g., window=3)."""
    if window and window > 1 and len(epoch_df) >= 3:
        for m in metrics:
            if m in epoch_df.columns:
                epoch_df[f"{m}_roll"] = epoch_df[m].rolling(window=min(window, len(epoch_df)), center=True).mean()
    return epoch_df


def _plot_epoch_curves(epoch_df: pd.DataFrame,
                       metrics: Iterable[str],
                       outdir: Optional[Path],
                       show: bool,
                       save_png: bool) -> None:
    """Plot one figure per metric over epochs (no seaborn, no custom colors)."""
    x = epoch_df["epoch"]
    for m in metrics:
        if m not in epoch_df.columns:
            continue
        plt.figure()
        plt.plot(x, epoch_df[m], label=f"{m} (epoch mean)")
        roll_col = f"{m}_roll"
        if roll_col in epoch_df.columns:
            plt.plot(x, epoch_df[roll_col], label=f"{m} (rolling)")
        plt.xlabel("epoch")
        plt.ylabel(m)
        plt.title(f"{m} over epochs")
        plt.legend()
        plt.tight_layout()
        if save_png and outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / f"epoch_{m}.png", dpi=150)
            plt.show()
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()


def analyze_mmdet_log_epoch_only(
    log_path: str,
    outdir: str = "./epoch_out",
    metrics: Tuple[str, ...] = (
        "loss_total", "loss_cls", "loss_bbox", "loss_rpn_cls", "loss_rpn_bbox", "lr", "grad_norm", "acc"
    ),
    roll_window: int = 3,
    show: bool = True,
    save_png: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Parse an MMDetection log and produce *epoch-only* outputs.

    Parameters
    ----------
    log_path : str
        Path to the MMDetection training log file.
    outdir : str
        Directory where the epoch CSV and (optionally) PNGs are saved.
    metrics : tuple of str
        Which metrics to aggregate and plot. Missing ones are skipped.
    roll_window : int
        Window size for centered rolling means on epoch curves (set 0 or 1 to disable).
    show : bool
        If True, display figures inline (useful in notebooks).
    save_png : bool
        If True, save PNG plots to `outdir`.

    Returns
    -------
    dict with:
        - "epoch_df": pandas.DataFrame of per-epoch means (plus optional *_roll columns).
    """
    log_path = Path(log_path)
    outdir_path = Path(outdir) if outdir else None

    # Parse per-iteration rows (only to compute per-epoch means)
    df_iter = _parse_iterations(log_path)

    # Compute epoch means for requested metrics
    epoch_df = _epoch_means(df_iter, metrics)

    # Save CSV (epoch-only)
    if outdir_path is not None:
        outdir_path.mkdir(parents=True, exist_ok=True)
        csv_path = outdir_path / "mmdet_epoch_means.csv"
        epoch_df.to_csv(csv_path, index=False)
        print(f"[+] Saved epoch CSV -> {csv_path}")

    # Optional smoothing
    epoch_df = _maybe_add_rolling(epoch_df, metrics, roll_window)

    # Plot epoch-wise curves (no iteration plots)
    _plot_epoch_curves(epoch_df, metrics, outdir_path, show=show, save_png=save_png)

    if outdir_path is not None:
        print(f"[+] Figures saved to -> {outdir_path.resolve()}")
    return {"epoch_df": epoch_df}


dfs = analyze_mmdet_log_epoch_only(
    log_path="./work_dirs/faster_rcnn_orpn_r50_fpn_1x_dota10_20250814_195045/20250814_195051.log",
    outdir="./epoch_out",
    metrics=("loss_total","loss_cls","loss_bbox","loss_rpn_cls","loss_rpn_bbox","lr","grad_norm","acc"),
    roll_window=3,
    show=True,
    save_png=True
)