"""
Title:       plot.py
Author:      Sarper Sakmak
ID:          14008175400
Section:     1
Assignment:  CMPE 442 Programming Assignment-1
Description:
    - Creates and saves three loss curve figures using ONLY matplotlib.
    - Figure 1: Train Loss vs Epoch      -> saved as train_loss.png
    - Figure 2: Validation Loss vs Epoch -> saved as val_loss.png
    - Figure 3: Test Loss vs Epoch       -> saved as test_loss.png
    - Each figure contains 9 lines (one per model combination).
    - Colors distinguish regularization type; line styles distinguish learning rate.
      * Blue  shades -> Ridge
      * Red   shades -> Lasso
      * Green shades -> ElasticNet
      * Solid  (--)  -> l1
      * Dashed (--)  -> l2
      * Dash-dot(-.) -> l3
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for all environments
import matplotlib.pyplot as plt
import numpy as np

# ── Visual encoding for 9 model combinations ─────────────────────────────────

# Color palette: three shades per regularization type
COLORS = {
    "Ridge"     : ["#1f77b4", "#aec7e8", "#08519c"],   # blue shades
    "Lasso"     : ["#d62728", "#f5a6a6", "#8b0000"],   # red shades
    "ElasticNet": ["#2ca02c", "#98df8a", "#006400"],   # green shades
}

# Line style encodes the learning rate
LINE_STYLES = {
    "l1": "-",    # solid      -> slowest learning rate
    "l2": "--",   # dashed     -> moderate learning rate
    "l3": "-.",   # dash-dot   -> fastest learning rate
}

LR_KEYS = ["l1", "l2", "l3"]


# ── Internal helper ───────────────────────────────────────────────────────────
def _plot_figure(all_losses, ylabel, title, filename):
    """
    STEP A: Generic figure builder used by all three public plot functions.

    Parameters
    ----------
    all_losses : dict  { (reg_type, lr_key) -> list of float }
    ylabel     : str   y-axis label
    title      : str   figure title
    filename   : str   output PNG filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs  = np.arange(1, len(next(iter(all_losses.values()))) + 1)

    # STEP A1: Draw one line per model combination
    for (reg, lr_key), losses in all_losses.items():
        color     = COLORS[reg][LR_KEYS.index(lr_key)]
        linestyle = LINE_STYLES[lr_key]
        label     = f"{reg}-{lr_key}"
        ax.plot(epochs, losses,
                color=color, linestyle=linestyle,
                linewidth=1.8, label=label)

    # STEP A2: Add labels, title, grid, and legend
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # STEP A3: Save figure to disk
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {filename}")


# ── Public interface ──────────────────────────────────────────────────────────
def plot_all_losses(results):
    """
    STEP B: Build and save all three loss figures.

    Parameters
    ----------
    results : dict  { (reg_type, lr_key) -> {
                          "train_losses": [...],
                          "val_losses"  : [...],
                          "test_losses" : [...]
                      } }
    """
    # STEP B1: Separate the three loss types into individual dicts
    train_dict = {k: v["train_losses"] for k, v in results.items()}
    val_dict   = {k: v["val_losses"]   for k, v in results.items()}
    test_dict  = {k: v["test_losses"]  for k, v in results.items()}

    # STEP B2: Create and save each figure
    _plot_figure(train_dict,
                 ylabel="Cross-Entropy Loss (+ Regularization)",
                 title="Train Loss vs Epoch — All 9 Models",
                 filename="train_loss.png")

    _plot_figure(val_dict,
                 ylabel="Cross-Entropy Loss (+ Regularization)",
                 title="Validation Loss vs Epoch — All 9 Models",
                 filename="val_loss.png")

    _plot_figure(test_dict,
                 ylabel="Cross-Entropy Loss (+ Regularization)",
                 title="Test Loss vs Epoch — All 9 Models",
                 filename="test_loss.png")