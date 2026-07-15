# plots/plot_script.py
import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from src.deco.online_baselines import (
    ADAPTIVE_BASELINE_NAMES,
    ONLINE_BASELINE_NAMES,
    best_rate_and_loss,
)
from src.deco.plotting import mask_large_losses
from src.deco.utils import load_results_from_hdf5

# --- Configuration and Styling ---
RESULTS_DIR = "results"
SAVE_DIR = "tex/Figs"
os.makedirs(SAVE_DIR, exist_ok=True)

# IEEE Publication Standards Configuration
plt.rcParams.update(
    {
        # Font settings - IEEE prefers Times New Roman or similar serif fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # For math symbols to match Times
        # Text sizes for IEEE papers (8-12pt for text, 8-10pt for captions)
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 10,
        # High quality output
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "pdf.fonttype": 42,  # True Type fonts for better compatibility
        # Clean appearance
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    }
)

FONT_SETTINGS = {"fontsize": 10}
TITLE_SETTINGS = {"fontsize": 10, "fontweight": "normal"}

BASE_COLORS = {
    "DECO-i (KT)": "#648FFF",
    "DECO-i (exp)": "#785EF0",
    "DECO-ii (exp)": "#FE6100",
    "DECO-ii (KT)": "#FFB000",
    "DOGD": "#dc267f",
    "D-AdaGrad": "#008b8b",
    "D-RMSProp": "#7a5195",
    "D-Adam": "#2ca02c",
    "Centralized": "#393939",
}

ER_COLORS = {"low": "#fde725", "med": "#35b779", "hi": "#31688e"}
GOSSIP_COLORS = {"low": "#f0f921", "med": "#ed7953", "hi": "#9c179e"}
BASE_LINESTYLES = {"DECO-i": "--", "DECO-ii": ":", "DOGD": "-", "Centralized": "-."}
SETTING_MARKERS = {"KT": "x", "exp": "+", "central": ""}

STYLE_GUIDE = {
    "DECO-i (exp)": {
        "color": BASE_COLORS["DECO-i (exp)"],
        "linestyle": "-",
        "marker": "x",
    },
    "DECO-ii (exp)": {
        "color": BASE_COLORS["DECO-ii (exp)"],
        "linestyle": "--",
        "marker": "+",
    },
    "DECO-i (KT)": {
        "color": BASE_COLORS["DECO-i (KT)"],
        "linestyle": "--",
        "marker": "+",
    },
    "DECO-ii (KT)": {
        "color": BASE_COLORS["DECO-ii (KT)"],
        "linestyle": ":",
        "marker": "x",
    },
    "DOGD_curve": {
        "color": BASE_COLORS["DOGD"],
        "linestyle": BASE_LINESTYLES["DOGD"],
        "marker": "v",
    },
    "DOGD": {
        "color": BASE_COLORS["DOGD"],
        "linestyle": "-",
        "marker": "v",
    },
    "D-AdaGrad": {
        "color": BASE_COLORS["D-AdaGrad"],
        "linestyle": "--",
        "marker": "o",
    },
    "D-RMSProp": {
        "color": BASE_COLORS["D-RMSProp"],
        "linestyle": "-.",
        "marker": "s",
    },
    "D-Adam": {
        "color": BASE_COLORS["D-Adam"],
        "linestyle": ":",
        "marker": "^",
    },
    "Centralized": {
        "color": BASE_COLORS["Centralized"],
        "linestyle": BASE_LINESTYLES["Centralized"],
        "marker": SETTING_MARKERS["central"],
    },
    "ER (p=0.1)": {"color": ER_COLORS["low"], "linestyle": "-", "marker": "o"},
    "ER (p=0.3)": {"color": ER_COLORS["med"], "linestyle": "--", "marker": "x"},
    "ER (p=1.0)": {"color": ER_COLORS["hi"], "linestyle": ":", "marker": "+"},
    "Constant (q=1)": {"color": GOSSIP_COLORS["low"], "linestyle": "-", "marker": "o"},
    "Logarithmic (q=log(t))": {
        "color": GOSSIP_COLORS["med"],
        "linestyle": "--",
        "marker": "x",
    },
    "Linear (q=0.1*t)": {"color": GOSSIP_COLORS["hi"], "linestyle": ":", "marker": "+"},
    "default": {"color": "#bcbd22", "linestyle": ":", "marker": "X"},
}

LATEX_LABEL_MAP = {
    "Constant (q=1)": r"Constant",
    "Logarithmic (q=log(t))": r"Logarithmic",
    "Linear (q=0.1*t)": r"Linear",
    "DECO-i (exp)": r"DECO-i (exp)",
    "DECO-ii (exp)": r"DECO-ii (exp)",
    "DECO-i (KT)": r"DECO-i (KT)",
    "DECO-ii (KT)": r"DECO-ii (KT)",
    "DGD (initial_lr=0.1)": r"DOGD ($\eta_0 = 0.1$)",
    "DGD (initial_lr=1.0)": r"DOGD ($\eta_0 = 1.0$)",
    "DGD (initial_lr=10.0)": r"DOGD ($\eta_0 = 10.0$)",
    "Centralized": r"Centralized",
    "ER (p=0.1)": r"ER ($p = 0.1$)",
    "ER (p=0.3)": r"ER ($p = 0.3$)",
    "ER (p=1.0)": r"ER ($p = 1.0$)",
}


def get_display_label(name):
    """Convert algorithm name to a nicely formatted LaTeX label for display."""
    for baseline_name in ADAPTIVE_BASELINE_NAMES:
        if name.startswith(baseline_name) and ("(p=" in name or "(q=" in name):
            return baseline_name
    if "eta0=" in name:
        prefix, remainder = name.split("eta0=", 1)
        value, separator, suffix = remainder.partition(")")
        closing = ")" if separator else ""
        return f"{prefix}$\\eta_0={value}${closing}{suffix}"
    return LATEX_LABEL_MAP.get(name, name)


def get_plot_style(name, data_length):
    """Fetches a consistent style for a given algorithm name from the guide."""
    style_name = name
    for baseline_name in ONLINE_BASELINE_NAMES:
        if name.startswith(baseline_name):
            style_name = baseline_name
            break
    base_style = STYLE_GUIDE.get(style_name, STYLE_GUIDE["default"]).copy()
    base_style["linewidth"] = 1.5
    base_style["markersize"] = 4
    base_style["markevery"] = max(1, data_length // 8)
    return base_style


def save_fig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved figure to {path}")


# --- FIGURE 1: Online baseline learning-rate sensitivity ---
baseline_tuning_file = os.path.join(RESULTS_DIR, "online_baseline_tuning_results.h5")
deco_results_file = os.path.join(RESULTS_DIR, "synthetic_results_cycle.h5")

if os.path.exists(baseline_tuning_file) and os.path.exists(deco_results_file):
    with h5py.File(baseline_tuning_file, "r") as f:
        baseline_results = load_results_from_hdf5(f)
    with h5py.File(deco_results_file, "r") as f:
        deco_results = load_results_from_hdf5(f)

    fig1, ax1 = plt.subplots(figsize=(3.5, 2.8))

    baseline_rates = {}
    plot_losses = {}
    for name in ONLINE_BASELINE_NAMES:
        rate_results = baseline_results[name]
        ordered = sorted(rate_results.items(), key=lambda item: float(item[0]))
        baseline_rates[name] = [float(key) for key, _ in ordered]
        plot_losses[name] = [np.sum(history["network_loss"]) for _, history in ordered]

    deco_final_losses = {
        name: np.sum(data["network_loss"])
        for name, data in deco_results.items()
        if name in {"DECO-i (KT)", "DECO-ii (KT)", "Centralized"}
    }
    reference_rates = baseline_rates[ONLINE_BASELINE_NAMES[0]]
    for name, loss in deco_final_losses.items():
        plot_losses[name] = [loss] * len(reference_rates)

    plot_losses = mask_large_losses(plot_losses)
    for name in ONLINE_BASELINE_NAMES:
        rates = baseline_rates[name]
        ax1.plot(
            rates,
            plot_losses[name],
            label=name,
            **get_plot_style(name, len(rates)),
        )

    for name in deco_final_losses:
        loss_vector = plot_losses[name]
        style = get_plot_style(name, len(loss_vector))
        display_label = get_display_label(name)
        ax1.plot(reference_rates, loss_vector, label=display_label, **style)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Base / Initial Learning-Rate Scale ($\\eta_0$)")
    ax1.set_ylabel("Final Cumulative Network Loss")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    save_fig(fig1, "online_baseline_sensitivity.pdf")
    plt.close()


# --- FIGURE 2: Impact of Network Connectivity ---
def moving_average(data, window_size=100):
    """Computes the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size), "valid") / window_size


connectivity_file = os.path.join(RESULTS_DIR, "connectivity_results.h5")
if os.path.exists(connectivity_file):
    with h5py.File(connectivity_file, "r") as f:
        results = load_results_from_hdf5(f)

    # Create a 2x1 subplot figure to show both cumulative and per-round loss
    fig2, (ax_cum, ax_inst) = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharex=True, constrained_layout=True
    )

    # Panel 1: Cumulative Network Loss (Original Plot)
    for name, data in results.items():
        cumulative_loss = np.cumsum(data["network_loss"])
        style = get_plot_style(name, len(cumulative_loss))
        display_label = get_display_label(name)
        ax_cum.plot(cumulative_loss, label=display_label, **style)

    ax_cum.set_ylabel("Cumulative Network Loss")
    ax_cum.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_cum.legend(fontsize=6.5, ncol=2)

    # Panel 2: Per-Round Network Loss (Smoothed)
    smoothing_window = 250
    for name, data in results.items():
        network_loss = data["network_loss"]
        smoothed_loss = moving_average(network_loss, window_size=smoothing_window)
        # Adjust time axis to center the moving average window
        time_axis = np.arange(len(smoothed_loss)) + smoothing_window / 2
        style = get_plot_style(name, len(network_loss))
        display_label = get_display_label(name)
        ax_inst.plot(time_axis, smoothed_loss, label=display_label, **style)

    ax_inst.set_xlabel("Time (t)")
    ax_inst.set_ylabel("Per-Round Network Loss\n(Smoothed)")
    ax_inst.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_inst.legend(fontsize=6.5, ncol=2)
    ax_inst.set_yscale("log")
    save_fig(fig2, "connectivity_impact.pdf")
    plt.close()

# --- FIGURE 3: Communication-Loss Trade-off ---
gossip_tradeoff_file = os.path.join(RESULTS_DIR, "gossip_tradeoff_results.h5")
if os.path.exists(gossip_tradeoff_file):
    with h5py.File(gossip_tradeoff_file, "r") as f:
        results = load_results_from_hdf5(f)

    fig3, ax3 = plt.subplots(figsize=(3.5, 2.6))  # IEEE single column
    for name, data in results.items():
        cumulative_loss = np.cumsum(data["network_loss"])
        style = get_plot_style(name, len(cumulative_loss))
        display_label = get_display_label(name)  # Use nicer LaTeX label
        ax3.plot(cumulative_loss, label=display_label, **style)

    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("Cumulative Network Loss")
    ax3.legend(fontsize=6.5, ncol=2)
    save_fig(fig3, "gossip_tradeoff.pdf")
    plt.close()


def plot_multi_dataset_sensitivity(all_data):
    """Plot all online baseline sweeps and parameter-free references."""
    dataset_results = [res for res in all_data["results"] if res is not None]
    n_datasets = len(dataset_results)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 3 * n_rows), sharex=False)
    axes_flat = axes.flatten()
    algos_to_plot = ["DECO-i (KT)", "DECO-ii (KT)", "Centralized"]

    for i, data in enumerate(dataset_results):
        ax = axes_flat[i]
        dataset_name = data["dataset_name"]
        results = data["results"]
        tuned_results = results["Online_tune"]
        baseline_rates = {}
        plot_losses = {}

        for name in ONLINE_BASELINE_NAMES:
            rate_results = tuned_results[name]
            ordered = sorted(rate_results.items(), key=lambda item: float(item[0]))
            baseline_rates[name] = [float(key) for key, _ in ordered]
            plot_losses[name] = [
                np.sum(history["network_loss"]) for _, history in ordered
            ]

        reference_rates = baseline_rates[ONLINE_BASELINE_NAMES[0]]
        for name in algos_to_plot:
            if name in results:
                final_loss = np.sum(results[name]["network_loss"])
                plot_losses[name] = [final_loss] * len(reference_rates)

        plot_losses = mask_large_losses(plot_losses)
        for name in ONLINE_BASELINE_NAMES:
            rates = baseline_rates[name]
            ax.plot(
                rates,
                plot_losses[name],
                label=name,
                **get_plot_style(name, len(rates)),
            )

        for name in algos_to_plot:
            if name in results:
                loss_vector = plot_losses[name]
                style = get_plot_style(name, len(loss_vector))
                display_label = get_display_label(name)
                ax.plot(reference_rates, loss_vector, label=display_label, **style)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dataset_name.replace("_", " ").title())
        ax.set_ylabel("Final Cumulative Network Loss")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if i >= n_datasets - n_cols:
            ax.set_xlabel("Base / Initial Learning-Rate Scale ($\\eta_0$)")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    save_fig(fig, "multi_dataset_sensitivity.pdf")
    plt.close()


def write_online_baseline_table(all_data):
    """Write the exact best-in-grid values used in the manuscript table."""
    rows = []
    for data in all_data["results"]:
        if data is None:
            continue
        dataset_name = data["dataset_name"]
        results = data["results"]
        for name in ("DECO-i (KT)", "DECO-ii (KT)"):
            history = results[name]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": name,
                    "selected_eta0": "",
                    "cumulative_network_loss": float(np.sum(history["network_loss"])),
                    "communication_scalars": float(
                        np.sum(history["communication_scalars"])
                    ),
                }
            )
        for name in ONLINE_BASELINE_NAMES:
            rate_results = results["Online_tune"][name]
            rate, loss = best_rate_and_loss(rate_results)
            rate_key = min(rate_results, key=lambda key: abs(float(key) - rate))
            history = rate_results[rate_key]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": name,
                    "selected_eta0": rate,
                    "cumulative_network_loss": loss,
                    "communication_scalars": float(
                        np.sum(history["communication_scalars"])
                    ),
                }
            )

    path = os.path.join(RESULTS_DIR, "online_baseline_table.csv")
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved table values to {path}")


# --- FIGURE 4: Performance on Real-World Datasets (New Version) ---
multi_dataset_file = os.path.join(RESULTS_DIR, "multi_dataset_comparison.h5")
if os.path.exists(multi_dataset_file):
    with h5py.File(multi_dataset_file, "r") as f:
        all_data = load_results_from_hdf5(f)
    plot_multi_dataset_sensitivity(all_data)
    write_online_baseline_table(all_data)
