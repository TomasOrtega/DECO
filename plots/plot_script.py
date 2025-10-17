# plots/plot_script.py
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from src.deco.utils import load_results_from_hdf5

# --- Configuration and Styling ---
RESULTS_DIR = "results"
SAVE_DIR = "plots/Figs"
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
    return LATEX_LABEL_MAP.get(name, name)


def get_plot_style(name, data_length):
    """Fetches a consistent style for a given algorithm name from the guide."""
    base_style = STYLE_GUIDE.get(name, STYLE_GUIDE["default"]).copy()
    base_style["linewidth"] = 1.5
    base_style["markersize"] = 4
    base_style["markevery"] = max(1, data_length // 8)
    return base_style


def save_fig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved figure to {path}")


# --- FIGURE 1: DGD Sensitivity ---
dgd_tuning_file = os.path.join(RESULTS_DIR, "dgd_tuning_results.h5")
deco_results_file = os.path.join(RESULTS_DIR, "synthetic_results_cycle.h5")

if os.path.exists(dgd_tuning_file) and os.path.exists(deco_results_file):
    with h5py.File(dgd_tuning_file, "r") as f:
        dgd_results = load_results_from_hdf5(f)
    with h5py.File(deco_results_file, "r") as f:
        deco_results = load_results_from_hdf5(f)

    learning_rates = []
    final_losses = []
    # Convert string keys from HDF5 back to float for correct numeric sorting
    sorted_lrs = sorted([float(k) for k in dgd_results.keys()])
    for lr in sorted_lrs:
        data = dgd_results[str(lr)]
        learning_rates.append(lr)
        # Access the 'network_loss' field of the structured array
        final_losses.append(np.sum(data["network_loss"]))

    # IEEE single column width is ~3.5 inches, double column is ~7 inches
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.6))  # IEEE single column, 3:4 aspect ratio

    style_dgd = get_plot_style("DOGD_curve", len(final_losses))
    ax1.plot(learning_rates, final_losses, label="DOGD", **style_dgd)

    deco_final_losses = {
        name: np.sum(data["network_loss"])
        for name, data in deco_results.items()
        if "DECO" in name or name == "Centralized"
    }

    for name, loss in deco_final_losses.items():
        loss_vector = [loss] * len(learning_rates)
        style = get_plot_style(name, len(loss_vector))
        display_label = get_display_label(name)
        ax1.plot(learning_rates, loss_vector, label=display_label, **style)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Initial Learning Rate ($\\eta_0$)")
    ax1.set_ylabel("Final Cumulative Network Loss")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    save_fig(fig1, "dgd_sensitivity.pdf")
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
    ax_cum.legend()

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
    ax_inst.legend()
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
    ax3.legend()
    save_fig(fig3, "gossip_tradeoff.pdf")
    plt.close()


def plot_multi_dataset_sensitivity(all_data):
    """
    Creates a multi-panel plot showing DOGD sensitivity and DECO performance
    for each real-world dataset.
    """
    dataset_results = [res for res in all_data["results"] if res is not None]
    n_datasets = len(dataset_results)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 2.8 * n_rows), sharex=False)
    axes_flat = axes.flatten()
    algos_to_plot = ["DECO-i (KT)", "DECO-ii (KT)", "Centralized"]

    for i, data in enumerate(dataset_results):
        ax = axes_flat[i]
        dataset_name = data["dataset_name"]
        results = data["results"]
        all_losses = []
        learning_rates = []
        final_losses = []

        if "DGD_tune" in results:
            dgd_tune_results = results["DGD_tune"]
            learning_rates = sorted([float(k) for k in dgd_tune_results.keys()])
            final_losses = [
                np.sum(dgd_tune_results[str(lr)]["network_loss"])
                for lr in learning_rates
            ]
            all_losses.extend(final_losses)

        for name in algos_to_plot:
            if name in results:
                res_data = results[name]
                final_loss = np.sum(res_data["network_loss"])
                all_losses.append(final_loss)

        # Calculate threshold: 5 times the minimum
        threshold = None
        if all_losses:
            threshold = 5 * min(all_losses)

        # Plot DOGD curve with NaN for values above threshold
        if final_losses and threshold is not None:
            final_losses_filtered = np.array(final_losses, dtype=float)
            final_losses_filtered[final_losses_filtered > threshold] = np.nan
            style_dgd = get_plot_style("DOGD_curve", len(final_losses_filtered))
            ax.plot(
                learning_rates,
                final_losses_filtered,
                label=get_display_label("DOGD"),
                **style_dgd,
            )

        for name in algos_to_plot:
            if name in results:
                res_data = results[name]
                final_loss = np.sum(res_data["network_loss"])
                display_loss = (
                    final_loss
                    if threshold is None or final_loss <= threshold
                    else np.nan
                )
                loss_vector = [display_loss] * len(learning_rates)
                style = get_plot_style(name, len(loss_vector))
                display_label = get_display_label(name)
                ax.plot(learning_rates, loss_vector, label=display_label, **style)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dataset_name.replace("_", " ").title())
        ax.set_ylabel("Final Cumulative Network Loss")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if i >= n_datasets - n_cols:
            ax.set_xlabel("Initial Learning Rate ($\\eta_0$)")
        ax.legend()

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    save_fig(fig, "multi_dataset_sensitivity.pdf")
    plt.close()


# --- FIGURE 4: Performance on Real-World Datasets (New Version) ---
multi_dataset_file = os.path.join(RESULTS_DIR, "multi_dataset_comparison.h5")
if os.path.exists(multi_dataset_file):
    with h5py.File(multi_dataset_file, "r") as f:
        all_data = load_results_from_hdf5(f)
    plot_multi_dataset_sensitivity(all_data)
