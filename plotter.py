from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from entity import Agent

def configure_plot() -> None:
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        "figure.dpi": 100,
        "font.family": "serif",
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.5,
        "legend.frameon": True,
        "legend.edgecolor": "black",
    })

def plot_tracking_error(agents: list[Agent], active_topology_history: np.ndarray, time_step_delta: float) -> None:
    configure_plot()

    time_steps = agents[0].e.shape[1]
    time = np.arange(time_steps) * time_step_delta
    active_topology_history = np.asarray(active_topology_history)[:time_steps]
    fig, (ax_error, ax_estimation, ax_regulation, ax_topology) = plt.subplots(
        4, 1, figsize=(10, 11), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2.5, 2.5, 1.4], "hspace": 0.08}
    )
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, agent in enumerate(agents):
        ax_error.plot(time, np.linalg.norm(agent.e, axis=0), color=colors[i], label=rf"$e_{{{i+1}}}$")
        ax_estimation.plot(time, np.linalg.norm(agent.e_hat, axis=0), color=colors[i], label=rf"$\hat{{e}}_{{{i+1}}}$")
        ax_regulation.plot(time, np.linalg.norm(agent.e_tilde, axis=0), color=colors[i], label=rf"$\tilde{{e}}_{{{i+1}}}$")

    ax_error.set_ylabel(r"$\|e_i(t)\|$")
    ax_estimation.set_ylabel(r"$\|\hat{e}_i(t)\|$")
    ax_regulation.set_ylabel(r"$\|\tilde{e}_i(t)\|$")
    ax_error.grid(True, alpha=0.3); ax_estimation.grid(True, alpha=0.3); ax_regulation.grid(True, alpha=0.3)

    zoom_start_time = 0.2 * time[-1]
    start_index = np.searchsorted(time, zoom_start_time)
    inset = inset_axes(ax_regulation, width="40%", height="55%", loc="upper right", borderpad=1.0)
    zoom_min, zoom_max = np.inf, -np.inf

    for i, agent in enumerate(agents):
        reg = np.linalg.norm(agent.e_tilde, axis=0)
        inset.plot(time, reg, color=colors[i])
        zoom_min = min(zoom_min, np.min(reg[start_index:]))
        zoom_max = max(zoom_max, np.max(reg[start_index:]))

    zoom_range = zoom_max - zoom_min if zoom_max - zoom_min != 0 else 1.0
    pad = 0.1 * zoom_range
    inset.set_xlim(zoom_start_time, time[-1])
    inset.set_ylim(max(0.0, zoom_min - pad), zoom_max + pad)
    inset.tick_params(axis="both", labelsize=8); inset.grid(True, alpha=0.3)
    mark_inset(ax_regulation, inset, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=0.8)

    ax_topology.step(time, active_topology_history, where="post", color="black")
    active_topologies = np.unique(active_topology_history).astype(int)
    ax_topology.set_yticks(active_topologies)
    ax_topology.set_yticklabels([rf"$\mathcal{{F}}_{{{n}}}$" for n in active_topologies])
    ax_topology.set_ylabel(r"$\sigma(t)$"); ax_topology.set_xlabel("Time (s)")
    ax_topology.set_ylim(active_topologies.min() - 0.5, active_topologies.max() + 0.5)
    ax_topology.grid(True, axis="x", alpha=0.3)

    ax_error.set_xlim(time[0], time[-1])
    fig.align_ylabels([ax_error, ax_estimation, ax_regulation, ax_topology])
    figures_path = Path(__file__).parent / "figures"; figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_path / "tracking_errors.pdf", bbox_inches="tight")
    plt.show()