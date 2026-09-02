from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from entity import Agent

def configure_plot() -> None:
    plt.style.use(['science', 'ieee'])
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 28
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams.update({
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.5,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
    })

def plot_tracking_error(
    agents: list[Agent],
    active_topology_history: np.ndarray,
    time_step_delta: float
) -> None:
    configure_plot()

    time_steps = agents[0].e.shape[1]
    time = np.arange(time_steps) * time_step_delta
    active_topology_history = np.asarray(active_topology_history)[:time_steps]

    fig, (ax_error, ax_topology) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1.4],
            "hspace": 0.08
        }
    )

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple"
    ]

    for agent_index, agent in enumerate(agents):
        tracking_error = np.linalg.norm(agent.e, axis=0)

        print(
            f"PLOTTER {agent.id}: "
            f"first={tracking_error[0]:.6f}, "
            f"max={np.max(tracking_error):.6f}, "
            f"last={tracking_error[-1]:.6f}"
        )

        ax_error.plot(
            time,
            tracking_error,
            color=colors[agent_index],
            label=rf"$e_{{{agent_index + 1}}}$"
        )

    ax_error.set_ylabel(r"$\|e_i(t)\|$")
    ax_error.grid(True, alpha=0.3)

    ax_topology.step(
        time,
        active_topology_history,
        where="post",
        color="black"
    )

    active_topologies = np.unique(active_topology_history).astype(int)

    ax_topology.set_yticks(active_topologies)
    ax_topology.set_yticklabels(
        [
            rf"$\mathcal{{F}}_{{{topology_num}}}$"
            for topology_num in active_topologies
        ]
    )

    ax_topology.set_ylabel(r"$\sigma(t)$")
    ax_topology.set_xlabel("Time (s)")
    ax_topology.set_ylim(
        active_topologies.min() - 0.5,
        active_topologies.max() + 0.5
    )
    ax_topology.grid(True, axis="x", alpha=0.3)

    ax_error.text(
        0.01,
        0.95,
        r"\textbf{(a)}",
        transform=ax_error.transAxes,
        verticalalignment="top"
    )

    ax_topology.text(
        0.01,
        0.92,
        r"\textbf{(b)}",
        transform=ax_topology.transAxes,
        verticalalignment="top"
    )

    ax_error.set_xlim(time[0], time[-1])
    fig.align_ylabels([ax_error, ax_topology])

    figures_path = Path(__file__).parent / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    figure_file = figures_path / "tracking_error.pdf"

    fig.savefig(
        figure_file,
        bbox_inches="tight"
    )

    plt.show()