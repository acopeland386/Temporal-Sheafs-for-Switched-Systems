from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from entity import Agent, Target, compute_exact_extension
from simulation.dynamics import get_initial_conditions
from building_sheaf import agent_coboundary, target_coboundary
from plotter import plot_results

def build_entity_specs(base_config: dict[str, Any], section_key: str) -> List[Tuple[NDArray[np.float64], dict[str, Any]]]:
    specs: List[Tuple[NDArray[np.float64], dict[str, Any]]] = []
    for index, item in enumerate(base_config[section_key]):
        dynamics_type = item["dynamics_type"]
        initial_position = np.array(
            item.get("initial_position", get_initial_conditions(dynamics_type)),dtype=float)
        merged_config = {**base_config, **item, "dynamics_type": dynamics_type}
        specs.append((initial_position, merged_config))

    return specs 

# ---------------------------------------------------------------------

def admissable_topologies() -> List[Tuple[int, List[List[int]]]]:

    config_path = Path(__file__).parent / "config_common.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    admissible_topologies = []
    for key, value in config_data.items():
        if key.startswith("top_") and key.endswith("_agent_edge"):
            num_agents = int(key.split('_')[1])
            edge_set = value
            admissible_topologies.append((num_agents, edge_set))

    return admissible_topologies

# ---------------------------------------------------------------------

def construct_undirected_neighborhood_set(entities: list[Any], edge_set: list[list[int]]) -> None:
    for i, j in edge_set:
        a, b = entities[i - 1], entities[j - 1]
        if b not in a.neighbors:
            a.neighbors.append(b)
        if a not in b.neighbors:
            b.neighbors.append(a)

# ---------------------------------------------------------------------

def construct_agent_target_edge_set(
    pin_matrix: list[list[int]]
) -> list[list[int]]:

    agent_target_edge_set = []

    for agent_index, row in enumerate(pin_matrix):
        if row[0] == 1:
            agent_target_edge_set.append([agent_index + 1, 1])

    return agent_target_edge_set

# ---------------------------------------------------------------------

def build_switching_schedule(base_config: dict[str, Any]) -> list[Tuple[float, int]]:   
    switching_schedule = []
    for block in base_config["switching_block"]:
        block["block_start_time"] = float(block["block_start_time"])
        block["block_end_time"] = float(block["block_end_time"])
        for subinterval in block["subintervals"]:
            current_time = subinterval["subinterval_start_time"]

            subinterval["subinterval_start_time"] = float(subinterval["subinterval_start_time"])
            subinterval["subinterval_end_time"] = float(subinterval["subinterval_end_time"])
            topology_num = int(subinterval["topology_num"])
            duration = float(subinterval["duration"])
            next_switch_time = current_time + duration
            
            switching_schedule.append((current_time, next_switch_time, topology_num))
            current_time = next_switch_time

    return switching_schedule

#---------------------------------------------------------------------

def get_active_topology(
    time: float,
    schedule: list[tuple[float, float, int]]
) -> int:

    for start_time, end_time, topology_num in schedule:
        if start_time <= time < end_time:
            return topology_num

    return schedule[-1][2]

#----------------------------------------------------------------------

def apply_active_topology(
    agents: list[Agent],
    topology: dict[str, Any]
) -> tuple[list[list[int]], list[list[int]]]:

    agent_edge_set = topology["agent_edges"]
    pin_matrix = topology["pin_matrix"]

    for agent in agents:
        agent.neighbors.clear()

    construct_undirected_neighborhood_set(agents,agent_edge_set)
    agent_target_edge_set = construct_agent_target_edge_set(pin_matrix)

    for agent_index, agent in enumerate(agents):
        agent.pin_row = np.array(pin_matrix[agent_index], dtype=np.float64)

    return agent_edge_set, agent_target_edge_set

#----------------------------------------------------------------------

def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]

    final_time: float = base_config["final_time"]
    time_step_delta: float = base_config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    
    np.random.seed(base_config["seed"])

    target_specs = build_entity_specs(base_config, section_key="target")
    agent_specs = build_entity_specs(base_config, section_key="agents")

    target: list[Target] = [
        Target(initial_position, time_steps, config)
        for initial_position, config in target_specs
    ]

    agents: list[Agent] = [
        Agent(initial_position=pos, time_steps=time_steps, config=conf,
              targets=target, pin_row=np.zeros(0, dtype=np.float64))
        for pos, conf in agent_specs
    ]

    active_topology_history = np.zeros(time_steps,dtype=int)
    schedule = build_switching_schedule(base_config)
    current_topology_num = None
    delta_q = []
    delta_T = []

    # update dynamics for each time step of active topology
    for step in range(1, time_steps):
        time = step * time_step_delta
        topology_num = get_active_topology(time, schedule)
        active_topology_history[step] = topology_num

        if topology_num != current_topology_num:
            topology = base_config["topologies"][str(topology_num)]
            agent_edge_set, agent_target_edge_set = apply_active_topology(agents, topology)

            print(f"\nTime: {time:.2f}, Active Topology: {topology_num}")
            print(f"Agent edge set: {agent_edge_set}")
            print(f"Agent-target edge set: {agent_target_edge_set}")

            delta_q = agent_coboundary(
                agents=agents,
                agent_edge_set=agent_edge_set,
                agent_target_edge_set=agent_target_edge_set
            )

            delta_T = target_coboundary(
                agents=agents,
                targets=target,
                agent_edge_set=agent_edge_set,
                agent_target_edge_set=agent_target_edge_set
            )

            current_topology_num = topology_num
            print(f"Agent coboundary matrix (delta_q):\n{delta_q}")
            print(f"Target coboundary matrix (delta_T):\n{delta_T}")

        # target dynamics update
        for target_entity in target:
            target_entity.compute_control_output()
            target_entity.update_dynamics(step)

        # agent dynamics update
        for agent in agents:
            agent.compute_control_output(step)
            agent.update_dynamics(step)

        # exact-extension update
        compute_exact_extension(
            agents=agents,
            targets=target,
            delta_q=delta_q,
            delta_T=delta_T,
            step=step
        )

        # observer update
        for agent in agents:
            agent.update_observer(step)

    plot_results(
        agents=agents,
        targets=target,
        active_topology_history=active_topology_history,
        time_step_delta=time_step_delta
    )

config_path = Path(__file__).parent / "config_common.json"

#----------------------------------------------------------------------

def load_configurations() -> list[dict[str, Any]]:
    config_path = Path(__file__).parent / "config_common.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return [config]

# ---------------------------------------------------------------------

def run_batch_simulation_with_results() -> None:
    configs = load_configurations()
    run_simulation_from_configs(configs)

run_batch_simulation_with_results()