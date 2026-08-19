from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from entity import Agent, Target
from simulation.dynamics import get_initial_conditions
from building_sheaf import agent_coboundary, target_coboundary

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

def construct_target_edge_set(
    pin_matrix: list[list[int]]
) -> list[list[int]]:

    target_edge_set = []

    for agent_index, row in enumerate(pin_matrix):

        if row[0] == 1:
            target_edge_set.append(
                [agent_index + 1, 1]
            )

    return target_edge_set

# ---------------------------------------------------------------------

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

    topologies = admissable_topologies()

    for topology_num, agent_edge_set in topologies:

        pin_matrix = base_config[f"top_{topology_num}_pin_matrix"]

        target_edge_set = construct_target_edge_set(pin_matrix)

        print(f"\nTopology {topology_num}")
        print(f"Agent edge set: {agent_edge_set}")
        print(f"Target edge set: {target_edge_set}")

        delta_q = agent_coboundary(
            agents=agents,
            agent_edge_set=agent_edge_set,
            target_edge_set=target_edge_set
        )

        delta_T = target_coboundary(
            agents=agents,
            targets=target,
            agent_edge_set=agent_edge_set,
            target_edge_set=target_edge_set
        )

        print(f"Agent coboundary matrix (delta_q):\n{delta_q}")
        print(f"Target coboundary matrix (delta_T):\n{delta_T}")
       
        step = 0
        target_state = np.concatenate([target_entity.positions[:, step] for target_entity in target])

        agent_star = (-np.linalg.pinv(delta_q.T @ delta_q) @ delta_q.T @ delta_T @ target_state)
        residual = (delta_q.T @ delta_q @ agent_star + delta_q.T @ delta_T @ target_state)

        print(f"\nExact-extension check for topology {topology_num}:")
        print(f"Residual:\n{residual}")
        
config_path = Path(__file__).parent / "config_common.json"

# ---------------------------------------------------------------------

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