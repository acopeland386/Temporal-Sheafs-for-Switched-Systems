from entity import Agent, Target
import numpy as np
from restriction_maps import (
    agent_agent_restriction_maps,
    agent_target_restriction_maps,
)

def agent_coboundary(
    agents: list[Agent],
    agent_edge_set: list[list[int]],
    target_edge_set: list[list[int]]
) -> np.ndarray:

    F_i_ij, F_j_ij = agent_agent_restriction_maps()
    F_i_iT, _ = agent_target_restriction_maps()

    num_agents = len(agents)
    state_dim = agents[0].num_states

    num_agent_edges = len(agent_edge_set)
    num_target_edges = len(target_edge_set)
    num_edges = num_agent_edges + num_target_edges

    delta_q = np.zeros(
        (
            num_edges * state_dim,
            num_agents * state_dim
        )
    )

    # Agent-agent edges-------------------------------

    for edge_index, (i, j) in enumerate(agent_edge_set):

        edge = (i, j)

        F_i = F_i_ij[edge]
        F_j = F_j_ij[edge]

        row_start = edge_index * state_dim
        row_end = row_start + state_dim

        col_i_start = (i - 1) * state_dim
        col_i_end = col_i_start + state_dim

        col_j_start = (j - 1) * state_dim
        col_j_end = col_j_start + state_dim

        delta_q[row_start:row_end, col_i_start:col_i_end] = F_i
        delta_q[row_start:row_end, col_j_start:col_j_end] = -F_j

    # Agent-target edges-------------------------------

    for target_edge_index, (i, T) in enumerate(target_edge_set):

        F_i = F_i_iT[i]

        edge_index = num_agent_edges + target_edge_index

        row_start = edge_index * state_dim
        row_end = row_start + state_dim

        col_i_start = (i - 1) * state_dim
        col_i_end = col_i_start + state_dim

        delta_q[row_start:row_end, col_i_start:col_i_end] = F_i

    return delta_q

def target_coboundary(
    agents: list[Agent],
    targets: list[Target],
    agent_edge_set: list[list[int]],
    target_edge_set: list[list[int]]
) -> np.ndarray:

    _, F_T_iT = agent_target_restriction_maps()

    state_dim = agents[0].num_states
    num_targets = len(targets)

    num_agent_edges = len(agent_edge_set)
    num_target_edges = len(target_edge_set)
    num_edges = num_agent_edges + num_target_edges

    delta_p = np.zeros(
        (
            num_edges * state_dim,
            num_targets * state_dim
        )
    )

    for target_edge_index, (i, T) in enumerate(target_edge_set):

        F_T = F_T_iT[i]

        edge_index = num_agent_edges + target_edge_index

        row_start = edge_index * state_dim
        row_end = row_start + state_dim

        col_T_start = (T - 1) * state_dim
        col_T_end = col_T_start + state_dim

        delta_p[row_start:row_end, col_T_start:col_T_end] = F_T

    return delta_p
