from entity import Agent, Target
import numpy as np
from restriction_maps import (
    agent_agent_restriction_maps_1,
    agent_target_restriction_maps_1,
    agent_agent_restriction_maps_2,
    agent_target_restriction_maps_2
)


# for trivial restriction maps
def agent_coboundary_1(
    agents: list[Agent],
    agent_edge_set: list[list[int]],
    agent_target_edge_set: list[list[int]]
) -> np.ndarray:

    F_i_ij, F_j_ij = agent_agent_restriction_maps_1()
    F_i_iT, _ = agent_target_restriction_maps_1()

    num_agents = len(agents)
    state_dim = agents[0].num_states

    num_agent_edges = len(agent_edge_set)
    num_target_edges = len(agent_target_edge_set)
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
    for target_edge_index, (i, T) in enumerate(agent_target_edge_set):

        F_i = F_i_iT[i]

        edge_index = num_agent_edges + target_edge_index

        row_start = edge_index * state_dim
        row_end = row_start + state_dim

        col_i_start = (i - 1) * state_dim
        col_i_end = col_i_start + state_dim

        delta_q[row_start:row_end, col_i_start:col_i_end] = F_i

    return delta_q

#----------------------------------------------------------------

def target_coboundary_1(
    agents: list[Agent],
    targets: list[Target],
    agent_edge_set: list[list[int]],
    agent_target_edge_set: list[list[int]]
) -> np.ndarray:

    _, F_T_iT = agent_target_restriction_maps_1()

    state_dim = agents[0].num_states
    num_targets = len(targets)

    num_agent_edges = len(agent_edge_set)
    num_target_edges = len(agent_target_edge_set)
    num_edges = num_agent_edges + num_target_edges

    delta_p = np.zeros(
        (
            num_edges * state_dim,
            num_targets * state_dim
        )
    )

    for target_edge_index, (i, T) in enumerate(agent_target_edge_set):

        F_T = F_T_iT[i]

        edge_index = num_agent_edges + target_edge_index

        row_start = edge_index * state_dim
        row_end = row_start + state_dim

        col_T_start = (T - 1) * state_dim
        col_T_end = col_T_start + state_dim

        delta_p[row_start:row_end, col_T_start:col_T_end] = -F_T

    return delta_p

#------------------------------------------------

# for heterogeneous restriction maps
def agent_coboundary_2(
    agents: list[Agent],
    agent_edge_set: list[list[int]],
    agent_target_edge_set: list[list[int]]
) -> np.ndarray:

    F_i_ij, F_j_ij = agent_agent_restriction_maps_2()
    F_i_iT, _ = agent_target_restriction_maps_2()

    total_agent_state_dim = sum(
        agent.num_states
        for agent in agents
    )

    total_edge_dim = 0
    for i, j in agent_edge_set:
        edge = (i, j)
        F_i = F_i_ij[edge]
        total_edge_dim += F_i.shape[0]

    for i, T in agent_target_edge_set:
        F_i = F_i_iT[i]
        total_edge_dim += F_i.shape[0]

    delta_q = np.zeros((total_edge_dim, total_agent_state_dim))
    agent_column_offsets = [0]

    for agent in agents:
        agent_column_offsets.append(agent_column_offsets[-1] + agent.num_states)

    row_start = 0

    # Agent-agent edges-------------------------------
    for i, j in agent_edge_set:

        edge = (i, j)

        F_i = F_i_ij[edge]
        F_j = F_j_ij[edge]

        edge_dim = F_i.shape[0]

        row_end = row_start + edge_dim

        col_i_start = agent_column_offsets[i - 1]
        col_i_end = agent_column_offsets[i]

        col_j_start = agent_column_offsets[j - 1]
        col_j_end = agent_column_offsets[j]

        delta_q[row_start:row_end, col_i_start:col_i_end] = F_i
        delta_q[row_start:row_end, col_j_start:col_j_end] = -F_j

        row_start = row_end

    # Agent-target edges-------------------------------
    for i, T in agent_target_edge_set:
        F_i = F_i_iT[i]

        edge_dim = F_i.shape[0]
        row_end = row_start + edge_dim
        col_i_start = agent_column_offsets[i - 1]
        col_i_end = agent_column_offsets[i]

        delta_q[row_start:row_end, col_i_start:col_i_end] = F_i

        row_start = row_end

    return delta_q

#----------------------------------------------------------------

def target_coboundary_2(
    agents: list[Agent],
    targets: list[Target],
    agent_edge_set: list[list[int]],
    agent_target_edge_set: list[list[int]]
) -> np.ndarray:

    F_i_ij, _ = agent_agent_restriction_maps_2()
    F_i_iT, F_T_iT = agent_target_restriction_maps_2()

    total_target_state_dim = sum(
        target.num_states
        for target in targets
    )

    total_edge_dim = 0

    for i, j in agent_edge_set:
        edge = (i, j)
        F_i = F_i_ij[edge]
        total_edge_dim += F_i.shape[0]

    for i, T in agent_target_edge_set:
        F_i = F_i_iT[i]
        total_edge_dim += F_i.shape[0]

    delta_p = np.zeros((total_edge_dim, total_target_state_dim))
    target_column_offsets = [0]

    for target in targets:
        target_column_offsets.append(target_column_offsets[-1] + target.num_states)

    row_start = 0

    # Agent-agent edges-------------------------------
    for i, j in agent_edge_set:
        edge = (i, j)
        F_i = F_i_ij[edge]
        edge_dim = F_i.shape[0]
        row_start += edge_dim

    # Agent-target edges-------------------------------
    for i, T in agent_target_edge_set:
        F_T = F_T_iT[i]
        edge_dim = F_T.shape[0]
        row_end = row_start + edge_dim
        col_T_start = target_column_offsets[T - 1]
        col_T_end = target_column_offsets[T]
        delta_p[row_start:row_end, col_T_start:col_T_end] = -F_T

        row_start = row_end

    return delta_p



