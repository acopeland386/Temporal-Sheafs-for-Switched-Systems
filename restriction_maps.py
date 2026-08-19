import numpy as np

def agent_agent_restriction_maps():

    """
    Restriction maps stay consistent across all topologies.
    If edge (i,j) appears in multiple topologies, it always uses
    the same pair (F_i_ij, F_j_ij).
    """

    F_i_ij = {}
    F_j_ij = {}

    # Edge (1, 2)--------------------------------------

    F_i_ij[(1, 2)] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    F_j_ij[(1, 2)] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    # Edge (2, 3)--------------------------------------

    F_i_ij[(2, 3)] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    F_j_ij[(2, 3)] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    # Edge (3, 4)--------------------------------------

    F_i_ij[(3, 4)] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    F_j_ij[(3, 4)] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    # Edge (4, 5)--------------------------------------

    F_i_ij[(4, 5)] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    F_j_ij[(4, 5)] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return F_i_ij, F_j_ij


def agent_target_restriction_maps():

    """
    Restriction maps stay consistent across all topologies.
    If edge (i,T) appears in multiple topologies, it always uses
    the same pair (F_i_iT, F_j_iT).
    """

    F_i_iT = {}
    F_T_iT = {}

    # Edge (1, T)--------------------------------------

    F_i_iT[1] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    F_T_iT[1] = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # Edge (2, T)--------------------------------------

    F_i_iT[2] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    F_T_iT[2] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    # Edge (3, T)--------------------------------------

    F_i_iT[3] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    F_T_iT[3] = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    # Edge (4, T)--------------------------------------

    F_i_iT[4] = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    F_T_iT[4] = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    # Edge (5, T)--------------------------------------

    F_i_iT[5] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    F_T_iT[5] = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return F_i_iT, F_T_iT
