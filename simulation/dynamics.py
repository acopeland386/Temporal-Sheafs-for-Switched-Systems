from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------

def attitude_mrp(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rigid-body attitude kinematics in Modified Rodrigues Parameters.

    State vector
        r : np.ndarray, shape (3,)  -- Modified Rodrigues Parameters, unitless

    Returns
        r_dot : np.ndarray, shape (3,)  -- time derivative of r, 1/s
    """

    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the 3x3 skew-symmetric matrix of vector v (rad/s)."""
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    r: NDArray[np.float64] = state
    r_sq: float = float(np.dot(r, r))
    b_mat: NDArray[np.float64] = (1.0 - r_sq) * np.eye(3) + 2.0 * _skew(r) + 2.0 * np.outer(r, r)

    j_inertia: NDArray[np.float64] = np.diag([2.0, 1.2, 1.6])              # kg·m^2
    tau_body: NDArray[np.float64] = np.array([0.0, 0.15, 0.0])             # N·m
    omega_body: NDArray[np.float64] = np.linalg.inv(j_inertia) @ tau_body  # rad/s

    r_dot: NDArray[np.float64] = 0.5 * b_mat @ omega_body
    return r_dot

#---------------------------------------------------------------------

def planar_dynamics(
    state: NDArray[np.float64]
) -> NDArray[np.float64]:

    x = state[0]
    y = state[1]

    omega = 0.1

    x_dot = -omega * y
    y_dot = omega * x

    return np.array([
        x_dot,
        y_dot
    ], dtype=np.float64)

# ---------------------------------------------------------------------

def chua(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Dimensionless Chua double-scroll circuit.

    State vector
        x : capacitor voltage proxy, unitless
        y : capacitor voltage proxy, unitless
        z : inductor current proxy, unitless
    """

    x, y, z = state
    alpha: float = 15.6          # unitless
    beta: float = 28.0           # unitless
    m0: float = -1.143           # unitless
    m1: float = -0.714           # unitless

    g: float = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))

    x_dot: float = alpha * (y - x - g)
    y_dot: float = x - y + z
    z_dot: float = -beta * y
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)

#---------------------------------------------------------------------

def trophic_dynamics(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Three-tier ecological food chain.

    State vector
        H : prey (herbivore) population, individuals
        P : predator population, individuals
        T : top-predator population, individuals
    """

    h_pop, p_pop, t_pop = state
    r_h: float = 0.6               # 1/day
    k_cap: float = 100.0           # individuals
    a_hp: float = 0.02             # 1/(individual·day)
    a_pt: float = 0.01             # 1/(individual·day)
    d_p: float = 0.3               # 1/day
    d_t: float = 0.1               # 1/day

    h_dot: float = r_h * h_pop * (1.0 - h_pop / k_cap) - a_hp * h_pop * p_pop
    p_dot: float = -d_p * p_pop + a_hp * h_pop * p_pop - a_pt * p_pop * t_pop
    t_dot: float = -d_t * t_pop + a_pt * p_pop * t_pop
    return np.array([h_dot, p_dot, t_dot], dtype=np.float64)

# ---------------------------------------------------------------------

def desired_target_velocity(
    position: NDArray[np.float64]
) -> NDArray[np.float64]:

    x = position[0]
    y = position[1]

    center_x = 0.0
    center_y = 0.0

    omega = 0.1

    desired_velocity = np.array([
        -omega * (y - center_y),
         omega * (x - center_x),
         0.0
    ])

    return desired_velocity

# ---------------------------------------------------------------------

def get_dynamics_function(dynamics_type: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return the dynamics function associated with `dynamics_type`."""
    dynamics_map: Dict[str, Callable[..., NDArray[np.float64]]] = {
        "attitude_mrp": attitude_mrp,
        "chua": chua,
        "trophic_dynamics": trophic_dynamics,
        "trophic_agent": trophic_dynamics,     
        "trophic_target": trophic_dynamics,   
        "chua_agent": chua,
        "chua_target": chua,  
        "desired_target_velocity": desired_target_velocity,
        "planar_dynamics": planar_dynamics
    }
    return dynamics_map[dynamics_type]

# ---------------------------------------------------------------------

def get_initial_conditions(dynamics_type: str) -> List[float]:
    """Return a list of reasonable initial conditions for the chosen model."""
    initial_conditions_map: Dict[str, List[float]] = {
        "attitude_mrp": [5, 10, 20],               
        "trophic_dynamics": [40.0, 9.0, 2.0],    
        "trophic_agent": [2, -3, 5],               
        "trophic_target": [-2, 6, 10],
        "chua": [0.2, 5, 5],     
        "chua_agent": [0.2, 5, 5],
        "chua_target": [0.2, 5, 5],
        "desired_target_velocity": [5, 5, 5],    
        "planar_dynamics": [5, 5]
    }
   
    return initial_conditions_map[dynamics_type]
