from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from collections.abc import Callable
from typing import Any, List

from simulation import dynamics
from simulation.integrate import integrate_step
from restriction_maps import (agent_agent_restriction_maps, agent_target_restriction_maps)

class Entity:
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        self.id: str = config['id']
        self.num_states: int = config['num_states']
        self.time_step_delta: float = config['time_step_delta']
        dynamics_type = config['dynamics_type']
        self.dynamics_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = dynamics.get_dynamics_function(dynamics_type)
        self.neighbors: List["Entity"] = []

        # coboudary matrices
        self.delta_q: NDArray[np.float64] = np.zeros((0, 0))
        self.delta_T: NDArray[np.float64] = np.zeros((0, 0))

        # true state initializations
        self.positions: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position # position state
        self.velocities: NDArray[np.float64] = np.zeros((self.num_states, time_steps)) # velocity state

        # observer state initializations
        self.observer: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.observer[:, 0] = initial_position # observer state
        self.observer_dot: NDArray[np.float64] = np.zeros((self.num_states, time_steps))

        # exact-extension state initialization
        self.agent_star: NDArray[np.float64] = np.zeros((self.num_states, time_steps)) # exact-extension state

        # error states initializations
        self.e: NDArray[np.float64] = np.zeros((self.num_states, time_steps)) # error representing the difference between the exact extension and the actual state of an agent
        self.e_hat: NDArray[np.float64] = np.zeros((self.num_states, time_steps)) # error representing the difference between the exact extension and the observer
        self.e_tilde: NDArray[np.float64] = np.zeros((self.num_states, time_steps)) # error representing the difference between the actual and observer

        # control output initialization
        self.control_output: NDArray[np.float64] = np.zeros(self.num_states)

        # disturbance term
        self.d: NDArray[np.float64] = np.sin(2 * np.pi * np.arange(time_steps) * self.time_step_delta) * 0.1

    # Update true dynamics, error states------------------------------

    def update_dynamics(self, step: int) -> None:
        def dynamics_with_control(t: float, pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.dynamics_function(pos) + self.control_output + self.d[step]
        
        # agent model is of the form \dot(x,t) = f(x,t) + u(t) + d(t)       
        self.velocities[:, step] = self.dynamics_function(self.positions[:, step - 1]) + self.control_output + self.d[step]
        result_true = integrate_step(self.positions[:, step - 1], 
                                step, 
                                self.time_step_delta, 
                                dynamics_with_control)
        
        self.positions[:, step] = result_true

    # Update observer dynamics------------------------------
    def update_observer(self, step: int) -> None:
        F_i_ij, F_j_ij = agent_agent_restriction_maps()
        F_i_iT, F_T_iT = agent_target_restriction_maps()
        
        def observer_dynamics(t: float, x_hat_i: NDArray[np.float64]) -> NDArray[np.float64]:

            # Agent-agent neighbors-------------------------------
            agent_term = np.zeros(self.num_states)

            for neighbor in self.neighbors:
                if isinstance(neighbor, Agent):
                    i = int(self.id[1:])
                    j = int(neighbor.id[1:])

                    if i < j:
                        edge = (i, j)
                        F_i = F_i_ij[edge]
                        F_j = F_j_ij[edge]

                    else:
                        edge = (j, i)
                        F_i = F_j_ij[edge]
                        F_j = F_i_ij[edge]

                    x_hat_j = neighbor.observer[:, step - 1]

                    agent_term += (F_i.T @ (F_j @ x_hat_j - F_i @ x_hat_i))

            # Agent-target neighbors-------------------------------
            target_term = np.zeros(self.num_states)
            i = int(self.id[1:])

            for target_index, pin_weight in enumerate(self.pin_row):
                if pin_weight != 0.0:
                    F_i = F_i_iT[i]
                    F_T = F_T_iT[i]
                    target_state = self.targets[target_index].positions[:, step - 1]
                    target_term += (pin_weight * F_i.T @ (F_T @ target_state - F_i @ x_hat_i))

            return self.k1 * (agent_term + target_term)

        # observer model is of the form \dot{\hat{x}}_i = k_1(agent_term + target_term)
        self.observer_dot[:, step] = observer_dynamics(step * self.time_step_delta, self.observer[:, step - 1])

        result_observer = integrate_step(
            self.observer[:, step - 1],
            step,
            self.time_step_delta,
            observer_dynamics
        )

        self.observer[:, step] = result_observer

        # update error states
        self.e[:, step] = self.agent_star[:, step] - self.positions[:, step]
        self.e_hat[:, step] = self.agent_star[:, step] - self.observer[:, step]
        self.e_tilde[:, step] = self.positions[:, step] - self.observer[:, step]

# ---------------------------------------------------------------------

class Agent(Entity):
    def __init__(
        self,
        initial_position: NDArray[np.float64],
        time_steps: int,
        config: dict[str, Any],
        targets: List["Target"],
        pin_row: NDArray[np.float64]
    ) -> None:
        
        super().__init__(initial_position, time_steps, config)
        self.pin_row: NDArray[np.float64] = pin_row
        self.k1: float = config['observer_gain']
        self.k2: float = config['controller_gain']
        self.targets: List["Target"] = targets

    def compute_control_output(self, step: int) -> None:
            self.control_output = self.k2 * (self.e_tilde[:, step - 1] + self.observer_dot[:, step])

# ---------------------------------------------------------------------

class Target(Entity):
    def __init__(self, 
                 initial_position: NDArray[np.float64], 
                 time_steps: int, 
                 config: dict[str, Any]) -> None:

        super().__init__(initial_position, time_steps, config)

    def compute_control_output(self) -> None:
            desired_velocity = np.zeros(self.num_states)
            self.control_output = desired_velocity

# ---------------------------------------------------------------------

def compute_exact_extension(
    agents: list[Agent],
    targets: list[Target],
    delta_q: NDArray[np.float64],
    delta_T: NDArray[np.float64],
    step: int
) -> None:

    target_state = np.concatenate([target.positions[:, step] for target in targets])
    agent_star = (-np.linalg.pinv(delta_q.T @ delta_q) @ delta_q.T @ delta_T @ target_state)

    for agent_index, agent in enumerate(agents):

        start = agent_index * agent.num_states
        end = start + agent.num_states
        agent.agent_star[:, step] = (agent_star[start:end])


