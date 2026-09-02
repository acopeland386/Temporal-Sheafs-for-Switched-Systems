from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def integrate_step(
    state: NDArray[np.float64],
    step: int,
    dt: float,
    derivative: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
) -> NDArray[np.float64]:

    t = (step - 1) * dt

    k1 = derivative(t, state)

    k2 = derivative(
        t + dt / 2,
        state + dt * k1 / 2
    )

    k3 = derivative(
        t + dt / 2,
        state + dt * k2 / 2
    )

    k4 = derivative(
        t + dt,
        state + dt * k3
    )

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)