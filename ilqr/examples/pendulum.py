# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Inverted pendulum example."""

import numpy as np
import theano.tensor as T
from ..dynamics import BatchAutoDiffDynamics, tensor_constrain


class InvertedPendulumDynamics(BatchAutoDiffDynamics):

    """Inverted pendulum auto-differentiated dynamics model."""

    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 m=1.0,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Constructs an InvertedPendulumDynamics model.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.

        Note:
            state: [sin(theta), cos(theta), theta']
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        def f(x, u, i):
            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)

            sin_theta = x[..., 0]
            cos_theta = x[..., 1]
            theta_dot = x[..., 2]
            torque = u[..., 0]

            # Deal with angle wrap-around.
            theta = T.arctan2(sin_theta, cos_theta)

            # Define acceleration.
            theta_dot_dot = -3.0 * g / (2 * l) * T.sin(theta + np.pi)
            theta_dot_dot += 3.0 / (m * l**2) * torque

            next_theta = theta + theta_dot * dt

            return T.stack([
                T.sin(next_theta),
                T.cos(next_theta),
                theta_dot + theta_dot_dot * dt,
            ]).T

        super(InvertedPendulumDynamics, self).__init__(f,
                                                       state_size=3,
                                                       action_size=1,
                                                       **kwargs)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).

        In this case, it converts:

            [theta, theta'] -> [sin(theta), cos(theta), theta']

        Args:
            state: State vector [reducted_state_size].

        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            theta, theta_dot = state
        else:
            theta = state[..., 0].reshape(-1, 1)
            theta_dot = state[..., 1].reshape(-1, 1)

        return np.hstack([np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [sin(theta), cos(theta), theta'] -> [theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            sin_theta, cos_theta, theta_dot = state
        else:
            sin_theta = state[..., 0].reshape(-1, 1)
            cos_theta = state[..., 1].reshape(-1, 1)
            theta_dot = state[..., 2].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([theta, theta_dot])
