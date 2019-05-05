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
"""Cartpole example."""

import numpy as np
import theano.tensor as T
from ..dynamics import BatchAutoDiffDynamics, tensor_constrain


class CartpoleDynamics(BatchAutoDiffDynamics):

    """Cartpole auto-differentiated dynamics model."""

    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 mc=1.0,
                 mp=0.1,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Cartpole dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N].
            max_bounds: Maximum bounds for action [N].
            mc: Cart mass [kg].
            mp: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        def f(x, u, i):
            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)

            x_ = x[..., 0]
            x_dot = x[..., 1]
            sin_theta = x[..., 2]
            cos_theta = x[..., 3]
            theta_dot = x[..., 4]
            F = u[..., 0]

            # Define dynamics model as per Razvan V. Florian's
            # "Correct equations for the dynamics of the cart-pole system".
            # Friction is neglected.

            # Eq. (23)
            temp = (F + mp * l * theta_dot**2 * sin_theta) / (mc + mp)
            numerator = g * sin_theta - cos_theta * temp
            denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
            theta_dot_dot = numerator / denominator

            # Eq. (24)
            x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

            # Deaugment state for dynamics.
            theta = T.arctan2(sin_theta, cos_theta)
            next_theta = theta + theta_dot * dt

            return T.stack([
                x_ + x_dot * dt,
                x_dot + x_dot_dot * dt,
                T.sin(next_theta),
                T.cos(next_theta),
                theta_dot + theta_dot_dot * dt,
            ]).T

        super(CartpoleDynamics, self).__init__(f,
                                               state_size=5,
                                               action_size=1,
                                               **kwargs)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).

        In this case, it converts:

            [x, x', theta, theta'] -> [x, x', sin(theta), cos(theta), theta']

        Args:
            state: State vector [reducted_state_size].

        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            x, x_dot, theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            theta = state[..., 2].reshape(-1, 1)
            theta_dot = state[..., 3].reshape(-1, 1)

        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            sin_theta = state[..., 2].reshape(-1, 1)
            cos_theta = state[..., 3].reshape(-1, 1)
            theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])
