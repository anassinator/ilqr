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
"""Dynamics model."""

import six
import abc
import theano
import numpy as np
import theano.tensor as T
from scipy.optimize import approx_fprime
from .autodiff import (as_function, batch_jacobian, hessian_vector,
                       jacobian_vector)


@six.add_metaclass(abc.ABCMeta)
class Dynamics():

    """Dynamics Model."""

    @property
    @abc.abstractmethod
    def state_size(self):
        """State size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_size(self):
        """Action size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        raise NotImplementedError

    @abc.abstractmethod
    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError


class AutoDiffDynamics(Dynamics):

    """Auto-differentiated Dynamics Model."""

    def __init__(self, f, x_inputs, u_inputs, i=None, hessians=False, **kwargs):
        """Constructs an AutoDiffDynamics model.

        Args:
            f: Vector Theano tensor expression.
            x_inputs: Theano state input variables.
            u_inputs: Theano action input variables.
            i: Theano tensor time step variable.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._tensor = f
        self._i = T.dscalar("i") if i is None else i

        non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        inputs = np.hstack([x_inputs, u_inputs, self._i]).tolist()
        self._x_inputs = x_inputs
        self._u_inputs = u_inputs
        self._inputs = inputs
        self._non_t_inputs = non_t_inputs

        x_dim = len(x_inputs)
        u_dim = len(u_inputs)
        self._state_size = x_dim
        self._action_size = u_dim

        self._J = jacobian_vector(f, non_t_inputs, x_dim)

        self._f = as_function(f, inputs, name="f", **kwargs)

        self._f_x = as_function(self._J[:, :x_dim],
                                inputs,
                                name="f_x",
                                **kwargs)
        self._f_u = as_function(self._J[:, x_dim:],
                                inputs,
                                name="f_u",
                                **kwargs)

        self._has_hessians = hessians
        if hessians:
            self._Q = hessian_vector(f, non_t_inputs, x_dim)
            self._f_xx = as_function(self._Q[:, :x_dim, :x_dim],
                                     inputs,
                                     name="f_xx",
                                     **kwargs)
            self._f_ux = as_function(self._Q[:, x_dim:, :x_dim],
                                     inputs,
                                     name="f_ux",
                                     **kwargs)
            self._f_uu = as_function(self._Q[:, x_dim:, x_dim:],
                                     inputs,
                                     name="f_uu",
                                     **kwargs)

        super(AutoDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    @property
    def tensor(self):
        """The dynamics model variable."""
        return self._tensor

    @property
    def x(self):
        """The state variables."""
        return self._x_inputs

    @property
    def u(self):
        """The control variables."""
        return self._u_inputs

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        z = np.hstack([x, u, i])
        return self._f(*z)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        z = np.hstack([x, u, i])
        return self._f_x(*z)

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        z = np.hstack([x, u, i])
        return self._f_u(*z)

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        z = np.hstack([x, u, i])
        return self._f_xx(*z)

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        z = np.hstack([x, u, i])
        return self._f_ux(*z)

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        z = np.hstack([x, u, i])
        return self._f_uu(*z)


class BatchAutoDiffDynamics(Dynamics):

    """Batch Auto-differentiated Dynamics Model.

    NOTE: This offers faster derivatives than AutoDiffDynamics if you can
          describe your dynamics model as a symbolic function that can take
          batches of inputs.

    NOTE: This does not currently support computing hessians.
    """

    def __init__(self, f, state_size, action_size, **kwargs):
        """Constructs a BatchAutoDiffDynamics model.

        Args:
            f: Symbolic function with the following signature:
                Args:
                    x: Batch of state variables.
                    u: Batch of action variables.
                    i: Batch of time step variables.
                Returns:
                    f: Batch of next state variables.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._fn = f
        self._state_size = state_size
        self._action_size = action_size

        self._x = x = T.dvector("x")
        self._u = u = T.dvector("u")
        self._i = i = T.dscalar("i")
        inputs = [x, u, i]

        x_rep_1 = T.tile(x, (1, 1))
        u_rep_1 = T.tile(u, (1, 1))
        i_rep_1 = T.tile(i, (1, 1))
        self._tensor = f(x_rep_1, u_rep_1, i_rep_1)
        self._f = as_function(self._tensor, inputs, name="f", **kwargs)

        self._J_x, self._J_u, _ = batch_jacobian(f, inputs, state_size)
        self._f_x = as_function(self._J_x, inputs, name="f_x", **kwargs)
        self._f_u = as_function(self._J_u, inputs, name="f_u", **kwargs)

        super(BatchAutoDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return False

    @property
    def tensor(self):
        """The dynamics model variable."""
        return self._tensor

    @property
    def x(self):
        """The state variables."""
        return self._x

    @property
    def u(self):
        """The control variables."""
        return self._u

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x, u, i)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        return self._f_x(x, u, i)

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        return self._f_u(x, u, i)

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError(
            "Hessians are not supported in BatchAutoDiffDynamics yet")


class FiniteDiffDynamics(Dynamics):

    """Finite difference approximated Dynamics Model."""

    def __init__(self, f, state_size, action_size, x_eps=None, u_eps=None):
        """Constructs an FiniteDiffDynamics model.

        Args:
            f: Function to approximate. Signature: (x, u, i) -> x.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._f = f
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(FiniteDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return True

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x, u, i)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        J = np.vstack([
            approx_fprime(x, lambda x: self._f(x, u, i)[m], self._x_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        J = np.vstack([
            approx_fprime(u, lambda u: self._f(x, u, i)[m], self._u_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_x(x, u, i)[m, n], eps)
                for n in range(self._state_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_u(x, u, i)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        eps = self._u_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(u, lambda u: self.f_u(x, u, i)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q


def constrain(u, min_bounds, max_bounds):
    """Constrains a control vector between given bounds through a squashing
    function.

    Args:
        u: Control vector [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector [action_size].
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * np.tanh(u) + mean


def tensor_constrain(u, min_bounds, max_bounds):
    """Constrains a control vector tensor variable between given bounds through
    a squashing function.

    This is implemented with Theano, so as to be auto-differentiable.

    Args:
        u: Control vector tensor variable [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector tensor variable [action_size].
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * T.tanh(u) + mean
