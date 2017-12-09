# -*- coding: utf-8 -*-
"""Instantaneous Cost Function."""

import six
import abc
import numpy as np
import theano.tensor as T
from scipy.optimize import approx_fprime
from .autodiff import as_function, hessian_scalar, jacobian_scalar


@six.add_metaclass(abc.ABCMeta)
class Cost():

    """Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and t, whereas
          the non-terminal cost can be a function of x, u and t.
    """

    @abc.abstractmethod
    def l(self, x, u, t, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_x(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_u(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_xx(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_ux(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_uu(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        raise NotImplementedError


class AutoDiffCost(Cost):

    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and t, whereas
          the non-terminal cost can be a function of x, u and t.
    """

    def __init__(self, l, l_terminal, x_inputs, u_inputs, t=None, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            x_inputs: Theano state input variables [state_size].
            u_inputs: Theano action input variables [action_size].
            t: Theano tensor time variable.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._t = T.dscalar("t") if t is None else t
        self._x_inputs = x_inputs
        self._u_inputs = u_inputs

        non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        inputs = np.hstack([x_inputs, u_inputs, self._t]).tolist()
        terminal_inputs = np.hstack([x_inputs, self._t]).tolist()

        x_dim = len(x_inputs)
        u_dim = len(u_inputs)

        self._state_size = x_dim
        self._action_size = u_dim

        self._J = jacobian_scalar(l, non_t_inputs)
        self._Q = hessian_scalar(l, non_t_inputs)

        self._l = as_function(l, inputs, name="l", **kwargs)

        self._l_x = as_function(self._J[:x_dim], inputs, name="l_x", **kwargs)
        self._l_u = as_function(self._J[x_dim:], inputs, name="l_u", **kwargs)

        self._l_xx = as_function(
            self._Q[:x_dim, :x_dim], inputs, name="l_xx", **kwargs)
        self._l_ux = as_function(
            self._Q[x_dim:, :x_dim], inputs, name="l_ux", **kwargs)
        self._l_uu = as_function(
            self._Q[x_dim:, x_dim:], inputs, name="l_uu", **kwargs)

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        self._J_terminal = jacobian_scalar(l_terminal, x_inputs)
        self._Q_terminal = hessian_scalar(l_terminal, x_inputs)

        self._l_terminal = as_function(
            l_terminal, terminal_inputs, name="l_term_xx", **kwargs)
        self._l_x_terminal = as_function(
            self._J_terminal[:x_dim],
            terminal_inputs,
            name="l_term_ux",
            **kwargs)
        self._l_xx_terminal = as_function(
            self._Q_terminal[:x_dim, :x_dim],
            terminal_inputs,
            name="l_term_xx",
            **kwargs)

        super(AutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x_inputs

    @property
    def u(self):
        """The control variables."""
        return self._u_inputs

    @property
    def t(self):
        """The time variable."""
        return self._t

    def l(self, x, u, t, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            z = np.hstack([x, t])
            return np.asscalar(self._l_terminal(*z))

        z = np.hstack([x, u, t])
        return np.asscalar(self._l(*z))

    def l_x(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            z = np.hstack([x, t])
            return np.array(self._l_x_terminal(*z))

        z = np.hstack([x, u, t])
        return np.array(self._l_x(*z))

    def l_u(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        z = np.hstack([x, u, t])
        return np.array(self._l_u(*z))

    def l_xx(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            z = np.hstack([x, t])
            return np.array(self._l_xx_terminal(*z))

        z = np.hstack([x, u, t])
        return np.array(self._l_xx(*z))

    def l_ux(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        z = np.hstack([x, u, t])
        return np.array(self._l_ux(*z))

    def l_uu(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        z = np.hstack([x, u, t])
        return np.array(self._l_uu(*z))


class FiniteDiffCost(Cost):

    """Finite difference approximated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and t, whereas
          the non-terminal cost can be a function of x, u and t.
    """

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None):
        """Constructs an FiniteDiffCost.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, t) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, t) -> scalar.
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
        self._l = l
        self._l_terminal = l_terminal
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(FiniteDiffCost, self).__init__()

    def l(self, x, u, t, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x, t)

        return self._l(x, u, t)

    def l_x(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x, t),
                                 self._x_eps)

        return approx_fprime(x, lambda x: self._l(x, u, t), self._x_eps)

    def l_u(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return approx_fprime(u, lambda u: self._l(x, u, t), self._u_eps)

    def l_xx(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_x(x, u, t, terminal)[i], eps)
            for i in range(self._state_size)
        ])
        return Q

    def l_ux(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_u(x, u, t)[i], eps)
            for i in range(self._action_size)
        ])
        return Q

    def l_uu(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess
        Q = np.vstack([
            approx_fprime(u, lambda u: self.l_u(x, u, t)[i], eps)
            for i in range(self._action_size)
        ])
        return Q


class QRCost(Cost):

    """Quadratic Regulator Instantaneous Cost."""

    def __init__(self, Q, R, Q_terminal=None, x_goal=None, u_goal=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
            x_goal: Goal state [state_size].
            u_goal: Goal control [action_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if x_goal is None:
            self.x_goal = np.zeros(Q.shape[0])
        else:
            self.x_goal = np.array(x_goal)

        if u_goal is None:
            self.u_goal = np.zeros(R.shape[0])
        else:
            self.u_goal = np.array(u_goal)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert self.Q.shape[0] == self.x_goal.shape[0], "Q & x_goal mismatch"
        assert self.R.shape[0] == self.u_goal.shape[0], "R & u_goal mismatch"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(QRCost, self).__init__()

    def l(self, x, u, t, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_goal
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_goal
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_goal)

        u_diff = u - self.u_goal
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T


class PathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Q, R, x_path, u_path=None, Q_terminal=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [T+1, state_size].
            u_path: Goal control path [T, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_path = np.array(x_path)

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros(path_length - 1, action_size)
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
            "x_path must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(PathQRCost, self).__init__()

    def l(self, x, u, t, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_path[t]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[t]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_path[t]
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, t, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[t]
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, t, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            t: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T
