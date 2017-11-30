# -*- coding: utf-8 -*-
"""Controllers."""

import six
import abc
import warnings
import numpy as np


@six.add_metaclass(abc.ABCMeta)
class BaseController():

    """Base trajectory optimizer controller."""

    @abc.abstractmethod
    def fit(self, x0, us_init, *args, **kwargs):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [T, action_size].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            Tuple of
                xs: optimal state path [T+1, state_size].
                us: optimal control path [T, action_size].
        """
        raise NotImplementedError


class iLQR(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost, T, max_reg=1e10, hessians=False):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            T: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost = cost
        self.T = T
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        super(iLQR, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [T, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [T+1, state_size].
                us: optimal control path [T, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        xs = self._forward_rollout(x0, us)

        J_opt = self._trajectory_cost(xs, us)

        converged = False
        for i in range(n_iterations):
            accepted = False

            try:
                k, K = self._backward_pass(xs, us)

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(i, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

    def _control(self, xs, us, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [T+1, state_size].
            us: Nominal control path [T, action_size].
            k: Feedforward gains [T, action_size].
            K: Feedback gains [T, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [T+1, state_size].
                us: control path [T, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for t in range(self.T):
            # Eq (12).
            # Applying alpha only on k[t] as in the paper for some reason
            # doesn't converge.
            us_new[t] = us[t] + alpha * (k[t] + K[t].dot(xs_new[t] - xs[t]))

            # Eq (8c).
            xs_new[t + 1] = self.dynamics.f(xs_new[t], us_new[t], t)

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [T+1, state_size].
            us: Control path [T, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.T)))
        return sum(J) + self.cost.l(xs[-1], None, self.T, terminal=True)

    def _forward_rollout(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [T, action_size].

        Returns:
            State path [T+1, state_size].
        """
        xs = np.array([x0])
        for t in range(self.T):
            x_new = self.dynamics.f(xs[-1], us[t], t)
            xs = np.append(xs, [x_new], axis=0)

        return xs

    def _backward_pass(self, xs, us):
        """Computes the feedforward and feedback gains k and K.

        Args:
            xs: State path [T+1, state_size].
            us: Control path [T, action_size].

        Returns:
            Tuple of
                k: feedforward gains [T, action_size].
                K: feedback gains [T, action_size, state_size].
        """
        V_x = self.cost.l_x(xs[-1], None, self.T, terminal=True)
        V_xx = self.cost.l_xx(xs[-1], None, self.T, terminal=True)

        k = [None] * self.T
        K = [None] * self.T

        for t in range(self.T - 1, -1, -1):
            x = xs[t]
            u = us[t]

            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(x, u, V_x, V_xx, t)
            Q_uu_inv = np.linalg.pinv(Q_uu)

            # Eq (6).
            k[t] = -Q_uu_inv.dot(Q_u)
            K[t] = -Q_uu_inv.dot(Q_ux)

            # Eq (11b).
            V_x = Q_x + K[t].T.dot(Q_uu).dot(k[t])
            V_x += K[t].T.dot(Q_u) + Q_ux.T.dot(k[t])

            # Eq (11c).
            V_xx = Q_xx + K[t].T.dot(Q_uu).dot(K[t])
            V_xx += K[t].T.dot(Q_ux) + Q_ux.T.dot(K[t])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self, x, u, V_x, V_xx, t):
        """Computes second order expansion.

        Args:
            x: State [state_size].
            u: Control [action_size].
            V_x: d/dx of the value function at the next time step [state_size].
            V_xx: d^2/dx^2 of the value function at the next time step
                [state_size, state_size].
            t: Current time step.

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        f_x = self.dynamics.f_x(x, u, t)
        f_u = self.dynamics.f_u(x, u, t)

        l_x = self.cost.l_x(x, u, t)
        l_u = self.cost.l_u(x, u, t)
        l_xx = self.cost.l_xx(x, u, t)
        l_ux = self.cost.l_ux(x, u, t)
        l_uu = self.cost.l_uu(x, u, t)

        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            f_xx = self.dynamics.f_xx(x, u, t)
            f_ux = self.dynamics.f_ux(x, u, t)
            f_uu = self.dynamics.f_uu(x, u, t)

            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu


class RecedingHorizonController(object):

    """Receding horizon controller."""

    def __init__(self, x0, controller):
        """Constructs a RecedingHorizonController.

        Args:
            x0: Initial state [state_size].
            controller: Controller to fit with.
        """
        self._x = x0
        self._controller = controller

    def set_state(self, x):
        """Sets the current state of the controller.

        Args:
            x: Current state [state_size].
        """
        self._x = x

    def control(self, us_init, step_size=1, *args, **kwargs):
        """Yields the optimal controls to run at every step as a receding
        horizon problem.

        Note: The first iteration will be slow, but the successive ones will be
        significantly faster.

        Note: This will automatically move the current controller's state to
        what the dynamics model believes will be the next state after applying
        the entire control path computed. Should you want to correct this state
        between iterations, simply use the `set_state()` method.

        Args:
            us_init: Initial control path [T, action_size].
            step_size: Number of steps between each controller fit. Default: 1.
                i.e. re-fit at every time step. You might need to increase this
                depending on how powerful your machine is in order to run this
                in real-time.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to `controller.fit()`.

        Yields:
            Tuple of
                xs: optimal state path [step_size+1, state_size].
                us: optimal control path [step_size, action_size].
        """
        action_size = self._controller.dynamics.action_size
        while True:
            xs, us = self._controller.fit(self._x, us_init, *args, **kwargs)
            self._x = xs[step_size]
            yield xs[:step_size + 1], us[:step_size]

            # Set up next action path seed by simply moving along the current
            # optimal path and appending random unoptimal values at the end.
            us_start = us[step_size:]
            us_end = np.random.uniform(-1, 1, (step_size, action_size))
            us_init = np.vstack([us_start, us_end])
