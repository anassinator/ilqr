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
"""Autodifferentiation helper methods."""

import theano
import theano.tensor as T


def jacobian_scalar(expr, wrt):
    """Computes the Jacobian of a scalar expression with respect to varaibles.

    Args:
        expr: Scalar Theano tensor expression.
        wrt: List of Theano variables.

    Returns:
        Theano tensor.
    """
    J = T.grad(expr, wrt, disconnected_inputs="ignore")
    return J


def jacobian_vector(expr, wrt, size):
    """Computes the Jacobian of a vector expression with respect to varaibles.

    Args:
        expr: Vector Theano tensor expression.
        wrt: List of Theano variables.
        size: Vector size.

    Returns:
        Theano tensor.
    """
    return _tensor_map(lambda f: jacobian_scalar(f, wrt), expr, size)


def batch_jacobian(f, wrt, size=None, *args, **kwargs):
    """Computes the jacobian of f(x) w.r.t. x in parallel.

    Args:
        f: Symbolic function.
        x: Variables to differentiate with respect to.
        size: Expected vector size of f(x).
        *args: Additional positional arguments to pass to `f()`.
        **kwargs: Additional key-word arguments to pass to `f()`.

    Returns:
        Theano tensor.
    """
    if isinstance(wrt, T.TensorVariable):
        if size is None:
            y = f(wrt, *args, **kwargs).shape[-1]
        x_rep = T.tile(wrt, (size, 1))
        y_rep = f(x_rep, *args, **kwargs)
    else:
        if size is None:
            size = f(*wrt, *args, **kwargs).shape[-1]
        x_rep = [T.tile(x, (size, 1)) for x in wrt]
        y_rep = f(*x_rep, *args, **kwargs)

    J = T.grad(
        cost=None,
        wrt=x_rep,
        known_grads={y_rep: T.identity_like(y_rep)},
        disconnected_inputs="ignore",
    )
    return J


def hessian_scalar(expr, wrt):
    """Computes the Hessian of a scalar expression with respect to varaibles.

    Args:
        expr: Theano tensor expression.
        wrt: List of Theano variables.

    Returns:
        Theano tensor.
    """
    J = T.grad(expr, wrt, disconnected_inputs="ignore")
    Q = T.stack([T.grad(g, wrt, disconnected_inputs="ignore") for g in J])
    return Q


def hessian_vector(expr, wrt, size):
    """Computes the Hessian of a vector expression with respect to varaibles.

    Args:
        expr: Vector Theano tensor expression.
        wrt: List of Theano variables.
        size: Vector size.

    Returns:
        Theano tensor.
    """
    return _tensor_map(lambda x: hessian_scalar(x, wrt), expr, size)


def _tensor_map(f, expr, size):
    """Maps a function onto of a vector expression.

    Args:
        f: Function to apply.
        expr: Theano tensor expression.
        wrt: List of Theano variables.
        size: Vector size.

    Returns:
        Theano tensor.
    """
    return T.stack([f(expr[i]) for i in range(size)])


def as_function(expr, inputs, **kwargs):
    """Converts and optimizes a Theano expression into a function.

    Args:
        expr: Theano tensor expression.
        inputs: List of Theano variables to use as inputs.
        **kwargs: Additional key-word arguments to pass to `theano.function()`.

    Returns:
        A function.
    """
    return theano.function(inputs, expr, on_unused_input="ignore", **kwargs)
