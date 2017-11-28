# -*- coding: utf-8 -*-
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


def jacobian_vector(expr, wrt):
    """Computes the Jacobian of a vector expression with respect to varaibles.

    Args:
        expr: Vector Theano tensor expression.
        wrt: List of Theano variables.

    Returns:
        Theano tensor.
    """
    return _tensor_map(lambda f: jacobian_scalar(f, wrt), expr)


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


def hessian_vector(expr, wrt):
    """Computes the Hessian of a vector expression with respect to varaibles.

    Args:
        expr: Vector Theano tensor expression.
        wrt: List of Theano variables.

    Returns:
        Theano tensor.
    """
    return _tensor_map(lambda f: hessian_scalar(f, wrt), expr)


def _tensor_map(f, expr):
    """Maps a function onto of a vector expression.

    Args:
        f: Function to apply.
        expr: Theano tensor expression.
        wrt: List of Theano variables.

    Returns:
        Theano tensor.
    """
    return T.stack([f(y) for y in expr])


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
