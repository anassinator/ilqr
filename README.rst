Iterative Linear Quadratic Regulator
====================================

.. image:: https://travis-ci.org/anassinator/ilqr.svg?branch=master
  :target: https://travis-ci.org/anassinator/ilqr

This is an implementation of the Iterative Linear Quadratic Regulator (iLQR)
for non-linear trajectory optimization based on Yuval Tassa's
`paper <https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf>`_.

It is compatible with both Python 2 and 3 and has built-in support for
auto-differentiating both the dynamics model and the cost function using
`Theano <http://deeplearning.net/software/theano/>`_.

Install
-------

To install, clone and run:

.. code-block:: bash

  python setup.py install

You may also install the dependencies with `pipenv` as follows:

.. code-block:: bash

  pipenv install

Usage
-----

After installing, :code:`import` as follows:

.. code-block:: python

  from ilqr import iLQR

You can see the `examples <examples/>`_ directory for
`Jupyter <https://jupyter.org>`_ notebooks to see how common control problems
can be solved through iLQR.

Dynamics model
^^^^^^^^^^^^^^

You can set up your own dynamics model by either extending the :code:`Dynamics`
class and hard-coding it and its partial derivatives. Alternatively, you can
write it up as a `Theano` expression and use the :code:`AutoDiffDynamics` class
for it to be auto-differentiated. Finally, if all you have is a function, you
can use the :code:`FiniteDiffDynamics` class to approximate the derivatives
with finite difference approximation.

This section demonstrates how to implement the following dynamics model:

.. math::

  m \dot{v} = F - \alpha v

where :math:`m` is the object's mass in :math:`kg`, :math:`alpha` is the
friction coefficient, :math:`v` is the object's velocity in :math:`m/s`,
:math:`\dot{v}` is the object's acceleration in :math:`m/s^2`, and :math:`F` is
the control (or force) you're applying to the object in :math:`N`.

Automatic differentiation
"""""""""""""""""""""""""

.. code-block:: python

  import theano.tensor as T
  from ilqr.dynamics import AutoDiffDynamics

  x = T.dscalar("x")  # Position.
  x_dot = T.dscalar("x_dot")  # Velocity.
  F = T.dscalar("F")  # Force.

  dt = 0.01  # Discrete time-step in seconds.
  m = 1.0  # Mass in kg.
  alpha = 0.1  # Friction coefficient.

  # Acceleration.
  x_dot_dot = x_dot * (1 - alpha * dt / m) + F * dt / m

  # Discrete dynamics model definition.
  f = T.stack([
      x + x_dot * dt,
      x_dot + x_dot_dot * dt,
  ])

  x_inputs = [x, x_dot]  # State vector.
  u_inputs = [F]  # Control vector.

  # Compile the dynamics.
  # NOTE: This can be slow as it's computing and compiling the derivatives.
  # But that's okay since it's only a one-time cost on startup. You could save
  # a serialized version of this object to reuse on the next startup in order
  # to avoid incurring this cost every time.
  dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)

*Note*: If you want to be able to use the Hessians (:code:`f_xx`, :code:`f_ux`,
and :code:`f_uu`), you need to pass the :code:`hessians=True` argument to the
constructor. This will increase compilation time. Note that :code:`iLQR` does
not require second-order derivatives to function.

Finite difference approximation
"""""""""""""""""""""""""""""""

.. code-block:: python

  from ilqr.dynamics import FiniteDiffDynamics

  state_size = 2  # [position, velocity]
  action_size = 1  # [force]

  dt = 0.01  # Discrete time-step in seconds.
  m = 1.0  # Mass in kg.
  alpha = 0.1  # Friction coefficient.

  def f(x, u, i):
      """Dynamics model function.

      Args:
          x: State vector [state_size].
          u: Control vector [action_size].
          i: Current time step.

      Returns:
          Next state vector [state_size].
      """
      [x, x_dot] = x
      [F] = u

      # Acceleration.
      x_dot_dot = x_dot * (1 - alpha * dt / m) + F * dt / m

      return np.array([
        x + x_dot * dt,
        x_dot + x_dot_dot * dt,
      ])

  # NOTE: Unlike with AutoDiffDynamics, this is instantaneous, but will not be
  # as accurate.
  dynamics = FiniteDiffDynamics(f, state_size, action_size)

*Note*: It is possible you might need to play with the epsilon values
(:code:`x_eps` and :code:`u_eps`) used when computing the approximation if you
run into numerical instability issues.

Usage
"""""

Regardless of the method used for constructing your dynamics model, you can use
them as follows:

.. code-block:: python

  curr_x = np.array([1.0, 2.0])
  curr_u = np.array([0.0])
  i = 0  # This dynamics model is not time-varying, so this doesn't matter.

  >>> dynamics.f(curr_x, curr_u, i)
  ... array([ 1.02   ,  2.01998])
  >>> dynamics.f_x(curr_x, curr_u, i)
  ... array([[ 1.     ,  0.01   ],
             [ 0.     ,  1.00999]])
  >>> dynamics.f_u(curr_x, curr_u, i)
  ... array([[ 0.    ],
             [ 0.0001]])

Comparing the output of the :code:`AutoDiffDynamics` and the
:code:`FiniteDiffDynamics` models should generally yield consistent results,
but the auto-differentiated method will always be more accurate. Generally, the
finite difference approximation will be faster unless you're also computing the
Hessians: in which case, Theano's compiled derivatives are more optimized.

Cost function
^^^^^^^^^^^^^

Similarly, you can set up your own cost function by either extending the
:code:`Cost` class and hard-coding it and its partial derivatives.
Alternatively, you can write it up as a `Theano` expression and use the
:code:`AutoDiffCost` class for it to be auto-differentiated. Finally, if all
you have are a loss functions, you can use the :code:`FiniteDiffCost` class to
approximate the derivatives with finite difference approximation.

The most common cost function is the quadratic format used by Linear Quadratic
Regulators:

.. math::

  (x - x_{goal})^T Q (x - x_{goal}) + (u - u_{goal})^T R (u - u_{goal})

where :math:`Q` and :math:`R` are matrices defining your quadratic state error
and quadratic control errors and :math:`x_{goal}` is your target state. For
convenience, an implementation of this cost function is made available as the
:code:`QRCost` class.

:code:`QRCost` class
""""""""""""""""""""

.. code-block:: python

  import numpy as np
  from ilqr.cost import QRCost

  # The coefficients weigh how much your state error is worth to you vs
  # the size of your controls. You can favor a solution that uses smaller
  # controls by increasing R's coefficient.
  Q = 100 * np.eye(state_size)
  R = 0.01 * np.eye(action_size)

  # This is optional if you want your cost to be computed differently at a
  # terminal state.
  Q_terminal = np.array([[100.0, 0.0], [0.0, 0.1]])

  # State goal is set to a position of 1 m with no velocity.
  x_goal = np.array([1.0, 0.0])

  # NOTE: This is instantaneous and completely accurate.
  cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

Automatic differentiation
"""""""""""""""""""""""""

.. code-block:: python

  import theano.tensor as T
  from ilqr.cost import AutoDiffCost

  x_inputs = [T.dscalar("x"), T.dscalar("x_dot")]
  u_inputs = [T.dscalar("F")]

  x = T.stack(x_inputs)
  u = T.stack(u_inputs)

  x_diff = x - x_goal
  l = x_diff.T.dot(Q).dot(x_diff) + u.T.dot(R).dot(u)
  l_terminal = x_diff.T.dot(Q_terminal).dot(x_diff)

  # Compile the cost.
  # NOTE: This can be slow as it's computing and compiling the derivatives.
  # But that's okay since it's only a one-time cost on startup. You could save
  # a serialized version of this object to reuse on the next startup in order
  # to avoid incurring this cost every time.
  cost = AutoDiffCost(l, l_terminal, x_inputs, u_inputs)

Finite difference approximation
"""""""""""""""""""""""""""""""

.. code-block:: python

  from ilqr.cost import FiniteDiffCost


  def l(x, u, i):
      """Instantaneous cost function.

      Args:
          x: State vector [state_size].
          u: Control vector [action_size].
          i: Current time step.

      Returns:
          Instantaneous cost [scalar].
      """
      x_diff = x - x_goal
      return x_diff.T.dot(Q).dot(x_diff) + u.T.dot(R).dot(u)


  def l_terminal(x, i):
      """Terminal cost function.

      Args:
          x: State vector [state_size].
          i: Current time step.

      Returns:
          Terminal cost [scalar].
      """
      x_diff = x - x_goal
      return x_diff.T.dot(Q_terminal).dot(x_diff)


  # NOTE: Unlike with AutoDiffCost, this is instantaneous, but will not be as
  # accurate.
  cost = FiniteDiffCost(l, l_terminal, state_size, action_size)

*Note*: It is possible you might need to play with the epsilon values
(:code:`x_eps` and :code:`u_eps`) used when computing the approximation if you
run into numerical instability issues.

Usage
"""""

Regardless of the method used for constructing your cost function, you can use
them as follows:

.. code-block:: python

  >>> cost.l(curr_x, curr_u, i)
  ... 400.0
  >>> cost.l_x(curr_x, curr_u, i)
  ... array([   0.,  400.])
  >>> cost.l_u(curr_x, curr_u, i)
  ... array([ 0.])
  >>> cost.l_xx(curr_x, curr_u, i)
  ... array([[ 200.,    0.],
             [   0.,  200.]])
  >>> cost.l_ux(curr_x, curr_u, i)
  ... array([[ 0.,  0.]])
  >>> cost.l_uu(curr_x, curr_u, i)
  ... array([[ 0.02]])

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  N = 1000  # Number of time-steps in trajectory.
  x0 = np.array([0.0, -0.1])  # Initial state.
  us_init = np.random.uniform(-1, 1, (N, 1)) # Random initial action path.

  ilqr = iLQR(dynamics, cost, N)
  xs, us = ilqr.fit(x0, us_init)

:code:`xs` and :code:`us` now hold the optimal state and control trajectory
that reaches the desired goal state with minimum cost.

Finally, a :code:`RecedingHorizonController` is also bundled with this package
to use the :code:`iLQR` controller in Model Predictive Control.

Important notes
^^^^^^^^^^^^^^^

To quote from Tassa's paper: "Two important parameters which have a direct
impact on performance are the simulation time-step :code:`dt` and the horizon
length :code:`N`. Since speed is of the essence, the goal is to choose those
values which minimize the number of steps in the trajectory, i.e. the largest
possible time-step and the shortest possible horizon. The size of :code:`dt`
is limited by our use of Euler integration; beyond some value the simulation
becomes unstable. The minimum length of the horizon :code:`N` is a
problem-dependent quantity which must be found by trial-and-error."

Contributing
------------

Contributions are welcome. Simply open an issue or pull request on the matter.

Linting
-------

We use `YAPF <https://github.com/google/yapf>`_ for all Python formatting
needs. You can auto-format your changes with the following command:

.. code-block:: bash

  yapf --recursive --in-place --parallel .

License
-------

See `LICENSE <LICENSE>`_.

Credits
-------

This implementation was partially based on Yuval Tassa's :code:`MATLAB`
`implementation <https://www.mathworks.com/matlabcentral/fileexchange/52069>`_,
and `navigator8972 <https://github.com/navigator8972>`_'s
`implementation <https://github.com/navigator8972/pylqr>`_.
