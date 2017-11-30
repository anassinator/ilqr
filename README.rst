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

You can set up your own dynamics model by either extending the
:code:`DynamicsModel` class and hard-coding it and its partial derivatives.
Alternatively, you can write it up as a `Theano` expression and use the
:code:`AutoDiffDynamicsModel` class for it to be auto-differentiated.

This is an example of the following dynamics model:

.. math::

  m \dot{v} = u - \alpha v

where :math:`m` is the object's mass in :math:`kg`, :math:`alpha` is the
friction coefficient, :math:`v` is the object's velocity in :math:`m/s`,
:math:`\dot{v}` is the object's acceleration in :math:`m/s^2`, and :math:`u` is
the control (or force) you're applying to the object in :math:`N`.

.. code-block:: python

  import theano.tensor as T
  from ilqr.dynamics import AutoDiffDynamics

  x = T.dscalar("x")  # Position.
  x_dot = T.dscalar("x_dot")  # Velocity.
  u = T.dscalar("u")  # Force.

  dt = 0.01  # Discrete time step in seconds.
  m = 1.0  # Mass in kg.
  alpha = 0.1  # Friction coefficient.

  # Acceleration.
  x_dot_dot = x_dot * (1 - alpha * dt / m) + u * dt / m

  # Discrete dynamics model definition.
  f = T.stack([
      x + x_dot * dt,
      x_dot + x_dot_dot * dt,
  ])

  x_inputs = [x, x_dot]  # State vector.
  u_inputs = [u]  # Control vector.

  # Compile the dynamics.
  # NOTE: This can be slow as it's computing the symbolic derivatives.
  # But that's okay since it's only a one-time cost on startup. You could save
  # a serialized version of this object to reuse on the next startup in order
  # to avoid incurring this cost everytime.
  dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)

You can then use this as follows:

.. code-block:: python

  # NOTE: This computes very quickly since the functions were compiled.
  curr_x = np.array([1.0, 2.0])
  curr_u = np.array([0.0])
  t = 0  # This dynamics model is not time-varying, so this doesn't matter.
  next_x = dynamics.f(curr_x, curr_u, t)
  d_dx = dynamics.f_x(curr_x, curr_u, t)

Cost function
^^^^^^^^^^^^^

Similarly, you can set up your own cost function by either extending the
:code:`Cost` class and hard-coding it and its partial derivatives.
Alternatively, you can write it up as a `Theano` expression and use the
:code:`AutoDiffCost` class for it to be auto-differentiated.

The most common cost function is the quadratic format used by Linear Quadratic Regulators:

.. math::

  (x - x_{goal})^T Q (x - x_{goal}) + (u - u_{goal})^T R (u - u_{goal})

where :math:`Q` and :math:`R` are matrices defining your quadratic state error
and quadratic control errors and :math:`x_{goal}` is your target state. An
implementation of this cost function is made available as the `QRCost` class
and can be used as follows:

.. code-block:: python

  import numpy as np
  from ilqr.cost import QRCost

  # The coefficients weigh how much your state error is worth to you vs
  # the size of your controls. You can favor a solution that uses smaller
  # controls by increasing R's coefficient.
  Q = 100 * np.eye(dynamics.state_size)
  R = 0.01 * np.eye(dynamics.action_size)

  # This is optional if you want your cost to be computed differently at a
  # terminal state.
  Q_terminal = np.array([[100.0, 0.0], [0.0, 0.1]])

  # State goal is set to a position of 1m with no velocity.
  x_goal = np.array([1.0, 0.0])

  cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)

You can then use this as follows:

.. code-block:: python

  instantaneous_cost = cost.l(curr_x, curr_u, t)
  d_dx = cost.l_x(curr_x, curr_u, t)

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  T = 1000  # Number of time steps in trajectory.
  x0 = np.array([0.0, -0.1])  # Initial state.
  us_init = np.random.uniform(-1, 1, (T, 1)) # Random initial action path.

  ilqr = iLQR(dynamics, cost, T)
  xs, us = ilqr.fit(x0, us_init)

:code:`xs` and :code:`us` now hold the optimal state and control trajectory
that reaches the desired goal state with minimum cost.

Finally, a :code:`RecedingHorizonController` is also bundled with this package
to use the :code:`iLQR` controller in Model Predictive Control.

Important notes
^^^^^^^^^^^^^^^

To quote from Tassa's paper: "Two important parameters which have a direct
impact on performance are the simulation time-step :code:`dt` and the horizon
length :code:`T`. Since speed is of the essence, the goal is to choose those
values which minimize the number of steps in the trajectory, i.e. the largest
possible time-step and the shortest possible horizon. The size of :code:`dt`
is limited by our use of Euler integration; beyond some value the simulation
becomes unstable. The minimum length of the horizon :code:`T` is a
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
