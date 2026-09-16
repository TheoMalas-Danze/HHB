"""
Two-mode SNAIL cat qubit -- the physics target.

Memory mode ``a`` (undamped) plus lossy buffer mode ``b``, with
``omega_b = 2*omega_a = omega_d`` and two-photon dissipation stabilising a cat
state in ``a``. 14 real degrees of freedom, exact to all orders in
``phi_a, phi_b`` (no small-amplitude Taylor truncation).

Equations, conventions and the bug registry live in ``CLAUDE.md``. Do not
modify the moment equations without checking them there first.
"""
