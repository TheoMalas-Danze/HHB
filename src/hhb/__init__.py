"""
HHB -- Husimi Harmonic Balance.

Periodic non-equilibrium steady states (NESS) of driven-dissipative bosonic
systems, without Fock-space truncation: Lindblad -> Husimi Q -> moment closure
-> harmonic balance.

Subpackages are deliberately NOT imported here. ``hhb.kerr.hb_moments_kerr_n``
re-derives its equations of motion symbolically at import time (0.2 s at
order 2, ~31 s at order 4), so ``import hhb`` must stay cheap. Import the
module you actually need:

    from hhb.kerr.hb_moments_kerr_n import KerrParams, MomentHillMethodN
    from hhb.cat.hhb_cat import SnailParams, HBSettingsSnail, SnailHillMethod
    from hhb.hill_method import CircuitParams, HillSettings, HillMethod
"""

__version__ = "0.1.0"
