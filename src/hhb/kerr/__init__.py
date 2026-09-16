"""
Single-mode Kerr oscillator -- the numerical testbed.

Exists so that every claim about the moment closure can be checked against an
exact truncated-Fock Lindblad answer from ``dynamiqs``.

``hb_moments_kerr`` -> ``_3`` -> ``_n`` is a progression, not a set of
alternatives: ``_n`` at ``order=2`` reproduces ``hb_moments_kerr`` exactly and
at ``order=3`` reproduces ``_3`` exactly (both verified). The older two are
kept because the ``sweep_kerr_hellinger*.ipynb`` notebooks import them
directly.

``hb_moments_kerr_max_entr_n`` is retained as a record of a closure that does
NOT work here (the Kerr Q is leptokurtic at every chi > 0, max-ent is
necessarily platykurtic) -- see README section 5.

Modules are not imported here: the symbolic derivation runs at import time.
"""
