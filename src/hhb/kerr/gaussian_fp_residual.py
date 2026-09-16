"""
Residual test of the Gaussian Husimi ansatz for the driven Kerr oscillator.

Given a converged harmonic-balance (HB) solution for the Gaussian moments, how
badly does the reconstructed Gaussian ``Q`` fail to satisfy the *exact* Husimi
equation?  This is a stronger test than any moment-level check: it probes the
whole phase-space PDE, not just its first two moments.

Exact equation (same convention as ``hb_moments_kerr_n.py``, which integrates
this PDE against ``a^m a*^n`` to get its moment recursion)::

    dQ/dt = d_a(A Q) + d_as(A* Q) + kappa d_a d_as Q
            + (i chi / 2) [ alpha^2 d_a^2 Q - alpha*^2 d_as^2 Q ]

    A(alpha, alpha*, t) = i Delta alpha + i (eps/2)(1 + e^{2 i wd t})
                          + i chi alpha^2 alpha* + (kappa/2) alpha

with ``d_a, d_as`` Wirtinger derivatives (``alpha`` and ``alpha*`` independent).

The ansatz is the one ``Q_hb_grid`` uses in the sweeps/notebooks::

    Z = alpha - mu,  Zb = alpha* - mu*,  Dd = sig^2 - |sigt|^2
    Q = 1/(pi sqrt(Dd)) exp[ (-sig Z Zb + (sigt* Z^2 + sigt Zb^2)/2) / Dd ]

``sig = sigma^2 = <da da^dag>`` (real, anti-normal, -> 1 in vacuum) and
``sigt = <da^2>``; under this Q, ``<Z Zb> = sig``, ``<Z^2> = sigt``.

Why this is cheap
-----------------
``log Q`` is quadratic in ``(Z, Zb)``, so every term on both sides is ``Q``
times a polynomial and the exponential cancels exactly::

    R(alpha, alpha*, t) = dQ/dt|_ansatz - RHS[Q_ansatz] = Q(...) * P(Z, Zb, t)

``P`` has total degree 4.  It is derived symbolically once (~1 s) and cached;
evaluation is then one small coefficient vector per time sample.  No
phase-space grid is ever built.

What survives, and what "degree >= 3" really means
--------------------------------------------------
The moment equations make ``R`` orthogonal to ``1, Z, Zb, Z^2, Z Zb, Zb^2``,
i.e. the six integrals ``<Z^p Zb^q P>_Q`` vanish for ``p + q <= 2``.  They do
**not** make the degree <= 2 *monomial coefficients* of ``P`` vanish: the
Gaussian average of a degree-4 monomial has a degree-0 part, so the low-order
coefficients are exactly the counterterms that cancel it.  (Only at ``chi = 0``,
where ``P`` has no degree >= 3 content at all, does every coefficient vanish
separately.)

The clean statement is therefore: in the Hermite basis adapted to the Gaussian,
``P`` has no degree <= 2 content.  Since a Hermite polynomial is its monomial
plus lower-degree corrections, the degree 3 and 4 *monomial* coefficients of
``P`` are already the degree 3 and 4 Hermite coefficients — those nine numbers
are the physics.  :func:`strip_low_order` removes the (numerically small)
degree <= 2 Hermite content left over by the HB truncation, by projecting it
out with the Gaussian Gram matrix; that is what the norms use by default.

Scalar norms are computed from Gaussian moments of ``P``, analytically for
``L2`` (``Q^2`` is again a Gaussian, with half the covariance) and by
Gauss-Hermite quadrature in whitened coordinates for ``L1`` — ``|P|`` is not a
polynomial, so the recipe's "no quadrature" is only achievable for ``L2``.  The
quadrature is over the Gaussian weight, not a phase-space grid: 64^2 nodes per
time sample, good to ~0.1-0.5% relative.

Usage
-----
    from hhb.kerr.hb_moments_kerr import KerrParams, HBSettingsMoments, MomentHillMethod
    from hhb.kerr.gaussian_fp_residual import moments_from_fourier, residual_report

    hb = MomentHillMethod(KerrParams(...), HBSettingsMoments(...))
    z = hb.solve(z0)
    m = moments_from_fourier(z, N_H=hb.N_H, omega_d=hb.p.omega_d, nu=hb.s.nu)
    rep = residual_report(m, Delta=..., kappa=..., eps=..., chi=..., wd=...)

Self-test:  python3.11 -m hhb.kerr.gaussian_fp_residual
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial

import numpy as np
import sympy as sp

__all__ = [
    'MONOMIAL_ARGS', 'LOW_PROBES', 'LAYOUT_KERR5', 'LAYOUT_GRADED',
    'build_residual_polynomial', 'lambdify_coefficients', 'chi_zero_odes',
    'moment_odes', 'moments_from_fourier', 'layout_from_hb',
    'residual_coeffs', 'gauss_moments', 'moment_projections',
    'strip_low_order', 'residual_norms', 'residual_report',
]


# ======================================================================
# Symbolic layer
# ======================================================================

#: argument order of every lambdified coefficient
MONOMIAL_ARGS = ('mu', 'mus', 's', 'st', 'sts',
                 'dmu', 'dmus', 'ds', 'dst', 'dsts',
                 'Delta', 'kappa', 'eps', 'chi', 'wd', 't')

#: the monomials the moment equations make R orthogonal to
LOW_PROBES = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


@dataclass(frozen=True)
class _Symbols:
    al: sp.Symbol
    als: sp.Symbol
    mu: sp.Symbol
    mus: sp.Symbol
    s: sp.Symbol
    st: sp.Symbol
    sts: sp.Symbol
    dmu: sp.Symbol
    dmus: sp.Symbol
    ds: sp.Symbol
    dst: sp.Symbol
    dsts: sp.Symbol
    Delta: sp.Symbol
    kappa: sp.Symbol
    eps: sp.Symbol
    chi: sp.Symbol
    wd: sp.Symbol
    t: sp.Symbol
    Z: sp.Symbol
    Zb: sp.Symbol

    @property
    def args(self):
        return tuple(getattr(self, name) for name in MONOMIAL_ARGS)


def _make_symbols() -> _Symbols:
    # CRITICAL: al/als, mu/mus, st/sts are INDEPENDENT non-real symbols.  If
    # sympy knows they are conjugates, sp.diff(..., al) picks up spurious
    # d_as contributions and the chi = 0 identity fails for bookkeeping
    # reasons alone.  Conjugation is imposed numerically, at evaluation time.
    al, als, mu, mus, st, sts = sp.symbols('al als mu mus st sts')
    dmu, dmus, dst, dsts = sp.symbols('dmu dmus dst dsts')
    s = sp.Symbol('s', positive=True)
    ds = sp.Symbol('ds', real=True)
    Delta, kappa, eps, chi, wd, t = sp.symbols(
        'Delta kappa eps chi wd t', real=True)
    Z, Zb = sp.symbols('Z Zb')
    return _Symbols(al, als, mu, mus, s, st, sts, dmu, dmus, ds, dst, dsts,
                    Delta, kappa, eps, chi, wd, t, Z, Zb)


@dataclass(frozen=True)
class ResidualPolynomials:
    """Output of :func:`build_residual_polynomial`."""
    P: sp.Poly          #: residual / Q, in (Z, Zb) — total degree 4
    P_lhs: sp.Poly      #: (dQ/dt)|_ansatz / Q, in (Z, Zb) — total degree 2
    sym: _Symbols
    Q: sp.Expr


@lru_cache(maxsize=1)
def build_residual_polynomial() -> ResidualPolynomials:
    """Derive ``P = R / Q`` symbolically.  Cached — runs once per process."""
    y = _make_symbols()

    Z = y.al - y.mu
    Zb = y.als - y.mus
    Dd = y.s**2 - y.st * y.sts
    # (sts*Z**2 + st*Zb**2)/2 is Re(conj(sigt) Z^2) analytically continued to
    # independent alpha, alpha*.  sp.re() here would break Wirtinger diff.
    Q = sp.exp((-y.s * Z * Zb + (y.sts * Z**2 + y.st * Zb**2) / 2) / Dd) \
        / (sp.pi * sp.sqrt(Dd))

    # ---- LHS: chain rule through the five moment parameters ----
    params = [(y.mu, y.dmu), (y.mus, y.dmus), (y.s, y.ds),
              (y.st, y.dst), (y.sts, y.dsts)]
    Qdot = sum(sp.diff(Q, p) * dp for p, dp in params)

    # ---- RHS: the exact Husimi equation ----
    A = (sp.I * y.Delta * y.al
         + sp.I * y.eps / 2 * (1 + sp.exp(2 * sp.I * y.wd * y.t))
         + sp.I * y.chi * y.al**2 * y.als
         + y.kappa * y.al / 2)
    # analytic conjugate: i -> -i, al <-> als, mu <-> mus, st <-> sts; the
    # kappa term keeps its sign (kappa*al/2 -> +kappa*als/2)
    As = (-sp.I * y.Delta * y.als
          - sp.I * y.eps / 2 * (1 + sp.exp(-2 * sp.I * y.wd * y.t))
          - sp.I * y.chi * y.als**2 * y.al
          + y.kappa * y.als / 2)

    RHS = (sp.diff(A * Q, y.al) + sp.diff(As * Q, y.als)
           + y.kappa * sp.diff(Q, y.al, 1, y.als, 1)
           + sp.I * y.chi / 2 * (y.al**2 * sp.diff(Q, y.al, 2)
                                 - y.als**2 * sp.diff(Q, y.als, 2)))

    def _to_poly(expr, name):
        # dividing by Q cancels exp(G) exactly (sympy folds exp(G)*exp(-G))
        e = sp.expand(expr / Q).subs({y.al: y.Z + y.mu, y.als: y.Zb + y.mus})
        e = sp.expand(e)
        if e.has(y.al) or e.has(y.als):
            raise RuntimeError(f"{name}: alpha survived the Z substitution")
        try:
            poly = sp.Poly(e, y.Z, y.Zb)
        except sp.PolynomialError as exc:      # pragma: no cover
            raise RuntimeError(
                f"{name}: not polynomial in (Z, Zb) — the exponential did not "
                f"cancel, something upstream is wrong") from exc
        return poly

    P = _to_poly(Qdot - RHS, 'P')
    P_lhs = _to_poly(Qdot, 'P_lhs')
    if P.total_degree() != 4:                  # pragma: no cover
        raise RuntimeError(f"P should have total degree 4, got {P.total_degree()}")
    if P_lhs.total_degree() != 2:              # pragma: no cover
        raise RuntimeError(f"P_lhs should have total degree 2, "
                           f"got {P_lhs.total_degree()}")
    return ResidualPolynomials(P=P, P_lhs=P_lhs, sym=y, Q=Q)


@lru_cache(maxsize=2)
def lambdify_coefficients(which: str = 'P'):
    """``[(m, n, callable), ...]`` — the coefficient of ``Z^m Zb^n``.

    Each callable takes :data:`MONOMIAL_ARGS`, numpy-broadcasting.
    ``which`` is ``'P'`` (residual) or ``'P_lhs'`` (dQ/dt of the ansatz).
    """
    rp = build_residual_polynomial()
    poly = {'P': rp.P, 'P_lhs': rp.P_lhs}[which]
    args = rp.sym.args
    out = []
    for (m, n), c in sorted(poly.terms()):
        out.append((m, n, sp.lambdify(args, c, modules='numpy')))
    return tuple(out)


def moment_odes(chi_zero: bool = False) -> dict:
    """The Gaussian-closure moment ODEs, as a sympy substitution dict.

    Keys are the ``dmu, dmus, ds, dst, dsts`` symbols of the cached derivation,
    so this plugs straight into ``P``.  These are exactly the equations
    ``hb_moments_kerr.py`` integrates (and, at order 2, the ones
    ``hb_moments_kerr_n.py`` derives); with ``chi_zero`` the chi = 0 limit.
    """
    y = build_residual_polynomial().sym
    I, mu, mus, s, st, sts = sp.I, y.mu, y.mus, y.s, y.st, y.sts
    chi = sp.Integer(0) if chi_zero else y.chi
    E = sp.exp(2 * I * y.wd * y.t)
    Eb = sp.exp(-2 * I * y.wd * y.t)
    return {
        y.dmu: (-I * y.eps / 2 * (1 + E)
                - (I * y.Delta + y.kappa / 2 - 2 * I * chi) * mu
                - I * chi * (mu**2 * mus + 2 * mu * s + mus * st)),
        y.dmus: (I * y.eps / 2 * (1 + Eb)
                 - (-I * y.Delta + y.kappa / 2 + 2 * I * chi) * mus
                 + I * chi * (mus**2 * mu + 2 * mus * s + mu * sts)),
        # -2 chi Im[mus^2 st] continued to independent symbols
        y.ds: y.kappa * (1 - s) + I * chi * (mus**2 * st - mu**2 * sts),
        y.dst: (-(2 * I * y.Delta + y.kappa - 5 * I * chi
                  + 4 * I * chi * mu * mus + 6 * I * chi * s) * st
                + I * chi * mu**2 * (1 - 2 * s)),
        y.dsts: (-(-2 * I * y.Delta + y.kappa + 5 * I * chi
                   - 4 * I * chi * mu * mus - 6 * I * chi * s) * sts
                 - I * chi * mus**2 * (1 - 2 * s)),
    }


def chi_zero_odes() -> dict:
    """``moment_odes(chi_zero=True)`` — kept for readability at call sites."""
    return moment_odes(chi_zero=True)


# ======================================================================
# Fourier -> moments and their exact derivatives
# ======================================================================

#: DOF layout of hb_moments_kerr.py / hb_moments_kerr_3.py (sigma^2 at index 2)
LAYOUT_KERR5 = {'mu': (0, 1), 'sig2': (2,), 'ts2': (3, 4)}
#: DOF layout of hb_moments_kerr_n.py (graded: mean, then k20, then k11)
LAYOUT_GRADED = {'mu': (0, 1), 'ts2': (2, 3), 'sig2': (4,)}


def layout_from_hb(hb) -> dict:
    """Pick the DOF layout for a solver instance (duck-typed on ``index_map``)."""
    imap = getattr(hb, 'index_map', None)
    if imap is None:
        return LAYOUT_KERR5
    return {'mu': tuple(imap['mu']), 'sig2': tuple(imap['k11']),
            'ts2': tuple(imap['k20'])}


def moments_from_fourier(z, N_H, omega_d, nu=1, n_times=256,
                         layout=LAYOUT_KERR5, t=None):
    """Moments and their exact time derivatives over one full period.

    ``z`` is in the repo's Fourier convention
    ``[c0, s1, c1, ..., s_NH, c_NH]`` with basis
    ``[1/sqrt2, sin(k wd t/nu), cos(k wd t/nu)]``.

    The derivatives are the analytic derivative of the Fourier series.  Do not
    replace them by finite differences: that pollutes the chi = 0 zero test at
    the 1e-6 level instead of 1e-15 and destroys the diagnostic's sharpness.

    Returns a dict with ``t, mu, mus, s, st, sts, dmu, dmus, ds, dst, dsts``
    (arrays of length ``n_times``), plus ``T`` (the period).
    """
    z = np.asarray(z, dtype=float)
    n_blocks = 2 * N_H + 1
    if z.size % n_blocks:
        raise ValueError(f"len(z)={z.size} is not a multiple of 2*N_H+1={n_blocks}")
    n_dof = z.size // n_blocks
    Zc = z.reshape(n_blocks, n_dof)

    T = 2 * np.pi * nu / omega_d
    if t is None:
        t = np.linspace(0.0, T, n_times, endpoint=False)
    t = np.atleast_1d(np.asarray(t, dtype=float))

    f = np.tile(Zc[0] / np.sqrt(2.0), (t.size, 1))          # (n_t, n_dof)
    fdot = np.zeros_like(f)
    for k in range(1, N_H + 1):
        wk = k * omega_d / nu
        sk, ck = Zc[1 + 2 * (k - 1)], Zc[2 + 2 * (k - 1)]
        S, C = np.sin(wk * t)[:, None], np.cos(wk * t)[:, None]
        f += sk * S + ck * C
        fdot += wk * (sk * C - ck * S)

    def _cplx(key):
        idx = layout[key]
        if len(idx) == 1:
            return f[:, idx[0]] + 0j, fdot[:, idx[0]] + 0j
        return (f[:, idx[0]] + 1j * f[:, idx[1]],
                fdot[:, idx[0]] + 1j * fdot[:, idx[1]])

    mu, dmu = _cplx('mu')
    sig, dsig = _cplx('sig2')
    st, dst = _cplx('ts2')
    s, ds = sig.real.copy(), dsig.real.copy()

    return {'t': t, 'T': T,
            'mu': mu, 'mus': mu.conj(), 's': s, 'st': st, 'sts': st.conj(),
            'dmu': dmu, 'dmus': dmu.conj(), 'ds': ds,
            'dst': dst, 'dsts': dst.conj()}


# ======================================================================
# Numeric evaluation
# ======================================================================

def _eval_args(moments, Delta, kappa, eps, chi, wd):
    shape = np.shape(moments['t'])
    scalars = {'Delta': Delta, 'kappa': kappa, 'eps': eps, 'chi': chi, 'wd': wd}
    vals = dict(moments)
    vals.update(scalars)
    missing = [k for k in MONOMIAL_ARGS if k not in vals]
    if missing:
        raise KeyError(f"missing evaluation inputs: {missing}")
    return tuple(np.broadcast_to(np.asarray(vals[k]), shape)
                 for k in MONOMIAL_ARGS)


def residual_coeffs(moments, Delta, kappa, eps, chi, wd, which='P'):
    """``(monoms, C)`` with ``C[i, j]`` the coefficient of monomial ``j`` at time ``i``.

    ``monoms`` is the tuple of ``(m, n)`` exponents of ``Z^m Zb^n``.
    """
    coeffs = lambdify_coefficients(which)
    args = _eval_args(moments, Delta, kappa, eps, chi, wd)
    n_t = args[0].size
    monoms = tuple((m, n) for m, n, _ in coeffs)
    C = np.empty((n_t, len(coeffs)), dtype=complex)
    for j, (_, _, fn) in enumerate(coeffs):
        C[:, j] = np.broadcast_to(np.asarray(fn(*args), dtype=complex), (n_t,))
    return monoms, C


def gauss_moments(m, n, s, st, sts):
    """``<Z^m Zb^n>`` for the Gaussian with ``<Z^2>=st, <Z Zb>=s, <Zb^2>=sts``.

    Wick / Isserlis, vectorised over the (array-valued) moments.
    """
    s, st, sts = np.asarray(s), np.asarray(st), np.asarray(sts)
    tot = np.zeros(np.broadcast(s, st, sts).shape, dtype=complex)
    for j in range(min(m, n) + 1):
        if (m - j) % 2 or (n - j) % 2:
            continue
        p, q = (m - j) // 2, (n - j) // 2
        tot = tot + (factorial(m) * factorial(n)
                     / (factorial(j) * factorial(p) * factorial(q))
                     * s**j * (st / 2)**p * (sts / 2)**q)
    return tot


def _project(monoms, C, probes, s, st, sts):
    """``<Z^p Zb^q P>_Q`` for every ``(p, q)`` in ``probes``."""
    out = np.zeros((C.shape[0], len(probes)), dtype=complex)
    for i, (p, q) in enumerate(probes):
        for j, (m, n) in enumerate(monoms):
            out[:, i] += C[:, j] * gauss_moments(m + p, n + q, s, st, sts)
    return out


def moment_projections(monoms, C, moments):
    """The six ``<Z^p Zb^q P>_Q``, ``p + q <= 2`` — zero iff the moment ODEs hold.

    This is the correctness check on the *solution*: an HB solve that has
    converged makes these vanish to its own residual tolerance.  Order matches
    :data:`LOW_PROBES`.
    """
    return _project(monoms, C, LOW_PROBES,
                    moments['s'], moments['st'], moments['sts'])


def strip_low_order(monoms, C, moments):
    """Remove the degree <= 2 Hermite content of ``P``.

    Solves for ``b`` such that ``P - sum_b b_pq Z^p Zb^q`` is orthogonal to all
    of ``1, Z, Zb, Z^2, Z Zb, Zb^2`` under the Gaussian, and subtracts it.  The
    degree 3 and 4 coefficients are untouched; what changes is only the
    low-order counterterms, which are exactly zero for an exactly converged
    solution.  Returns a new coefficient array with the same ``monoms``.
    """
    s, st, sts = moments['s'], moments['st'], moments['sts']
    n_t = C.shape[0]
    r = _project(monoms, C, LOW_PROBES, s, st, sts)          # (n_t, 6)
    G = np.empty((n_t, len(LOW_PROBES), len(LOW_PROBES)), dtype=complex)
    for i, (p, q) in enumerate(LOW_PROBES):
        for j, (a, b) in enumerate(LOW_PROBES):
            G[:, i, j] = gauss_moments(p + a, q + b, s, st, sts)
    coef = np.linalg.solve(G, r[:, :, None])[:, :, 0]        # (n_t, 6)
    C_out = C.copy()
    index = {mn: j for j, mn in enumerate(monoms)}
    for j, pq in enumerate(LOW_PROBES):
        C_out[:, index[pq]] -= coef[:, j]
    return C_out


def _l2_sq(monoms, C, s, st, sts):
    """``int |Q P|^2 d^2 alpha``.

    ``Q^2 = Q_half / (2 pi sqrt(Dd))`` with ``Q_half`` the normalised Gaussian
    at half the covariance, so this is an exact Gaussian moment sum.
    """
    Dd = s**2 - np.abs(st)**2
    acc = np.zeros(C.shape[0], dtype=complex)
    for j, (m, n) in enumerate(monoms):
        for k, (p, q) in enumerate(monoms):
            acc += (C[:, j] * C[:, k].conj()
                    * gauss_moments(m + q, n + p, s / 2, st / 2, sts / 2))
    return np.real(acc) / (2 * np.pi * np.sqrt(Dd))


def _l1(monoms, C, s, st, sts, n_gh=64):
    """``int |Q P| d^2 alpha = E_Q[|P|]`` by Gauss-Hermite in whitened coords.

    ``|P|`` is not a polynomial, so this one integral needs quadrature — over
    the Gaussian weight, though, not over a phase-space grid: ``n_gh**2`` nodes
    per time sample.  ``|P|`` has a kink on the zero set of ``P``, so
    convergence is algebraic and slightly erratic, not spectral: the default
    ``n_gh=64`` is good to ~0.1-0.5% relative, which is far more than a rate
    summary needs.  ``l2`` is exact by contrast.
    """
    s = np.asarray(s, dtype=float)
    st = np.asarray(st, dtype=complex)
    x, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / np.sqrt(2 * np.pi)                    # probabilists' normalisation
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    W = np.outer(w, w).ravel()
    u = np.stack([X1.ravel(), X2.ravel()])        # (2, n_q) standard normal

    # real covariance of (Re Z, Im Z) from <Z^2> = st, <|Z|^2> = s
    cxx = (s + st.real) / 2
    cyy = (s - st.real) / 2
    cxy = st.imag / 2
    # Cholesky of [[cxx, cxy], [cxy, cyy]], per time sample
    l11 = np.sqrt(cxx)
    l21 = cxy / l11
    l22 = np.sqrt(np.maximum(cyy - l21**2, 0.0))
    Zq = ((l11[:, None] * u[0][None, :])
          + 1j * (l21[:, None] * u[0][None, :] + l22[:, None] * u[1][None, :]))
    Zbq = Zq.conj()

    Pq = np.zeros_like(Zq)
    for j, (m, n) in enumerate(monoms):
        Pq += C[:, j][:, None] * Zq**m * Zbq**n
    return np.abs(Pq) @ W


def residual_norms(moments, Delta, kappa, eps, chi, wd, strip=True, n_gh=64):
    """Scalar summaries of the residual, per time sample.

    Returns a dict of arrays over ``moments['t']``:

    ``l1``
        ``int |R| d^2 alpha`` — a dimensionless *rate*, directly comparable to
        ``kappa``, since ``Q`` is normalised.
    ``l2``
        ``sqrt(int |R|^2)``.
    ``l2_qdot``
        ``sqrt(int |dQ/dt|^2)`` of the ansatz, the natural scale.
    ``l2_rel``
        ``l2 / l2_qdot`` — relative error on the PDE.
    ``proj_max``
        largest ``|<Z^p Zb^q P>|``, ``p + q <= 2``: the convergence diagnostic,
        not physics.  Should be at the HB residual level.

    With ``strip=True`` (default) the degree <= 2 Hermite content of ``P`` is
    projected out first, so ``l1``/``l2`` measure only what the ansatz
    structurally cannot represent.
    """
    s, st, sts = moments['s'], moments['st'], moments['sts']
    monoms, C = residual_coeffs(moments, Delta, kappa, eps, chi, wd, 'P')
    proj = moment_projections(monoms, C, moments)
    if strip:
        C = strip_low_order(monoms, C, moments)
    mon_l, C_l = residual_coeffs(moments, Delta, kappa, eps, chi, wd, 'P_lhs')

    l2 = np.sqrt(np.maximum(_l2_sq(monoms, C, s, st, sts), 0.0))
    l2_qdot = np.sqrt(np.maximum(_l2_sq(mon_l, C_l, s, st, sts), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        l2_rel = np.where(l2_qdot > 0, l2 / l2_qdot, np.nan)
    return {'l1': _l1(monoms, C, s, st, sts, n_gh=n_gh),
            'l2': l2, 'l2_qdot': l2_qdot, 'l2_rel': l2_rel,
            'proj_max': np.abs(proj).max(axis=1)}


def residual_report(moments, Delta, kappa, eps, chi, wd, strip=True, n_gh=64):
    """Everything at once: per-monomial channels + norms + period averages."""
    monoms, C_raw = residual_coeffs(moments, Delta, kappa, eps, chi, wd, 'P')
    C = strip_low_order(monoms, C_raw, moments) if strip else C_raw
    norms = residual_norms(moments, Delta, kappa, eps, chi, wd,
                           strip=strip, n_gh=n_gh)
    high = [(m, n) for (m, n) in monoms if m + n >= 3]
    idx = {mn: j for j, mn in enumerate(monoms)}
    channels = {mn: C[:, idx[mn]] for mn in high}
    return {'t': moments['t'], 'monoms': monoms, 'coeffs': C,
            'coeffs_raw': C_raw, 'channels': channels,
            'channel_rms': {mn: float(np.sqrt(np.mean(np.abs(v)**2)))
                            for mn, v in channels.items()},
            **norms,
            'l1_avg': float(np.mean(norms['l1'])),
            'l2_avg': float(np.mean(norms['l2'])),
            'l2_rel_avg': float(np.nanmean(norms['l2_rel']))}


# ======================================================================
# Demo / self-test:  python3.11 -m hhb.kerr.gaussian_fp_residual
# ======================================================================
def _demo():                                       # pragma: no cover
    from hhb.kerr.hb_moments_kerr import (KerrParams, HBSettingsMoments,
                                          MomentHillMethod)

    Delta, kappa, eps, wd = 1.0, 1.0, 3.0, 1.3
    settings = HBSettingsMoments(N_H=14, samples_per_harmonic=48)

    print(f"{'chi':>6} {'||R||_1':>11} {'||R||_2':>11} {'rel L2':>9} "
          f"{'proj':>9}   dominant channels")
    z = None
    for chi in [0.0, 1/6, 1/3, 0.5, 2/3, 5/6, 1.0, 7/6, 4/3, 1.5]:
        hb = MomentHillMethod(KerrParams(Delta, chi, kappa, eps, wd), settings)
        z0 = np.zeros(hb.dim) if z is None else z
        if z is None:
            z0[2] = np.sqrt(2.0)
        z = hb.solve(z0)
        m = moments_from_fourier(z, N_H=hb.N_H, omega_d=wd, nu=hb.s.nu,
                                 n_times=256)
        rep = residual_report(m, Delta, kappa, eps, chi, wd)
        top = sorted(rep['channel_rms'].items(), key=lambda kv: -kv[1])[:3]
        top_s = ', '.join(f"Z^{m_}Zb^{n_}:{v:.2e}" for (m_, n_), v in top)
        print(f"{chi:6.3f} {rep['l1_avg']:11.3e} {rep['l2_avg']:11.3e} "
              f"{rep['l2_rel_avg']:9.2e} {rep['proj_max'].max():9.1e}   {top_s}")


if __name__ == '__main__':                          # pragma: no cover
    _demo()
