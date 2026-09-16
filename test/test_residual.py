"""
Validation of hhb.kerr.gaussian_fp_residual.

Run either way:
    python3.11 test/test_residual.py
    python3.11 -m pytest test/test_residual.py -q

The three mandatory tests, in the order they must be believed:

1. chi = 0, symbolic. With the chi = 0 moment ODEs substituted, every
   coefficient of P must simplify to exactly 0.  A failure here is a bug in
   the derivation, not in the HB data.
2. chi = 0, numeric. On a real HB solution at chi = 0 every monomial
   coefficient must be < 1e-12.  A residue at 1e-6 means finite-differenced
   moment derivatives; a residue at O(1) with clean symbolics means a
   moment-ODE / HB convention mismatch.
3. General chi. The six Gaussian projections <Z^p Zb^q P>, p+q <= 2, must
   vanish identically once the moment ODEs are substituted.  This is the
   free second correctness check on the moment equations themselves.

   NOTE. The recipe this module implements asserts the stronger statement
   that the degree <= 2 *monomial coefficients* of P vanish at general chi.
   They do not, and cannot: <Z^2 Zb^2> has a non-zero Gaussian average, so
   the degree-4 content of P necessarily leaves a degree-0 footprint that
   the constant coefficient has to cancel.  Orthogonality holds against the
   Gaussian measure, which is what test 3 checks and what
   strip_low_order() uses.  test_low_degree_monomials_are_not_zero pins
   this down so the weaker claim is not silently "fixed" back.
"""

import numpy as np
import sympy as sp

from hhb.kerr.gaussian_fp_residual import (
    LOW_PROBES, build_residual_polynomial, gauss_moments, moment_odes,
    moments_from_fourier, moment_projections, residual_coeffs, residual_norms,
    strip_low_order,
)
from hhb.kerr.hb_moments_kerr import (KerrParams, HBSettingsMoments,
                                      MomentHillMethod)

# physical parameters of the sweeps / notebooks
DELTA, KAPPA, EPS, WD = 1.0, 1.0, 3.0, 1.3
N_H, SPH = 14, 48


def _solve_hb(chi, z0=None):
    hb = MomentHillMethod(KerrParams(Delta=DELTA, chi=chi, kappa=KAPPA,
                                     eps=EPS, omega_d=WD),
                          HBSettingsMoments(N_H=N_H, samples_per_harmonic=SPH))
    if z0 is None:
        z0 = np.zeros(hb.dim)
        z0[2] = np.sqrt(2.0)          # DC sigma^2 = 1 (vacuum)
    return hb, hb.solve(z0)


def _moments(hb, z, n_times=256):
    return moments_from_fourier(z, N_H=hb.N_H, omega_d=hb.p.omega_d,
                                nu=hb.s.nu, n_times=n_times)


# ----------------------------------------------------------------------
# 1. chi = 0, symbolic
# ----------------------------------------------------------------------
def test_chi_zero_symbolic():
    rp = build_residual_polynomial()
    subs = moment_odes(chi_zero=True)
    subs[rp.sym.chi] = 0
    bad = []
    for (m, n), c in sorted(rp.P.terms()):
        val = sp.simplify(sp.expand(c.subs(subs)))
        if val != 0:
            bad.append(((m, n), val))
    assert not bad, f"non-vanishing coefficients at chi=0: {bad}"


# ----------------------------------------------------------------------
# 2. chi = 0, numeric
# ----------------------------------------------------------------------
def test_chi_zero_numeric():
    hb, z = _solve_hb(0.0)
    mom = _moments(hb, z)
    monoms, C = residual_coeffs(mom, DELTA, KAPPA, EPS, 0.0, WD)
    worst = np.abs(C).max()
    assert worst < 1e-12, (
        f"chi=0 residual coefficients reach {worst:.3e} (want < 1e-12); "
        f"per monomial: "
        f"{ {mn: float(np.abs(C[:, j]).max()) for j, mn in enumerate(monoms)} }")

    norms = residual_norms(mom, DELTA, KAPPA, EPS, 0.0, WD)
    assert norms['l1'].max() < 1e-12
    assert norms['l2'].max() < 1e-12
    assert norms['l2_qdot'].max() > 1e-3      # dQ/dt itself is not zero


# ----------------------------------------------------------------------
# 3. general chi: the six Gaussian projections vanish identically
# ----------------------------------------------------------------------
def test_low_degree_projections_vanish_general_chi():
    rp = build_residual_polynomial()
    y = rp.sym
    subs = moment_odes()
    terms = [((m, n), c.subs(subs)) for (m, n), c in rp.P.terms()]

    def gauss_moment_sym(m, n):
        tot = sp.Integer(0)
        for j in range(min(m, n) + 1):
            if (m - j) % 2 or (n - j) % 2:
                continue
            p, q = (m - j) // 2, (n - j) // 2
            tot += (sp.Rational(sp.factorial(m) * sp.factorial(n),
                                sp.factorial(j) * sp.factorial(p) * sp.factorial(q))
                    * y.s**j * (y.st / 2)**p * (y.sts / 2)**q)
        return sp.expand(tot)

    # the Wick formula itself
    assert sp.expand(gauss_moment_sym(1, 1) - y.s) == 0
    assert sp.expand(gauss_moment_sym(2, 0) - y.st) == 0
    assert sp.expand(gauss_moment_sym(2, 2) - (y.st * y.sts + 2 * y.s**2)) == 0

    bad = []
    for (p, q) in LOW_PROBES:
        proj = sum(c * gauss_moment_sym(m + p, n + q) for (m, n), c in terms)
        val = sp.simplify(sp.expand(proj))
        if val != 0:
            bad.append(((p, q), val))
    assert not bad, f"non-vanishing moment projections at general chi: {bad}"


def test_low_degree_monomials_are_not_zero():
    """The recipe's stronger claim is false — keep it falsified on purpose."""
    rp = build_residual_polynomial()
    subs = moment_odes()
    nonzero = [(m, n) for (m, n), c in rp.P.terms()
               if m + n <= 2 and sp.simplify(sp.expand(c.subs(subs))) != 0]
    assert nonzero, ("degree <= 2 monomial coefficients all vanished; if this "
                     "is real, strip_low_order() is redundant and the module "
                     "docstring needs updating")


# ----------------------------------------------------------------------
# supporting checks
# ----------------------------------------------------------------------
def test_gauss_moments_numeric_matches_sampling():
    rng = np.random.default_rng(0)
    s, st = 1.7, 0.6 - 0.35j
    cxx, cyy, cxy = (s + st.real) / 2, (s - st.real) / 2, st.imag / 2
    C = np.array([[cxx, cxy], [cxy, cyy]])
    xy = rng.multivariate_normal([0, 0], C, size=4_000_000)
    Zs = xy[:, 0] + 1j * xy[:, 1]
    for (m, n) in [(1, 1), (2, 0), (2, 2), (3, 1), (4, 0)]:
        exact = complex(gauss_moments(m, n, s, st, st.conjugate()))
        emp = np.mean(Zs**m * np.conj(Zs)**n)
        assert abs(exact - emp) < 0.02 * (1 + abs(exact)), (m, n, exact, emp)


def test_fourier_derivative_is_analytic():
    """fdot must be the analytic derivative, matched by a fine difference."""
    hb, z = _solve_hb(0.5)
    h = 1e-6
    m0 = _moments(hb, z, n_times=64)
    tp = moments_from_fourier(z, N_H=hb.N_H, omega_d=WD, nu=hb.s.nu,
                              t=m0['t'] + h)
    tm = moments_from_fourier(z, N_H=hb.N_H, omega_d=WD, nu=hb.s.nu,
                              t=m0['t'] - h)
    for key, dkey in [('mu', 'dmu'), ('s', 'ds'), ('st', 'dst')]:
        fd = (tp[key] - tm[key]) / (2 * h)
        assert np.abs(fd - m0[dkey]).max() < 1e-5 * (1 + np.abs(m0[dkey]).max())


def _Q_ansatz(al, als, mu, mus, s, st, sts):
    Z, Zb = al - mu, als - mus
    Dd = s**2 - st * sts
    return (np.exp((-s * Z * Zb + (sts * Z**2 + st * Zb**2) / 2) / Dd)
            / (np.pi * np.sqrt(Dd)))


def test_residual_matches_finite_differences():
    """Independent check of the whole symbolic layer.

    Evaluate both sides of the Husimi PDE by finite differences at a handful
    of phase-space points, with moments and moment derivatives chosen at
    random (NOT satisfying any ODE), and compare with Q * P.  Catches a wrong
    chain rule, a mis-conjugated term or a bad Z substitution — none of which
    the chi = 0 tests would see.
    """
    from hhb.kerr.gaussian_fp_residual import (MONOMIAL_ARGS,
                                               lambdify_coefficients)
    mu, s, st = 0.7 - 0.4j, 1.8, 0.5 + 0.3j
    dmu, ds, dst = 0.3 + 0.2j, -0.15, 0.11 - 0.27j
    mus, sts, dmus, dsts = np.conj(mu), np.conj(st), np.conj(dmu), np.conj(dst)
    chi, t = 0.77, 0.41
    pars = dict(mu=mu, mus=mus, s=s, st=st, sts=sts, dmu=dmu, dmus=dmus,
                ds=ds, dst=dst, dsts=dsts, Delta=DELTA, kappa=KAPPA, eps=EPS,
                chi=chi, wd=WD, t=t)
    base = dict(mu=mu, mus=mus, s=s, st=st, sts=sts)
    h = 1e-5

    def q(a, b):
        return _Q_ansatz(a, b, **base)

    def A(a, b):
        return (1j * DELTA * a + 1j * EPS / 2 * (1 + np.exp(2j * WD * t))
                + 1j * chi * a**2 * b + KAPPA * a / 2)

    def As(a, b):
        return (-1j * DELTA * b - 1j * EPS / 2 * (1 + np.exp(-2j * WD * t))
                - 1j * chi * b**2 * a + KAPPA * b / 2)

    def num_R(al):
        als = np.conj(al)
        lhs = 0
        for name, dot in [('mu', dmu), ('mus', dmus), ('s', ds),
                          ('st', dst), ('sts', dsts)]:
            def ev(delta, name=name):
                b = dict(base)
                b[name] = b[name] + delta
                return _Q_ansatz(al, als, **b)
            lhs = lhs + (ev(h) - ev(-h)) / (2 * h) * dot
        AQ = lambda a, b: A(a, b) * q(a, b)          # noqa: E731
        AsQ = lambda a, b: As(a, b) * q(a, b)        # noqa: E731
        rhs = ((AQ(al + h, als) - AQ(al - h, als)) / (2 * h)
               + (AsQ(al, als + h) - AsQ(al, als - h)) / (2 * h)
               + KAPPA * (q(al + h, als + h) - q(al + h, als - h)
                          - q(al - h, als + h) + q(al - h, als - h)) / (4 * h**2)
               + 1j * chi / 2
               * (al**2 * (q(al + h, als) - 2 * q(al, als) + q(al - h, als)) / h**2
                  - als**2 * (q(al, als + h) - 2 * q(al, als) + q(al, als - h)) / h**2))
        return lhs - rhs

    coeffs = lambdify_coefficients('P')
    args = tuple(pars[k] for k in MONOMIAL_ARGS)
    for al in [0.2 + 0.1j, 1.5 - 0.9j, -1.1 + 2.0j, 2.5 + 2.5j, mu + 0.01]:
        P = sum(fn(*args) * (al - mu)**m * (np.conj(al) - mus)**n
                for m, n, fn in coeffs)
        ana = _Q_ansatz(al, np.conj(al), **base) * P
        num = num_R(al)
        assert abs(ana - num) < 1e-4 * abs(num), (al, ana, num)


def test_norms_match_grid_quadrature():
    """The Gaussian-moment L2 and the Gauss-Hermite L1 vs. a phase-space grid."""
    from hhb.kerr.gaussian_fp_residual import _l1, _l2_sq

    hb, z = _solve_hb(1.0)
    mom = _moments(hb, z, n_times=8)
    monoms, C = residual_coeffs(mom, DELTA, KAPPA, EPS, 1.0, WD)
    C = strip_low_order(monoms, C, mom)

    ax = np.linspace(-14, 14, 1201)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    dA = (ax[1] - ax[0])**2
    al = X + 1j * Y

    l1 = _l1(monoms, C, mom['s'], mom['st'], mom['sts'])
    l2 = np.sqrt(_l2_sq(monoms, C, mom['s'], mom['st'], mom['sts']))
    for k in range(len(mom['t'])):
        Q = _Q_ansatz(al, np.conj(al), mom['mu'][k], mom['mus'][k],
                      mom['s'][k], mom['st'][k], mom['sts'][k])
        Z = al - mom['mu'][k]
        P = sum(C[k, j] * Z**m * np.conj(Z)**n
                for j, (m, n) in enumerate(monoms))
        R = Q * P
        # L1 is quadrature on a kinked integrand: ~0.5% is the honest target
        assert abs(np.abs(R).sum() * dA - l1[k]) < 5e-3 * l1[k], k
        # L2 is an exact Gaussian moment sum
        assert abs(np.sqrt((np.abs(R)**2).sum() * dA) - l2[k]) < 1e-6 * l2[k], k


def test_strip_low_order_is_a_no_op_on_high_degree():
    hb, z = _solve_hb(1.0)
    mom = _moments(hb, z, n_times=64)
    monoms, C = residual_coeffs(mom, DELTA, KAPPA, EPS, 1.0, WD)
    Cs = strip_low_order(monoms, C, mom)
    for j, (m, n) in enumerate(monoms):
        if m + n >= 3:
            assert np.array_equal(C[:, j], Cs[:, j]), (m, n)
    # and it does make the projections vanish
    proj = moment_projections(monoms, Cs, mom)
    assert np.abs(proj).max() < 1e-10 * max(1.0, np.abs(C).max())


def test_residual_grows_with_chi():
    """Qualitative expectation: more Kerr, more non-Gaussianity."""
    z = None
    vals = []
    for chi in [0.0, 1 / 3, 2 / 3, 1.0]:
        hb, z = _solve_hb(chi, z)
        mom = _moments(hb, z, n_times=128)
        n = residual_norms(mom, DELTA, KAPPA, EPS, chi, WD)
        vals.append(float(np.mean(n['l1'])))
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


if __name__ == '__main__':
    import time
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fails = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {fn.__name__:<48} {time.time() - t0:6.2f}s")
        except AssertionError as exc:
            fails += 1
            print(f"FAIL  {fn.__name__:<48} {time.time() - t0:6.2f}s\n      {exc}")
    raise SystemExit(1 if fails else 0)
