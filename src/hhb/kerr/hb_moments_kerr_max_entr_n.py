"""
!!  NOT APPLICABLE TO THE DRIVEN KERR OSCILLATOR -- KEPT FOR REFERENCE  !!

    A quartic max-ent density exp(-P), deg P = 4, needs its top-degree form
    positive in every direction to be normalizable, so its tails decay FASTER
    than Gaussian and it is necessarily PLATYKURTIC (negative excess kurtosis).
    The driven-Kerr Q function is LEPTOKURTIC at every chi > 0 and every drive
    phase (max excess kurtosis over directions: +0.007 at chi=1/6, +0.027 at
    chi=1/3, +0.145 at chi=1). Its moments therefore lie outside the set where
    the max-ent problem has a solution at all (the known max-ent realizability
    gap / Junk's singularity). Order 6 does not rescue it either: fed the exact
    degree-6 moments, the sextic form still comes out non-positive.

    THIS FAILS SILENTLY. On a leptokurtic target the inner Newton "converges"
    to residual ~1e-14 because a finite Gauss-Hermite grid assigns a finite
    value to a divergent integral. The tell is grid dependence: closure moments
    differ by ~51% between 32 and 64 nodes (vs ~5e-7 for a representable,
    platykurtic target). Any sweep built on this produces smooth, plausible,
    quadrature-grid-dependent garbage.

    The code below is correct for problems that ARE representable -- the
    closure converges, reproduces target moments to 1e-15, and its analytic
    sensitivity matches finite differences -- and the self-tests at the bottom
    pass for such targets. See README.md ("Closure attempts") for the full
    record and for the directions that remain open.

Harmonic-balance solver for the driven-Kerr moment equations with a
MAXIMUM-ENTROPY closure at even truncation order n.

Why: the cumulant closure of hb_moments_kerr_n.py ("set kappa_{n+1} = 0") has
no positivity structure, so the truncated flow can leave the set of realizable
moments. At order 4 that shows up as (i) cumulant drift, (ii) a spurious fold
at chi ~ 1.08 where the periodic solution ceases to exist, and (iii) finite-time
runaway from generic initial data for chi >~ 0.83 -- none of which reflect the
true Lindblad dynamics.

The max-ent closure instead defines the closing moments as those of the
MAXIMUM-ENTROPY density consistent with the tracked moments:

    p(zeta) = argmax  -int p log p   subject to   <zeta^j zetabar^k> = M_jk,
                                                  2 <= j+k <= n
            = exp( -sum_{1<=p+q<=n} theta_pq x^p y^q )  / Z,   zeta = x + iy

This is a genuine probability density (positive, normalized, UNIMODAL for the
orders used here), so the closed system cannot leave the realizable set: the
moments always belong to an actual distribution. It also supplies the Husimi
function directly -- no Gram-Charlier/Edgeworth expansion, no negative tails,
no clipping.

ORDER MUST BE EVEN. p ~ exp(-polynomial of degree n) is normalizable only if
the degree-n part tends to +infinity in every direction, which is impossible
for odd n. n=2 recovers the Gaussian closure exactly (regression test below);
n=4 is the useful case.

Tracked objects: mu = <alpha>, and CENTRAL moments M_jk = <dzeta^j dzetabar^k>
for 2 <= j+k <= n (M_kj = conj(M_jk); M_jj real). Real-DOF count is the same
as the cumulant module, (n+1)(n+2)/2 - 1  (n=2 -> 5, n=4 -> 14), and the state
packing is identical in spirit (graded, (j,k) with j descending, Re/Im pairs,
M_jj a single real slot) -- but the variables are CENTRAL MOMENTS, not
cumulants. Use .index_map / .reconstruct rather than hard-coded indices.

Structure
  1. Symbolic (exact, cached per order): the FPE raw-moment recursion and the
     binomial raw<->central relation give dmu/dt and dM_jk/dt in terms of the
     tracked moments PLUS the closing moments of degree n+1 and n+2. The drive
     is excluded (an additive c-number force shifts only the mean and cancels
     in every central moment) and handled exactly by b_ext.
  2. Numerical (per time sample): MaxEntClosure maps the tracked moments to the
     closing moments by solving the convex dual problem for theta with Newton,
     using Gauss-Hermite quadrature against the reference Gaussian fixed by the
     tracked second moments. The sensitivity d(closing)/d(tracked) comes from
     the same quadrature via the exponential-family identity
         dM_a/dtheta_b = -Cov(phi_a, phi_b)
     so  d(closing)/d(tracked) = Cov(phi_out, phi) Cov(phi, phi)^{-1},
     with no finite differences anywhere.
  3. HB/AFT/Newton machinery identical to hb_moments_kerr_n.py.

A failure of the inner Newton is INFORMATIVE, not a bug: it means the requested
moments are (near) the boundary of the realizable set, i.e. no distribution
whatsoever has them. That is the diagnostic the cumulant closure lacked.
"""

import numpy as np
import sympy as sp
from math import comb
from scipy.sparse import lil_matrix
from scipy.optimize import root
from dataclasses import dataclass

I = sp.I


@dataclass
class KerrParams:
    Delta: float
    chi: float
    kappa: float
    eps: float
    omega_d: float


@dataclass
class HBSettingsMaxEnt:
    order: int = 4               # EVEN truncation degree
    N_H: int = 8
    samples_per_harmonic: int = 16
    nu: int = 1
    n_nodes: int = 32            # Gauss-Hermite nodes per axis (n_nodes^2 total)

    @property
    def n(self):
        return (self.order + 1) * (self.order + 2) // 2 - 1

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H


# ======================================================================
# Layout helpers
# ======================================================================
def state_layout(order):
    """['mu'] then ('M', j, k) for each degree 2..order, j descending, j >= k."""
    slots = [('mu',)]
    for m in range(2, order + 1):
        for j in range(m, (m - 1) // 2, -1):
            slots.append(('M', j, m - j))
    return slots


def closure_layout(order):
    """('M', j, k) for degrees order+1 and order+2."""
    slots = []
    for m in (order + 1, order + 2):
        for j in range(m, (m - 1) // 2, -1):
            slots.append(('M', j, m - j))
    return slots


def real_exps(d):
    """Exponents (p, q) with p + q = d, in a fixed order."""
    return [(d - i, i) for i in range(d + 1)]


def _complex_to_real_matrix(degree):
    """Real matrix B: (real-monomial moments of this degree) -> (complex
    components in layout order).  M_jk = sum_ab C(j,a)C(k,b) i^a (-i)^b m_{...}
    """
    exps = real_exps(degree)
    col = {e: i for i, e in enumerate(exps)}
    rows = []
    for j in range(degree, (degree - 1) // 2, -1):
        k = degree - j
        vec = np.zeros(len(exps), dtype=complex)
        for a in range(j + 1):
            for b in range(k + 1):
                coef = comb(j, a) * comb(k, b) * (1j)**a * (-1j)**b
                vec[col[(j + k - a - b, a + b)]] += coef
        if j == k:
            rows.append(vec.real)                    # M_jj is real
        else:
            rows.append(vec.real)
            rows.append(vec.imag)
    return np.array(rows)                            # (degree+1, degree+1)


# ======================================================================
# Maximum-entropy closure
# ======================================================================
class MaxEntClosure:
    """Maps tracked central moments -> closing moments of degree n+1, n+2.

    Vectorized over an arbitrary number of independent samples (time points).
    """

    def __init__(self, order, n_nodes=32, tol=1e-11, max_iter=60):
        if order % 2 or order < 2:
            raise ValueError("max-ent closure requires an EVEN order >= 2 "
                             "(exp(-odd polynomial) is not normalizable)")
        self.order = order
        self.tol = tol
        self.max_iter = max_iter

        # constrained monomials: degrees 1..order (degree-1 targets are 0)
        self.cons = [e for d in range(1, order + 1) for e in real_exps(d)]
        # closing monomials: degrees order+1, order+2
        self.out = [e for d in (order + 1, order + 2) for e in real_exps(d)]
        self.n_cons, self.n_out = len(self.cons), len(self.out)

        # every monomial we ever need a moment of: up to degree 2*order + 2
        self.max_deg = 2 * order + 2
        self.all_exps = [(0, 0)] + [e for d in range(1, self.max_deg + 1)
                                    for e in real_exps(d)]
        self.pos = {e: i for i, e in enumerate(self.all_exps)}
        self.i_cons = np.array([self.pos[e] for e in self.cons])
        self.i_out = np.array([self.pos[e] for e in self.out])
        # index tables for products, so covariances are a lookup not a quadrature
        self.i_cc = np.array([[self.pos[(a[0] + b[0], a[1] + b[1])]
                               for b in self.cons] for a in self.cons])
        self.i_oc = np.array([[self.pos[(a[0] + b[0], a[1] + b[1])]
                               for b in self.cons] for a in self.out])

        # fixed 2D Gauss-Hermite rule for weight exp(-|u|^2)
        u, w = np.polynomial.hermite.hermgauss(n_nodes)
        U1, U2 = np.meshgrid(u, u, indexing='ij')
        self.u = np.stack([U1.ravel(), U2.ravel()])          # (2, K)
        self.w = (np.outer(w, w).ravel()) / np.pi            # (K,)
        self.K = self.w.size

        # conversion matrices per degree
        self._B = {d: _complex_to_real_matrix(d)
                   for d in range(2, order + 3)}
        self._Binv = {d: np.linalg.inv(B) for d, B in self._B.items()}

    # ---------- state <-> real monomial moments ----------
    def state_to_targets(self, S):
        """S: (n_state-2, nsamp) central-moment components (degrees 2..order,
        layout order) -> targets m_pq, (n_cons, nsamp), degree-1 rows zero."""
        nsamp = S.shape[1]
        T = np.zeros((self.n_cons, nsamp))
        row_s, row_t = 0, 2                      # skip the two degree-1 rows
        for d in range(2, self.order + 1):
            nd = d + 1
            T[row_t:row_t + nd] = self._Binv[d] @ S[row_s:row_s + nd]
            row_s += nd
            row_t += nd
        return T

    def out_to_components(self, Mout):
        """Mout: (n_out, nsamp) real monomial moments of degree order+1, order+2
        -> complex components in closure_layout order."""
        nsamp = Mout.shape[1]
        C = np.zeros((self.n_out, nsamp))
        row = 0
        for d in (self.order + 1, self.order + 2):
            nd = d + 1
            C[row:row + nd] = self._B[d] @ Mout[row:row + nd]
            row += nd
        return C

    def _out_jac_wrap(self, dMout_dT):
        """Wrap d(real out)/d(real targets) into d(components)/d(state comps)."""
        nsamp = dMout_dT.shape[2]
        Bout = np.zeros((self.n_out, self.n_out))
        row = 0
        for d in (self.order + 1, self.order + 2):
            nd = d + 1
            Bout[row:row + nd, row:row + nd] = self._B[d]
            row += nd
        n_state_mom = self.n_cons - 2
        Bin = np.zeros((n_state_mom, n_state_mom))
        row = 0
        for d in range(2, self.order + 1):
            nd = d + 1
            Bin[row:row + nd, row:row + nd] = self._Binv[d]
            row += nd
        # dMout_dT columns are the degree>=2 targets
        return np.einsum('ab,bcs,cd->ads', Bout, dMout_dT, Bin)

    # ---------- the convex dual solve ----------
    def solve(self, T, eta0=None, chunk=64, want_jac=True):
        """T: (n_cons, nsamp) target monomial moments (degree-1 rows = 0).

        Returns dict with eta, closing moments Mout (n_out, nsamp), optional
        dMout/dT restricted to degree>=2 columns, and per-sample convergence.
        """
        nsamp = T.shape[1]
        eta = np.zeros((self.n_cons, nsamp)) if eta0 is None else eta0.copy()
        Mout = np.zeros((self.n_out, nsamp))
        dJ = np.zeros((self.n_out, self.n_cons - 2, nsamp)) if want_jac else None
        ok = np.zeros(nsamp, dtype=bool)
        resid = np.zeros(nsamp)

        for s0 in range(0, nsamp, chunk):
            s1 = min(s0 + chunk, nsamp)
            sl = slice(s0, s1)
            out = self._solve_chunk(T[:, sl], eta[:, sl], want_jac)
            eta[:, sl] = out['eta']
            Mout[:, sl] = out['Mout']
            ok[sl] = out['ok']
            resid[sl] = out['resid']
            if want_jac:
                dJ[:, :, sl] = out['dJ']
        return dict(eta=eta, Mout=Mout, dJ=dJ, ok=ok, resid=resid)

    def _nodes_and_monomials(self, T):
        """Quadrature nodes from the reference Gaussian fixed by the 2nd moments,
        plus every monomial evaluated there. T: (n_cons, c)."""
        c = T.shape[1]
        i20, i11, i02 = self.pos[(2, 0)], self.pos[(1, 1)], self.pos[(0, 2)]
        j20 = self.cons.index((2, 0))
        j11 = self.cons.index((1, 1))
        j02 = self.cons.index((0, 2))
        cxx, cxy, cyy = T[j20], T[j11], T[j02]
        # Cholesky of [[cxx,cxy],[cxy,cyy]] (2x2, done explicitly & vectorized)
        L11 = np.sqrt(np.maximum(cxx, 1e-12))
        L21 = cxy / L11
        L22 = np.sqrt(np.maximum(cyy - L21**2, 1e-12))
        r2 = np.sqrt(2.0)
        # zeta = sqrt(2) L u    ->  (c, K)
        x = r2 * (L11[:, None] * self.u[0][None, :])
        y = r2 * (L21[:, None] * self.u[0][None, :]
                  + L22[:, None] * self.u[1][None, :])
        mon = np.empty((len(self.all_exps), c, self.K))
        xp = {0: np.ones_like(x)}
        yp = {0: np.ones_like(y)}
        for d in range(1, self.max_deg + 1):
            xp[d] = xp[d - 1] * x
            yp[d] = yp[d - 1] * y
        for i, (p, q) in enumerate(self.all_exps):
            mon[i] = xp[p] * yp[q]
        return mon

    def _solve_chunk(self, T, eta, want_jac):
        c = T.shape[1]
        mon = self._nodes_and_monomials(T)          # (n_all, c, K)
        mcons = mon[self.i_cons]                    # (n_cons, c, K)
        w = self.w[None, :]

        def moments(eta_):
            H = np.einsum('ac,ack->ck', eta_, mcons)
            H -= H.min(axis=1, keepdims=True)       # overflow guard (ratios only)
            E = np.exp(-H) * w                      # (c, K)
            raw = np.einsum('ck,ack->ac', E, mon)   # (n_all, c)
            return raw / raw[self.pos[(0, 0)]][None, :]

        ok = np.zeros(c, dtype=bool)
        res_norm = np.full(c, np.inf)
        for _ in range(self.max_iter):
            M = moments(eta)
            F = M[self.i_cons] - T                  # (n_cons, c)
            res_norm = np.max(np.abs(F), axis=0)
            ok = res_norm < self.tol
            if ok.all():
                break
            # Cov(phi_a, phi_b) = <phi_a phi_b> - <phi_a><phi_b>
            Cov = (M[self.i_cc] - M[self.i_cons][:, None, :]
                   * M[self.i_cons][None, :, :])            # (n_cons,n_cons,c)
            step = np.zeros_like(F)
            for s in range(c):
                if ok[s]:
                    continue
                try:
                    step[:, s] = np.linalg.solve(Cov[:, :, s], F[:, s])
                except np.linalg.LinAlgError:
                    step[:, s] = np.linalg.lstsq(Cov[:, :, s], F[:, s],
                                                 rcond=None)[0]
            # damped Newton: eta <- eta + Cov^{-1} F, backtracking on |F|
            lam = np.ones(c)
            base = res_norm.copy()
            new_eta = eta + step * lam[None, :]
            for _bt in range(25):
                Mn = moments(new_eta)
                rn = np.max(np.abs(Mn[self.i_cons] - T), axis=0)
                bad = ~np.isfinite(rn) | (rn > base)
                bad &= ~ok
                if not bad.any():
                    break
                lam[bad] *= 0.5
                new_eta = eta + step * lam[None, :]
            eta = new_eta

        M = moments(eta)
        Mout = M[self.i_out]
        dJ = None
        if want_jac:
            Cov = (M[self.i_cc] - M[self.i_cons][:, None, :]
                   * M[self.i_cons][None, :, :])
            CovOut = (M[self.i_oc] - M[self.i_out][:, None, :]
                      * M[self.i_cons][None, :, :])        # (n_out,n_cons,c)
            dJ = np.zeros((self.n_out, self.n_cons - 2, c))
            for s in range(c):
                try:
                    full = np.linalg.solve(Cov[:, :, s].T, CovOut[:, :, s].T).T
                except np.linalg.LinAlgError:
                    full = CovOut[:, :, s] @ np.linalg.pinv(Cov[:, :, s])
                dJ[:, :, s] = full[:, 2:]      # drop the degree-1 columns
        return dict(eta=eta, Mout=Mout, dJ=dJ, ok=ok, resid=res_norm)

    # ---------- the density itself (this is the Husimi function) ----------
    def density(self, T, eta, mu, Z, chunk=None):
        """Evaluate the max-ent density on complex points Z (any shape), for a
        SINGLE sample: T (n_cons,), eta (n_cons,), mu complex."""
        T = T[:, None]
        mon_nodes = self._nodes_and_monomials(T)             # (n_all, 1, K)
        H = np.einsum('a,ak->k', eta, mon_nodes[self.i_cons, 0])
        shift = H.min()
        Zn = np.sum(self.w * np.exp(-(H - shift)))           # <e^-H>_g, shifted

        zeta = np.asarray(Z) - mu
        x, y = zeta.real, zeta.imag
        j20 = self.cons.index((2, 0)); j11 = self.cons.index((1, 1))
        j02 = self.cons.index((0, 2))
        cxx, cxy, cyy = T[j20, 0], T[j11, 0], T[j02, 0]
        det = cxx * cyy - cxy**2
        g = (np.exp(-(cyy * x**2 - 2 * cxy * x * y + cxx * y**2) / (2 * det))
             / (2 * np.pi * np.sqrt(det)))
        Hq = np.zeros(zeta.shape)
        for a, (p, q) in enumerate(self.cons):
            Hq = Hq + eta[a] * x**p * y**q
        return g * np.exp(-(Hq - shift)) / Zn


# ======================================================================
# Symbolic derivation: dmu/dt and dM_jk/dt in terms of tracked + closing
# ======================================================================
def _derive(order):
    Delta, chi, kap = sp.symbols('Delta chi kappa', real=True)
    params = (chi, kap, Delta)

    slots = state_layout(order)
    cslots = closure_layout(order)

    y_syms, Mc = [], {}
    mu_r, mu_i = sp.symbols('mu_r mu_i', real=True)
    y_syms += [mu_r, mu_i]
    mu, mus = mu_r + I * mu_i, mu_r - I * mu_i

    def add_moment(j, k, into):
        if j == k:
            s = sp.Symbol(f'M{j}{k}', real=True)
            into.append(s)
            Mc[(j, k)] = s
        else:
            sr, si = sp.symbols(f'M{j}{k}_r M{j}{k}_i', real=True)
            into += [sr, si]
            Mc[(j, k)] = sr + I * si
            Mc[(k, j)] = sr - I * si

    for _, j, k in [s for s in slots if s[0] == 'M']:
        add_moment(j, k, y_syms)
    c_syms = []
    for _, j, k in cslots:
        add_moment(j, k, c_syms)

    Mc[(0, 0)] = sp.Integer(1)
    Mc[(1, 0)] = Mc[(0, 1)] = sp.Integer(0)
    nvars, nclose = len(y_syms), len(c_syms)
    assert nvars == (order + 1) * (order + 2) // 2 - 1

    # exact FPE recursion, drive-free
    a, ac = sp.symbols('a ac')
    A = I * Delta * a + I * chi * a**2 * ac + kap / 2 * a
    Ac = -I * Delta * ac - I * chi * ac**2 * a + kap / 2 * ac

    def moment_rhs_monomials(m, n):
        f = a**m * ac**n
        expr = (-A * sp.diff(f, a) - Ac * sp.diff(f, ac)
                + kap * sp.diff(f, a, ac)
                + I * chi / 2 * (sp.diff(a**2 * f, a, 2)
                                 - sp.diff(ac**2 * f, ac, 2)))
        return {mo: co for mo, co in sp.Poly(sp.expand(expr), a, ac).terms()}

    def R(m, n):
        tot = sp.Integer(0)
        for j in range(m + 1):
            for k in range(n + 1):
                Mjk = Mc.get((j, k), sp.Integer(0))
                if Mjk == 0:
                    continue
                tot += (sp.binomial(m, j) * sp.binomial(n, k)
                        * mu**(m - j) * mus**(n - k) * Mjk)
        return sp.expand(tot)

    def dotR(m, n):
        return sp.expand(sum(c * R(mm, nn)
                             for (mm, nn), c in moment_rhs_monomials(m, n).items()))

    probes, dprobes = [], []
    for slot in slots:
        if slot[0] == 'mu':
            probes += [sp.re(mu), sp.im(mu)]
            d = dotR(1, 0)
            dprobes += [sp.expand(sp.re(d)), sp.expand(sp.im(d))]
        else:
            _, j, k = slot
            Rc, dRc = R(j, k), dotR(j, k)
            if j == k:
                probes.append(sp.expand(sp.re(Rc)))
                dprobes.append(sp.expand(sp.re(dRc)))
            else:
                probes += [sp.expand(sp.re(Rc)), sp.expand(sp.im(Rc))]
                dprobes += [sp.expand(sp.re(dRc)), sp.expand(sp.im(dRc))]
    assert len(probes) == nvars

    yvec = sp.Matrix(y_syms)
    J = sp.Matrix(probes).jacobian(yvec)
    for i in range(nvars):
        assert sp.expand(J[i, i] - 1) == 0, f"probe {i} diagonal != 1"
        for l in range(i + 1, nvars):
            assert sp.expand(J[i, l]) == 0, f"probe {i} sees later var {l}"
    F = []
    for i in range(nvars):
        e = dprobes[i]
        for l in range(i):
            if J[i, l] != 0:
                e -= J[i, l] * F[l]
        F.append(sp.expand(e))
    F = sp.Matrix(F)

    args = tuple(y_syms) + tuple(c_syms) + params
    dF_dy = F.jacobian(yvec)
    dF_dc = F.jacobian(sp.Matrix(c_syms))
    return dict(
        order=order, nvars=nvars, nclose=nclose, slots=slots, cslots=cslots,
        y_syms=y_syms, c_syms=c_syms, F_sym=F,
        F_func=sp.lambdify(args, F, modules='numpy'),
        dFdy_func=sp.lambdify(args, dF_dy, modules='numpy'),
        dFdc_func=sp.lambdify(args, dF_dc, modules='numpy'),
    )


_DERIVED = {}


def get_derivation(order):
    if order not in _DERIVED:
        _DERIVED[order] = _derive(order)
    return _DERIVED[order]


# ======================================================================
# Harmonic-balance solver
# ======================================================================
class MomentHillMethodMaxEnt:
    def __init__(self, kerr: KerrParams, settings: HBSettingsMaxEnt = None):
        self.p = kerr
        self.s = settings if settings is not None else HBSettingsMaxEnt()
        self.order = self.s.order
        self.d = get_derivation(self.order)
        self.closure = MaxEntClosure(self.order, n_nodes=self.s.n_nodes)
        self.n = self.s.n
        self.N_H, self.N = self.s.N_H, self.s.N
        self.dim = self.n * (2 * self.N_H + 1)

        self.T_fund = 2 * np.pi * self.s.nu / self.p.omega_d
        self.tau_j = np.linspace(0, self.T_fund, self.N, endpoint=False)
        self._params = (self.p.chi, self.p.kappa, self.p.Delta)

        self.ref = self._vacuum_state()
        Fr, Jr = self._rhs_and_jac(self.ref[:, None])
        self.Lin = Jr[:, :, 0]
        self.F0 = Fr[:, 0] - self.Lin @ self.ref

        self.L = self.build_L()
        self.Gamma = self.build_Gamma()
        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)
        self.b_ext = self.build_b_ext()

    # ---------- reference state: vacuum Gaussian ----------
    def _vacuum_state(self):
        """mu = 0, <|dz|^2> = 1, <dz^2> = 0, higher = the Gaussian values."""
        y = np.zeros(self.n)
        im = self.index_map
        y[im['M11'][0]] = 1.0
        if self.order >= 4:
            # circular complex Gaussian with <|z|^2>=1: <|z|^4> = 2
            y[im['M22'][0]] = 2.0
        return y

    @property
    def index_map(self):
        out, i = {}, 0
        for slot in self.d['slots']:
            if slot[0] == 'mu':
                out['mu'] = (i, i + 1); i += 2
            else:
                _, j, k = slot
                if j == k:
                    out[f'M{j}{k}'] = (i,); i += 1
                else:
                    out[f'M{j}{k}'] = (i, i + 1); i += 2
        return out

    # ---------- closure + RHS ----------
    def _rhs_and_jac(self, X, want_jac=True):
        """X: (n, nsamp) -> F (n, nsamp), dF/dX (n, n, nsamp)."""
        T = self.closure.state_to_targets(X[2:])
        sol = self.closure.solve(T, want_jac=want_jac)
        if not sol['ok'].all():
            bad = int((~sol['ok']).sum())
            raise RuntimeError(
                f"max-ent closure failed at {bad}/{X.shape[1]} time samples "
                f"(max residual {sol['resid'].max():.2e}). The requested moments "
                f"are at/outside the realizable set -- no density has them.")
        Cc = self.closure.out_to_components(sol['Mout'])
        args = tuple(X) + tuple(Cc) + self._params
        F = np.array(self.d['F_func'](*args), dtype=float).reshape(self.n, -1)
        if not want_jac:
            return F, None
        dFdy = np.array(self.d['dFdy_func'](*args), dtype=float)
        dFdc = np.array(self.d['dFdc_func'](*args), dtype=float)
        dFdy = np.broadcast_to(dFdy.reshape(self.n, self.n, -1),
                               (self.n, self.n, X.shape[1]))
        dFdc = np.broadcast_to(dFdc.reshape(self.n, self.d['nclose'], -1),
                               (self.n, self.d['nclose'], X.shape[1]))
        dCc = self.closure._out_jac_wrap(sol['dJ'])       # (nclose, n-2, nsamp)
        chain = np.einsum('acs,cbs->abs', dFdc, dCc)      # (n, n-2, nsamp)
        J = np.array(dFdy, dtype=float, copy=True)
        J[:, 2:, :] += chain
        return F, J

    # ---------- HB linear operator / AFT (as in hb_moments_kerr_n.py) ----------
    def build_L(self):
        n, N_H, nu = self.n, self.N_H, self.s.nu
        omega = self.p.omega_d
        L = np.zeros((self.dim, self.dim))
        L[0:n, 0:n] = -self.Lin
        for k in range(1, N_H + 1):
            freq = k * omega / nu
            is_, ic_ = 1 + 2 * (k - 1), 2 + 2 * (k - 1)
            L[is_ * n:(is_ + 1) * n, is_ * n:(is_ + 1) * n] = -self.Lin
            L[is_ * n:(is_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, is_ * n:(is_ + 1) * n] = freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -self.Lin
        return L

    def build_Gamma(self):
        t, N_H, n, nu = self.tau_j, self.N_H, self.n, self.s.nu
        omega = self.p.omega_d
        k = np.arange(1, N_H + 1)
        Phi = np.empty((t.size, 2 * N_H + 1))
        Phi[:, 0] = 1.0 / np.sqrt(2.0)
        Phi[:, 1::2] = np.sin(np.outer(t, k * omega / nu))
        Phi[:, 2::2] = np.cos(np.outer(t, k * omega / nu))
        return np.kron(Phi, np.eye(n))

    def x_tilde_to_X(self, xt):
        return xt.reshape(self.N, self.n).T

    def X_to_x_tilde(self, X):
        return X.T.reshape(-1)

    def build_b_ext(self):
        n, N_H = self.n, self.N_H
        b = np.zeros(self.dim)
        b[0:n] += self.F0 * np.sqrt(2.0)
        eps = self.p.eps
        b[1] += -eps / 2 * np.sqrt(2.0)
        k_drive = 2 * self.s.nu
        if N_H >= k_drive:
            isd, icd = 1 + 2 * (k_drive - 1), 2 + 2 * (k_drive - 1)
            b[isd * n + 0] += eps / 2.0
            b[icd * n + 1] += -eps / 2.0
        return b

    def residual(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        F, _ = self._rhs_and_jac(X, want_jac=False)
        Nt = F - self.Lin @ X - self.F0[:, None]
        return self.L @ z - self.Gamma_pinv @ self.X_to_x_tilde(Nt) - self.b_ext

    def jacobian(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        _, J = self._rhs_and_jac(X)
        Jbig = lil_matrix((self.n * self.N, self.n * self.N))
        for j in range(self.N):
            rows = slice(j * self.n, (j + 1) * self.n)
            Jbig[rows, rows] = J[:, :, j] - self.Lin
        return self.L - self.Gamma_pinv @ (Jbig.tocsr() @ self.Gamma)

    def solve(self, z0=None, **kwargs):
        if z0 is None:
            z0 = np.zeros(self.dim)
            z0[0:self.n] = self.ref * np.sqrt(2.0)
        sol = root(self.residual, z0, jac=self.jacobian, method='hybr', **kwargs)
        if not sol.success:
            raise RuntimeError(f"HB solve did not converge: {sol.message}")
        return sol.x

    # ---------- reconstruction ----------
    def reconstruct(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        out = {'t': self.tau_j, 'X': X}
        for name, idx in self.index_map.items():
            out[name] = X[idx[0]] + 1j * X[idx[1]] if len(idx) == 2 else X[idx[0]]
        return out

    def husimi(self, X_col, Zgrid):
        """Max-ent Husimi Q on the complex grid Zgrid, for one state column."""
        col = X_col.reshape(self.n, 1)
        T = self.closure.state_to_targets(col[2:])
        sol = self.closure.solve(T, want_jac=False)
        mu = col[0, 0] + 1j * col[1, 0]
        return self.closure.density(T[:, 0], sol['eta'][:, 0], mu, Zgrid)

    def rhs_time(self, t, y):
        F, _ = self._rhs_and_jac(np.asarray(y).reshape(self.n, 1), want_jac=False)
        dy = F[:, 0].copy()
        drive = -1j * self.p.eps / 2 * (1 + np.exp(2j * self.p.omega_d * t))
        dy[0] += drive.real
        dy[1] += drive.imag
        return dy


# ======================================================================
# Self-tests
# ======================================================================
if __name__ == "__main__":
    import time
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(0)

    # ---- 1. closure reproduces its target moments, and Gaussian input
    #         gives Gaussian output ----
    cl = MaxEntClosure(4, n_nodes=32)
    T = np.zeros((cl.n_cons, 3))
    for s, (sx, sy, rxy) in enumerate([(1.0, 1.0, 0.0), (1.3, 0.7, 0.2),
                                       (0.8, 1.1, -0.3)]):
        T[cl.cons.index((2, 0)), s] = sx
        T[cl.cons.index((1, 1)), s] = rxy
        T[cl.cons.index((0, 2)), s] = sy
        # Gaussian 3rd/4th central moments
        T[cl.cons.index((3, 0)), s] = 0
        T[cl.cons.index((2, 1)), s] = 0
        T[cl.cons.index((1, 2)), s] = 0
        T[cl.cons.index((0, 3)), s] = 0
        T[cl.cons.index((4, 0)), s] = 3 * sx**2
        T[cl.cons.index((3, 1)), s] = 3 * sx * rxy
        T[cl.cons.index((2, 2)), s] = sx * sy + 2 * rxy**2
        T[cl.cons.index((1, 3)), s] = 3 * sy * rxy
        T[cl.cons.index((0, 4)), s] = 3 * sy**2
    out = cl.solve(T)
    print(f"[closure] Gaussian targets: converged {out['ok'].all()}, "
          f"max |eta| = {np.abs(out['eta']).max():.2e} (want ~0)")
    assert out['ok'].all() and np.abs(out['eta']).max() < 1e-8

    # ---- 2. non-Gaussian target: does the density reproduce the moments? ----
    T2 = T[:, :1].copy()
    T2[cl.cons.index((3, 0)), 0] = 0.35
    T2[cl.cons.index((2, 1)), 0] = -0.15
    T2[cl.cons.index((4, 0)), 0] = 3.6
    T2[cl.cons.index((2, 2)), 0] = 1.15
    o2 = cl.solve(T2)
    print(f"[closure] skewed target: converged {o2['ok'][0]}, "
          f"residual {o2['resid'][0]:.2e}")
    assert o2['ok'][0]

    # ---- 3. closure Jacobian vs finite differences ----
    h = 1e-6
    base = cl.solve(T2, want_jac=True)
    an = base['dJ'][:, :, 0]
    fd = np.zeros_like(an)
    for c in range(cl.n_cons - 2):
        Tp = T2.copy(); Tp[c + 2, 0] += h
        Tm = T2.copy(); Tm[c + 2, 0] -= h
        fd[:, c] = ((cl.solve(Tp, want_jac=False)['Mout'][:, 0]
                     - cl.solve(Tm, want_jac=False)['Mout'][:, 0]) / (2 * h))
    rel = np.max(np.abs(an - fd)) / np.max(np.abs(fd))
    print(f"[closure] analytic vs finite-difference Jacobian: rel err {rel:.2e}")
    assert rel < 1e-5

    # ---- 4. order 2 max-ent == Gaussian cumulant closure ----
    t0 = time.time()
    for chi_v in (0.0, 0.15, 0.5):
        kerr = KerrParams(Delta=1.0, chi=chi_v, kappa=1.0, eps=3.0, omega_d=1.3)
        hb = MomentHillMethodMaxEnt(kerr, HBSettingsMaxEnt(
            order=2, N_H=8, samples_per_harmonic=16, n_nodes=24))
        z = hb.solve()
        rec = hb.reconstruct(z)
        try:
            from hhb.kerr.hb_moments_kerr_n import (KerrParams as KP2,
                                                    HBSettingsMomentsN,
                                                    MomentHillMethodN)
            hb2 = MomentHillMethodN(KP2(Delta=1.0, chi=chi_v, kappa=1.0, eps=3.0,
                                        omega_d=1.3),
                                    HBSettingsMomentsN(order=2, N_H=8,
                                                       samples_per_harmonic=16))
            z2 = hb2.solve()
            r2 = hb2.reconstruct(z2)
            e = max(np.max(np.abs(rec['mu'] - r2['mu'])),
                    np.max(np.abs(rec['M11'] - r2['k11'])),
                    np.max(np.abs(rec['M20'] - r2['k20'])))
            print(f"[order 2] chi={chi_v}: max |maxent - cumulant| = {e:.2e}")
            assert e < 1e-7
        except ImportError:
            pass
    print(f"  (order-2 checks took {time.time()-t0:.1f}s)")

    # ---- 5. order 4: HB vs direct time integration ----
    for chi_v in (0.0, 0.15, 0.5):
        kerr = KerrParams(Delta=1.0, chi=chi_v, kappa=1.0, eps=3.0, omega_d=1.3)
        hb = MomentHillMethodMaxEnt(kerr, HBSettingsMaxEnt(
            order=4, N_H=8, samples_per_harmonic=16, n_nodes=32))
        t0 = time.time()
        z = hb.solve()
        X = hb.x_tilde_to_X(hb.Gamma @ z)
        T_f = hb.T_fund
        sol = solve_ivp(hb.rhs_time, [0, 120 * T_f], X[:, 0], rtol=1e-9,
                        atol=1e-11, dense_output=True, max_step=T_f / 40)
        Y = sol.sol(hb.tau_j + 119 * T_f)
        err = np.max(np.abs(X - Y))
        print(f"[order 4] chi={chi_v}: HB residual "
              f"{np.linalg.norm(hb.residual(z)):.1e}, HB vs time-int {err:.2e}"
              f"  ({time.time()-t0:.0f}s)")
        assert err < 1e-5

    # ---- 6. THE POINT: no runaway from vacuum where the cumulant closure blew up
    for chi_v in (0.8333, 1.0):
        kerr = KerrParams(Delta=1.0, chi=chi_v, kappa=1.0, eps=3.0, omega_d=1.3)
        hb = MomentHillMethodMaxEnt(kerr, HBSettingsMaxEnt(
            order=4, N_H=8, samples_per_harmonic=16, n_nodes=32))
        T_f = hb.T_fund
        y0 = hb.ref.copy()
        try:
            sol = solve_ivp(hb.rhs_time, [0, 60 * T_f], y0, rtol=1e-8, atol=1e-10,
                            max_step=T_f / 40)
            done = sol.success and sol.t[-1] >= 60 * T_f - 1e-6
        except RuntimeError as e:
            done, sol = False, None
        print(f"[order 4] chi={chi_v}: time integration FROM VACUUM -> "
              f"{'bounded (cumulant closure blew up here)' if done else 'failed'}")
    print("All checks passed.")
