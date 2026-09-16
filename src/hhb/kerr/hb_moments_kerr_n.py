"""
Harmonic-balance solver for the driven-Kerr moment equations at ARBITRARY
cumulant truncation order n (cumulants of order >= n+1 set to zero).

Generalizes hb_moments_kerr.py (n=2, Gaussian) and hb_moments_kerr_3.py
(n=3). The ODEs are derived symbolically (once per order, cached) with the
same FPE moment recursion + cumulant closure as Kerr_3rd_order_sympy.ipynb.

Tracked objects: mu = <alpha>, and central cumulants kappa_{jk} of
(delta_alpha, delta_alpha*) for 2 <= j+k <= n, j >= k (kappa_{kj} is the
conjugate; kappa_{jj} is real). Real-DOF count: (n+1)(n+2)/2 - 1
(n=2 -> 5, n=3 -> 9, n=4 -> 14, n=5 -> 20).

State packing (graded: mean, then order 2, order 3, ...; within an order,
(j,k) with j descending; complex objects as Re, Im; kappa_{jj} one real slot):
    n=2: [Re mu, Im mu, Re k20, Im k20, k11]           (k20 = tilde-sigma^2,
                                                        k11 = sigma^2)
    n=3: [... , Re k30, Im k30, Re k21, Im k21]
    n=4: [... , Re k40, Im k40, Re k31, Im k31, k22]
    etc.
NOTE: this differs from the n=5/n=9 modules' packing (there sigma^2 sits at
index 2); use .index_map / .reconstruct rather than hard-coded indices.

Derivation structure (all exact except the closure):
  1. Exact FPE raw-moment recursion (drive EXCLUDED: an additive c-number
     force enters only the mean equation -- it translates the distribution
     and cancels in every central moment at every order -- so it is handled
     exactly by b_ext, as in the lower-order modules).
  2. Central moments from the truncated cumulant-generating function:
     M_jk = d_s^j d_u^k exp(K)|_0 with K = sum kappa_{jk} s^j u^k/(j!k!),
     exp(K) summed to p <= floor((n+2)/2) (higher powers only produce
     degrees > n+2, which the derivatives at 0 never see).
  3. Raw moments by binomial expansion around mu. Probe moments R_{jk}
     (j >= k, j+k <= n) are UNIT LOWER TRIANGULAR in the state when ordered
     by grade (dR_{jk}/dkappa_{jk} = 1, no same-order mixing), so the
     cumulant ODEs follow by generic forward substitution -- no per-order
     hand algebra.
  4. Regression check (run in __main__): F at order n with the order-n
     cumulants set to 0 equals F at order n-1 on the shared rows. For n=3
     this is exactly the notebook's "reduce to the tex equations" check.

The linear part Lin is the Jacobian of the drive-free RHS at y=0 and the
constant term F(0) goes to b_ext, so residual = L@z - b_nl(z) - b_ext never
double-counts linear terms.

WARNING: the symbolic derivation cost grows quickly with n (seconds for
n<=4, tens of seconds for n=5+). Results are cached per order within the
process.
"""

import numpy as np
import sympy as sp
from math import factorial
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
class HBSettingsMomentsN:
    order: int = 3               # cumulant truncation degree n
    N_H: int = 8                 # keep harmonics 0..N_H (odd ones stay zero by parity)
    samples_per_harmonic: int = 64
    nu: int = 1

    @property
    def n(self):
        return (self.order + 1) * (self.order + 2) // 2 - 1

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H


# ======================================================================
# Symbolic derivation for arbitrary truncation order (cached per order)
# ======================================================================
def state_layout(order):
    """The packing: list of ('mu',) / ('kjk', j, k) slot descriptors.

    Complex objects occupy two consecutive real slots (Re, Im); the real
    diagonal cumulants kappa_{jj} occupy one.
    """
    slots = [('mu',)]
    for m in range(2, order + 1):
        for j in range(m, (m - 1) // 2, -1):    # j from m down to ceil(m/2)
            slots.append(('kjk', j, m - j))
    return slots


def _derive(order):
    """Symbolic RHS at cumulant truncation `order`. Returns a dict with the
    lambdified pieces plus the raw sympy objects (for cross-order checks)."""
    if order < 2:
        raise ValueError("cumulant truncation order must be >= 2")

    Delta, chi, kap = sp.symbols('Delta chi kappa', real=True)
    params = (chi, kap, Delta)

    # ---- real-split state symbols ----
    slots = state_layout(order)
    y_syms = []          # flat list of real symbols, in packing order
    kappa_c = {}         # (j,k) -> complex expression, for ALL j+k in 2..order
    mu_r, mu_i = sp.symbols('mu_r mu_i', real=True)
    y_syms += [mu_r, mu_i]
    mu, mus = mu_r + I * mu_i, mu_r - I * mu_i
    for slot in slots[1:]:
        _, j, k = slot
        if j == k:
            kr = sp.Symbol(f'k{j}{k}', real=True)
            y_syms.append(kr)
            kappa_c[(j, k)] = kr
        else:
            kr, ki = sp.symbols(f'k{j}{k}_r k{j}{k}_i', real=True)
            y_syms += [kr, ki]
            kappa_c[(j, k)] = kr + I * ki
            kappa_c[(k, j)] = kr - I * ki
    nvars = len(y_syms)
    assert nvars == (order + 1) * (order + 2) // 2 - 1

    # ---- exact FPE moment recursion, DRIVE-FREE ----
    a, ac = sp.symbols('a ac')
    A = I * Delta * a + I * chi * a**2 * ac + kap / 2 * a
    Ac = -I * Delta * ac - I * chi * ac**2 * a + kap / 2 * ac

    def moment_rhs_monomials(m, n):
        f = a**m * ac**n
        expr = (
            -A * sp.diff(f, a)
            - Ac * sp.diff(f, ac)
            + kap * sp.diff(f, a, ac)
            + I * chi / 2 * (sp.diff(a**2 * f, a, 2) - sp.diff(ac**2 * f, ac, 2))
        )
        poly = sp.Poly(sp.expand(expr), a, ac)
        return {monom: coeff for monom, coeff in poly.terms()}

    # ---- central moments from the truncated CGF ----
    s, u = sp.symbols('s u')
    K = sum(kappa_c[(j, k)] * s**j * u**k / (factorial(j) * factorial(k))
            for (j, k) in kappa_c)
    max_moment = order + 2          # dotR of an order-n probe needs order n+2
    P = max_moment // 2             # cumulants have degree >= 2
    G = sp.Integer(1)
    Kp = sp.Integer(1)
    for p in range(1, P + 1):
        Kp = sp.expand(Kp * K)
        G += Kp / factorial(p)
    G = sp.expand(G)

    Mvals = {}
    for j in range(max_moment + 1):
        for k in range(max_moment + 1 - j):
            if j + k == 0:
                Mvals[(j, k)] = sp.Integer(1)
            elif j + k == 1:
                Mvals[(j, k)] = sp.Integer(0)     # central moments
            else:
                Mvals[(j, k)] = sp.expand(
                    sp.diff(G, s, j, u, k).subs({s: 0, u: 0}))

    def R(m, n):
        total = sp.Integer(0)
        for j in range(m + 1):
            for k in range(n + 1):
                Mjk = Mvals.get((j, k), sp.Integer(0))
                if Mjk == 0:
                    continue
                total += (sp.binomial(m, j) * sp.binomial(n, k)
                          * mu**(m - j) * mus**(n - k) * Mjk)
        return sp.expand(total)

    def dotR(m, n):
        monomials = moment_rhs_monomials(m, n)
        return sp.expand(sum(c * R(mm, nn) for (mm, nn), c in monomials.items()))

    # ---- probe moments, real-split, in the SAME packing order as y ----
    # slot ('mu',)      -> probes Re R10, Im R10
    # slot ('kjk',j,k)  -> probes Re R_jk, Im R_jk  (one probe Re R_jj if j==k)
    probes, dprobes = [], []
    for slot in slots:
        if slot[0] == 'mu':
            Rc, dRc = mu, dotR(1, 0)
            probes += [sp.re(Rc), sp.im(Rc)]
            dprobes += [sp.expand(sp.re(dRc)), sp.expand(sp.im(dRc))]
        else:
            _, j, k = slot
            Rc, dRc = R(j, k), dotR(j, k)
            if j == k:
                assert sp.expand(sp.im(Rc)) == 0 and sp.expand(sp.im(dRc)) == 0, \
                    f"R_{j}{k} / dotR_{j}{k} not real -- derivation bug"
                probes.append(sp.expand(sp.re(Rc)))
                dprobes.append(sp.expand(sp.re(dRc)))
            else:
                probes += [sp.expand(sp.re(Rc)), sp.expand(sp.im(Rc))]
                dprobes += [sp.expand(sp.re(dRc)), sp.expand(sp.im(dRc))]
    assert len(probes) == nvars

    # ---- generic forward substitution: J dy/dt = dprobes, J unit lower
    #      triangular in this ordering ----
    yvec = sp.Matrix(y_syms)
    J = sp.Matrix(probes).jacobian(yvec)
    for i in range(nvars):
        assert sp.expand(J[i, i] - 1) == 0, f"probe {i}: diagonal != 1"
        for l in range(i + 1, nvars):
            assert sp.expand(J[i, l]) == 0, \
                f"probe {i} depends on later state var {l} -- ordering bug"
    F = []
    for i in range(nvars):
        expr = dprobes[i]
        for l in range(i):
            if J[i, l] != 0:
                expr -= J[i, l] * F[l]
        F.append(sp.expand(expr))
    F = sp.Matrix(F)

    # ---- automatic linear/constant/nonlinear split ----
    zero_state = {v: 0 for v in y_syms}
    F0 = sp.expand(F.subs(zero_state))
    Lin_sym = F.jacobian(yvec).subs(zero_state)
    Nnl = sp.expand(F - Lin_sym * yvec - F0)

    args = tuple(y_syms) + params
    Jn = Nnl.jacobian(yvec)
    return dict(
        order=order, nvars=nvars, slots=slots, y_syms=y_syms, params=params,
        F_sym=F, Lin_sym=Lin_sym, F0_sym=F0,
        N_funcs=[sp.lambdify(args, Nnl[i], modules='numpy') for i in range(nvars)],
        J_funcs=[[sp.lambdify(args, Jn[i, j], modules='numpy') for j in range(nvars)]
                 for i in range(nvars)],
        F_funcs=[sp.lambdify(args, F[i], modules='numpy') for i in range(nvars)],
        Lin_func=sp.lambdify(params, Lin_sym, modules='numpy'),
        F0_func=sp.lambdify(params, F0, modules='numpy'),
    )


_DERIVED = {}


def get_derivation(order):
    if order not in _DERIVED:
        _DERIVED[order] = _derive(order)
    return _DERIVED[order]


def check_order_reduction(order):
    """F at `order`, with the order-`order` cumulants set to 0, must equal
    F at order-1 on the shared rows. Returns True or raises AssertionError."""
    hi, lo = get_derivation(order), get_derivation(order - 1)
    n_lo = lo['nvars']
    kill = {v: 0 for v in hi['y_syms'][n_lo:]}
    # map: shared symbols have identical names in both derivations
    for i in range(n_lo):
        diff = sp.expand(hi['F_sym'][i].subs(kill) - lo['F_sym'][i])
        assert diff == 0, f"row {i} differs between order {order} and {order-1}"
    return True


class MomentHillMethodN:
    """Cumulant-truncation-order-n harmonic-balance solver."""

    def __init__(self, kerr: KerrParams, settings: HBSettingsMomentsN = None):
        self.p = kerr
        self.s = settings if settings is not None else HBSettingsMomentsN()
        self.order = self.s.order
        self.d = get_derivation(self.order)
        self.n = self.s.n
        self.N_H = self.s.N_H
        self.N = self.s.N
        self.dim = self.n * (2 * self.N_H + 1)

        # one TRUE fundamental period of the HB basis (CLAUDE.md bug #5)
        self.T_fund = 2 * np.pi * self.s.nu / self.p.omega_d
        self.tau_j = np.linspace(0, self.T_fund, self.N, endpoint=False)

        self._params = (self.p.chi, self.p.kappa, self.p.Delta)
        self.Lin = np.asarray(self.d['Lin_func'](*self._params), dtype=float)
        self.F0 = np.asarray(self.d['F0_func'](*self._params), dtype=float).ravel()
        self.L = self.build_L()
        self.Gamma = self.build_Gamma()
        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)
        self.b_ext = self.build_b_ext()

    # ---------- packing helpers ----------
    @property
    def index_map(self):
        """dict: 'mu' -> (re_idx, im_idx); 'kjk' -> (re_idx, im_idx) or (idx,)."""
        out, i = {}, 0
        for slot in self.d['slots']:
            if slot[0] == 'mu':
                out['mu'] = (i, i + 1); i += 2
            else:
                _, j, k = slot
                if j == k:
                    out[f'k{j}{k}'] = (i,); i += 1
                else:
                    out[f'k{j}{k}'] = (i, i + 1); i += 2
        return out

    # ---------- first-order harmonic-balance linear operator ----------
    def build_L(self):
        n, N_H, nu = self.n, self.N_H, self.s.nu
        omega = self.p.omega_d
        L = np.zeros((self.dim, self.dim))

        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        L[0:n, 0:n] = -self.Lin
        for k in range(1, N_H + 1):
            freq = k * omega / nu
            is_, ic_ = idx_sk(k), idx_ck(k)
            L[is_ * n:(is_ + 1) * n, is_ * n:(is_ + 1) * n] = -self.Lin
            L[is_ * n:(is_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, is_ * n:(is_ + 1) * n] = freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -self.Lin
        return L

    # ---------- AFT machinery ----------
    def build_Gamma(self):
        t, N_H, n, nu = self.tau_j, self.N_H, self.n, self.s.nu
        omega = self.p.omega_d
        k = np.arange(1, N_H + 1)
        S = np.sin(np.outer(t, k * omega / nu))
        C = np.cos(np.outer(t, k * omega / nu))
        Phi = np.empty((t.size, 2 * N_H + 1))
        Phi[:, 0] = 1.0 / np.sqrt(2.0)
        Phi[:, 1::2] = S
        Phi[:, 2::2] = C
        return np.kron(Phi, np.eye(n))

    def x_tilde_to_X(self, xt):
        return xt.reshape(self.N, self.n).T

    def X_to_x_tilde(self, X):
        return X.T.reshape(-1)

    def N_time(self, X):
        Ncol = X.shape[1]
        out = np.empty((self.n, Ncol))
        args = tuple(X) + self._params
        for i in range(self.n):
            out[i, :] = np.broadcast_to(self.d['N_funcs'][i](*args), (Ncol,))
        return out

    def dN_dX_blocks(self, X):
        Ncol = X.shape[1]
        args = tuple(X) + self._params
        Jarr = np.zeros((self.n, self.n, Ncol))
        for i in range(self.n):
            for j in range(self.n):
                Jarr[i, j, :] = np.broadcast_to(
                    self.d['J_funcs'][i][j](*args), (Ncol,))
        return [Jarr[:, :, k] for k in range(Ncol)]

    def build_dNtilde_dx_tilde(self, X):
        blocks = self.dN_dX_blocks(X)
        Jbig = lil_matrix((self.n * self.N, self.n * self.N))
        for j, Jj in enumerate(blocks):
            rows = slice(j * self.n, (j + 1) * self.n)
            Jbig[rows, rows] = Jj
        return Jbig.tocsr()

    def b_nl(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        return self.Gamma_pinv @ self.X_to_x_tilde(self.N_time(X))

    def db_dz(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        return self.Gamma_pinv @ (self.build_dNtilde_dx_tilde(X) @ self.Gamma)

    # ---------- external forcing: F(0) constant + eps drive at k=0,2nu ----
    def build_b_ext(self):
        n, N_H = self.n, self.N_H
        b = np.zeros(self.dim)

        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        b[0:n] += self.F0 * np.sqrt(2.0)            # DC constant term
        eps = self.p.eps
        b[1] += -eps / 2 * np.sqrt(2.0)             # DC drive: Im(mu) row
        k_drive = 2 * self.s.nu                     # harmonic of exp(2i wd t)
        if N_H >= k_drive:
            isd, icd = idx_sk(k_drive), idx_ck(k_drive)
            b[isd * n + 0] += eps / 2.0             # Re(mu) row, sin
            b[icd * n + 1] += -eps / 2.0            # Im(mu) row, cos
        return b

    # ---------- residual / Jacobian / solve ----------
    def residual(self, z):
        return self.L @ z - self.b_nl(z) - self.b_ext

    def jacobian(self, z):
        return self.L - self.db_dz(z)

    def solve(self, z0=None, **kwargs):
        if z0 is None:
            z0 = np.zeros(self.dim)
            # DC sigma^2 = kappa_11 = 1 (vacuum-noise level)
            z0[self.index_map['k11'][0]] = np.sqrt(2.0)
        sol = root(self.residual, z0, jac=self.jacobian, method='hybr', **kwargs)
        if not sol.success:
            raise RuntimeError(f"HB solve did not converge: {sol.message}")
        return sol.x

    # ---------- reconstruction ----------
    def reconstruct(self, z):
        """dict of time series on self.tau_j: 'mu' (complex), and every
        cumulant 'kjk' (complex for j>k, real for j==k). Aliases:
        'sig2' = k11, 'ts2' = k20."""
        X = self.x_tilde_to_X(self.Gamma @ z)
        out = {'t': self.tau_j}
        for name, idx in self.index_map.items():
            out[name] = X[idx[0]] + 1j * X[idx[1]] if len(idx) == 2 else X[idx[0]]
        out['sig2'] = out['k11']
        out['ts2'] = out['k20']
        return out

    # full RHS including drive, for time-integration cross-checks
    def rhs_time(self, t, y):
        args = tuple(y) + self._params
        dy = np.array([float(self.d['F_funcs'][i](*args)) for i in range(self.n)])
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

    # ---- 1. order=2 must reproduce the tex Gaussian equations ----
    t0 = time.time()
    d2 = get_derivation(2)
    print(f"derivation order 2: {time.time()-t0:.1f}s, {d2['nvars']} DOFs")
    Delta, chi, kap = d2['params'][2], d2['params'][0], d2['params'][1]
    mu_r, mu_i, k20r, k20i, s2 = d2['y_syms']
    mu, mus = mu_r + I * mu_i, mu_r - I * mu_i
    ts2, ts2s = k20r + I * k20i, k20r - I * k20i
    tex_dmu = -(I * Delta + kap / 2 - 2 * I * chi) * mu \
              - I * chi * (mu**2 * mus + 2 * mu * s2 + mus * ts2)     # drive-free
    tex_ds2 = kap * (1 - s2) - 2 * chi * sp.im(mus**2 * ts2)
    tex_dts2 = -(2 * I * Delta + kap - 5 * I * chi + 4 * I * chi * mu * mus
                 + 6 * I * chi * s2) * ts2 + I * chi * mu**2 * (1 - 2 * s2)
    F2 = d2['F_sym']
    assert sp.expand(F2[0] - sp.re(tex_dmu)) == 0
    assert sp.expand(F2[1] - sp.im(tex_dmu)) == 0
    assert sp.expand(F2[2] - sp.re(tex_dts2)) == 0
    assert sp.expand(F2[3] - sp.im(tex_dts2)) == 0
    assert sp.expand(F2[4] - sp.expand(tex_ds2)) == 0
    print("order 2 == tex Gaussian equations. Good.")

    # ---- 2. cross-order reduction: F_n|_{kappa_n=0} == F_{n-1} ----
    for n_ord in (3, 4):
        t0 = time.time()
        get_derivation(n_ord)
        print(f"derivation order {n_ord}: {time.time()-t0:.1f}s, "
              f"{get_derivation(n_ord)['nvars']} DOFs")
        check_order_reduction(n_ord)
        print(f"order {n_ord} reduces exactly to order {n_ord-1}. Good.")

    # ---- 3. HB vs direct time integration, orders 2..4 ----
    for n_ord in (2, 3, 4):
        for chi_v in (0.0, 0.15):
            kerr = KerrParams(Delta=1.0, chi=chi_v, kappa=1.0, eps=3.0, omega_d=1.3)
            hb = MomentHillMethodN(kerr, HBSettingsMomentsN(
                order=n_ord, N_H=8, samples_per_harmonic=48))
            z = hb.solve()
            T = hb.T_fund
            y0 = np.zeros(hb.n)
            y0[hb.index_map['k11'][0]] = 1.0
            sol = solve_ivp(hb.rhs_time, [0, 600 * T], y0, max_step=T / 300,
                            dense_output=True, rtol=1e-10, atol=1e-12)
            Y = sol.sol(hb.tau_j + 599 * T)
            X = hb.x_tilde_to_X(hb.Gamma @ z)
            err = np.max(np.abs(X - Y), axis=1)
            scale = np.maximum(np.max(np.abs(Y), axis=1), 1e-12)
            ok = np.all(err < 1e-4 * scale + 1e-6)
            print(f"order {n_ord}, chi={chi_v}: max abs err {err.max():.2e} "
                  f"-> {'OK' if ok else 'FAIL'}")
            assert ok, "HB does not match time integration!"
    print("All checks passed.")
