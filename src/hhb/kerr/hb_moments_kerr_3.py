"""
Harmonic-balance solver for the driven-Kerr moment equations at THIRD
cumulant order (cumulants of order >= 4 set to zero).

Tracked objects (9 real DOFs):
    mu        = <alpha>                        (complex mean)
    sigma^2   = <delta_alpha delta_alpha*>     (real, anti-normal ordered)
    tilde_s2  = <delta_alpha^2>                (complex, squeezing)
    kappa_30  = <delta_alpha^3>                (complex, 3rd cumulant)
    kappa_21  = <delta_alpha^2 delta_alpha*>   (complex, 3rd cumulant)

State packing (extends the n=5 Gaussian solver's packing):
    y = [Re(mu), Im(mu), sigma2, Re(ts2), Im(ts2),
         Re(k30), Im(k30), Re(k21), Im(k21)]

The ODEs are NOT hand-transcribed: they are re-derived symbolically at
import time with the exact same FPE moment recursion + cumulant closure
used in Kerr_3rd_order_sympy.ipynb (whose printed checks validate the
derivation, incl. exact reduction to the Gaussian-order tex equations
when kappa_30 = kappa_21 = 0). Only the drive is excluded from the
symbolic RHS: an additive c-number force enters the mean equation alone
(it cancels in every central-moment ODE), so it is handled exactly by
the external-forcing vector b_ext, as in the n=5 solver.

The linear constant-coefficient part Lin is extracted automatically as
the Jacobian of the (drive-free) RHS at y=0, and the constant term F(0)
(the "+kappa" in the sigma^2 equation) goes to b_ext -- so the split
    residual = L@z - b_nl(z) - b_ext
never double-counts linear terms (the chi=0 bug class).

HB structure (Fourier ordering, AFT, Newton) is identical to the n=5
solver in husimi_hill_method_test_kerr.ipynb, itself adapted from
hill_method.py (Detroux et al. 2015).
"""

import numpy as np
import sympy as sp
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
class HBSettingsMoments3:
    N_H: int = 8                 # keep harmonics 0..N_H (odd ones stay zero by parity)
    samples_per_harmonic: int = 64
    nu: int = 1
    n: int = 9                   # see state packing above

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H


# ======================================================================
# Symbolic derivation (same code path as Kerr_3rd_order_sympy.ipynb,
# drive removed). Done ONCE at import, then lambdified.
# ======================================================================
def _derive_rhs_third_order():
    Delta, chi, kap = sp.symbols('Delta chi kappa', real=True)

    mu_r, mu_i = sp.symbols('mu_r mu_i', real=True)
    s2 = sp.symbols('s2', real=True)
    ts2_r, ts2_i = sp.symbols('ts2_r ts2_i', real=True)
    k30_r, k30_i = sp.symbols('k30_r k30_i', real=True)
    k21_r, k21_i = sp.symbols('k21_r k21_i', real=True)

    mu, mus = mu_r + I * mu_i, mu_r - I * mu_i
    ts2, ts2s = ts2_r + I * ts2_i, ts2_r - I * ts2_i
    k30, k30s = k30_r + I * k30_i, k30_r - I * k30_i
    k21, k21s = k21_r + I * k21_i, k21_r - I * k21_i

    # ---- exact FPE moment recursion, DRIVE-FREE (forcing -> b_ext) ----
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

    # ---- cumulant closure: kappa_{jk} = 0 for j+k >= 4 ----
    kappa_ = {
        (2, 0): ts2, (1, 1): s2, (0, 2): ts2s,
        (3, 0): k30, (2, 1): k21, (1, 2): k21s, (0, 3): k30s,
    }
    s, u = sp.symbols('s u')
    K2 = kappa_[(2, 0)] * s**2 / 2 + kappa_[(1, 1)] * s * u + kappa_[(0, 2)] * u**2 / 2
    K3 = (kappa_[(3, 0)] * s**3 / 6 + kappa_[(2, 1)] * s**2 * u / 2
          + kappa_[(1, 2)] * s * u**2 / 2 + kappa_[(0, 3)] * u**3 / 6)
    # exp(K2+K3) truncated at total degree 5 (exact bookkeeping, see ipynb)
    G = sp.expand(1 + K2 + K3 + sp.Rational(1, 2) * K2**2 + K2 * K3)

    def M(j, k):
        if j == 0 and k == 0:
            return sp.Integer(1)
        return sp.expand(sp.diff(G, s, j, u, k).subs({s: 0, u: 0}))

    Mvals = {(j, k): M(j, k) for j in range(6) for k in range(6) if 0 < j + k <= 5}

    def R(m, n):
        total = 0
        for j in range(m + 1):
            for k in range(n + 1):
                Mjk = sp.Integer(1) if (j == 0 and k == 0) else Mvals.get((j, k), sp.Integer(0))
                total += sp.binomial(m, j) * sp.binomial(n, k) * mu**(m - j) * mus**(n - k) * Mjk
        return sp.expand(total)

    def dotR(m, n):
        monomials = moment_rhs_monomials(m, n)
        return sp.expand(sum(coeff * R(mm, nn) for (mm, nn), coeff in monomials.items()))

    # ---- triangular back-substitution to central-moment/cumulant ODEs ----
    dmu, dmus = dotR(1, 0), dotR(0, 1)
    ds2 = sp.expand(dotR(1, 1) - dmu * mus - mu * dmus)
    dts2 = sp.expand(dotR(2, 0) - 2 * mu * dmu)
    dk30 = sp.expand(dotR(3, 0) - 3 * mu**2 * dmu - 3 * dmu * ts2 - 3 * mu * dts2)
    dk21 = sp.expand(dotR(2, 1)
                     - (2 * mu * mus * dmu + mu**2 * dmus)
                     - (dmus * ts2 + mus * dts2)
                     - (2 * dmu * s2 + 2 * mu * ds2))

    # sanity: sigma^2 must stay real, dR01 must be conj(dR10)
    assert sp.simplify(sp.im(sp.expand(ds2))) == 0
    assert sp.simplify(dmus - dmu.conjugate()) == 0

    F = sp.Matrix([
        sp.re(dmu), sp.im(dmu),
        sp.expand(ds2),
        sp.re(dts2), sp.im(dts2),
        sp.re(dk30), sp.im(dk30),
        sp.re(dk21), sp.im(dk21),
    ])
    y = sp.Matrix([mu_r, mu_i, s2, ts2_r, ts2_i, k30_r, k30_i, k21_r, k21_i])
    params = (chi, kappa_sym := kap, Delta)

    # ---- automatic linear/constant/nonlinear split ----
    zero_state = {v: 0 for v in y}
    F0 = sp.expand(F.subs(zero_state))               # constant term (kappa in s2 row)
    Lin_sym = F.jacobian(y).subs(zero_state)          # exact constant-coeff linear part
    N = sp.expand(F - Lin_sym * y - F0)               # genuinely nonlinear remainder

    nvars = 9
    args = tuple(y) + params
    N_funcs = [sp.lambdify(args, N[i], modules='numpy') for i in range(nvars)]
    J = N.jacobian(y)
    J_funcs = [[sp.lambdify(args, J[i, j], modules='numpy') for j in range(nvars)]
               for i in range(nvars)]
    Lin_func = sp.lambdify(params, Lin_sym, modules='numpy')
    F0_func = sp.lambdify(params, F0, modules='numpy')
    # full RHS (drive-free), for the time-integration cross-check
    F_funcs = [sp.lambdify(args, F[i], modules='numpy') for i in range(nvars)]
    return N_funcs, J_funcs, Lin_func, F0_func, F_funcs


_N_SYM, _J_SYM, _LIN_FUNC, _F0_FUNC, _F_SYM_FULL = _derive_rhs_third_order()


class MomentHillMethod3:
    """Third-cumulant-order harmonic-balance solver (n=9 real DOFs)."""

    def __init__(self, kerr: KerrParams, settings: HBSettingsMoments3 = None):
        self.p = kerr
        self.s = settings if settings is not None else HBSettingsMoments3()
        self.n = self.s.n
        self.N_H = self.s.N_H
        self.N = self.s.N
        self.dim = self.n * (2 * self.N_H + 1)

        # one TRUE fundamental period of the HB basis (frequencies k*omega_d/nu):
        # T_fund = 2*pi*nu/omega_d in physical time (CLAUDE.md bug #5).
        self.T_fund = 2 * np.pi * self.s.nu / self.p.omega_d
        self.tau_j = np.linspace(0, self.T_fund, self.N, endpoint=False)

        self._params = (self.p.chi, self.p.kappa, self.p.Delta)
        self.Lin = np.asarray(_LIN_FUNC(*self._params), dtype=float)
        self.F0 = np.asarray(_F0_FUNC(*self._params), dtype=float).ravel()
        self.L = self.build_L()
        self.Gamma = self.build_Gamma()
        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)
        self.b_ext = self.build_b_ext()

    # ---------- first-order harmonic-balance linear operator ----------
    def build_L(self):
        """
        L such that: (d/dt of Fourier series) - Lin*(coeffs) <-> L @ z
        Ordering [c0, s1, c1, ..., s_NH, c_NH], each block in R^n.
        """
        n, N_H, nu = self.n, self.N_H, self.s.nu
        omega = self.p.omega_d
        L = np.zeros((self.dim, self.dim))

        def idx_c0():  return 0
        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        i0 = idx_c0()
        L[i0 * n:(i0 + 1) * n, i0 * n:(i0 + 1) * n] = -self.Lin

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
        return xt.reshape(self.N, self.n).T   # (n, N)

    def X_to_x_tilde(self, X):
        return X.T.reshape(-1)

    def N_time(self, X):
        """Nonlinear part of the RHS, evaluated pointwise in time. X: (9, N)."""
        Ncol = X.shape[1]
        out = np.empty((self.n, Ncol))
        args = tuple(X) + self._params
        for i in range(self.n):
            out[i, :] = np.broadcast_to(_N_SYM[i](*args), (Ncol,))
        return out

    def dN_dX_blocks(self, X):
        Ncol = X.shape[1]
        args = tuple(X) + self._params
        Jarr = np.zeros((self.n, self.n, Ncol))
        for i in range(self.n):
            for j in range(self.n):
                Jarr[i, j, :] = np.broadcast_to(_J_SYM[i][j](*args), (Ncol,))
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

    # ---------- external forcing: F(0) constant + eps drive at k=0,2 ----
    def build_b_ext(self):
        n, N_H = self.n, self.N_H
        b = np.zeros(self.dim)

        def idx_c0():  return 0
        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        # constant part of the drive-free RHS (the "+kappa" in the s2 row);
        # DC basis function is 1/sqrt(2), so the coefficient is value*sqrt(2)
        i0 = idx_c0()
        b[i0 * n:(i0 + 1) * n] += self.F0 * np.sqrt(2.0)

        # mu-eq drive forcing: -i*eps/2*(1 + exp(2i wd t))
        eps = self.p.eps
        b[i0 * n + 1] += -eps / 2 * np.sqrt(2.0)   # DC: Im(mu) row
        k_drive = 2 * self.s.nu                     # harmonic index of exp(2i wd t)
        if N_H >= k_drive:
            isd, icd = idx_sk(k_drive), idx_ck(k_drive)
            # -i*eps/2*exp(2i wd t) = eps/2*sin(2 wd t) - i*eps/2*cos(2 wd t)
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
            z0[2] = np.sqrt(2.0)     # DC sigma^2 = 1 (vacuum-noise level)
        sol = root(self.residual, z0, jac=self.jacobian, method='hybr', **kwargs)
        if not sol.success:
            raise RuntimeError(f"HB solve did not converge: {sol.message}")
        return sol.x

    # ---------- reconstruction helpers ----------
    def reconstruct(self, z):
        """Return dict of moment time series on self.tau_j from HB coefficients."""
        X = self.x_tilde_to_X(self.Gamma @ z)
        return dict(
            t=self.tau_j,
            mu=X[0] + 1j * X[1],
            sig2=X[2],
            ts2=X[3] + 1j * X[4],
            k30=X[5] + 1j * X[6],
            k21=X[7] + 1j * X[8],
        )

    # full RHS including drive, for direct time integration (cross-check)
    def rhs_time(self, t, y):
        args = tuple(y) + self._params
        dy = np.array([float(_F_SYM_FULL[i](*args)) for i in range(self.n)])
        # drive enters the mean equation only: -i*eps/2*(1+exp(2i wd t))
        drive = -1j * self.p.eps / 2 * (1 + np.exp(2j * self.p.omega_d * t))
        dy[0] += drive.real
        dy[1] += drive.imag
        return dy


# ======================================================================
# Self-test: HB vs direct time integration of the same 9 ODEs
# ======================================================================
if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    for chi in (0.0, 0.15):
        kerr = KerrParams(Delta=1.0, chi=chi, kappa=1.0, eps=3.0, omega_d=1.3)
        hb = MomentHillMethod3(kerr, HBSettingsMoments3(N_H=14, samples_per_harmonic=48))

        z_sol = hb.solve()
        print(f"[chi={chi}] HB residual norm: {np.linalg.norm(hb.residual(z_sol)):.3e}")
        rec = hb.reconstruct(z_sol)

        T = hb.T_fund
        n_periods = 600
        y0 = np.zeros(9); y0[2] = 1.0
        sol = solve_ivp(hb.rhs_time, [0, n_periods * T], y0,
                        max_step=T / 300, dense_output=True, rtol=1e-10, atol=1e-12)
        t_last = hb.tau_j + (n_periods - 1) * T   # same absolute phases as tau_j
        Y = sol.sol(t_last)

        names = ['Re mu', 'Im mu', 'sig2', 'Re ts2', 'Im ts2',
                 'Re k30', 'Im k30', 'Re k21', 'Im k21']
        X_hb = hb.x_tilde_to_X(hb.Gamma @ z_sol)
        err = np.max(np.abs(X_hb - Y), axis=1)
        scale = np.maximum(np.max(np.abs(Y), axis=1), 1e-12)
        for nm, e, sc in zip(names, err, scale):
            print(f"    {nm:7s}: max abs err {e:.3e}  (rel {e/sc:.3e})")
        # mixed tolerance: tiny cumulant components are limited by Fourier
        # truncation at N_H=8 (errors drop to ~1e-10 at N_H=14)
        assert np.all(err < 1e-4 * scale + 1e-6), "HB does not match time integration!"
        print(f"[chi={chi}] HB matches direct time integration. Good.")
