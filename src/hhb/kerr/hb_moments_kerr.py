"""
Harmonic-balance solver for the driven-Kerr moment equations
    mu_dot        = -i*eps/2*(1+exp(2i wd t)) - (i*Delta + kappa/2 - 2i*chi) mu
                    - i*chi*(mu^2 mu* + 2 mu sigma2 + mu* sigmatilde2)
    sigma2_dot    = kappa*(1-sigma2) - 2*chi*Im[mu*^2 sigmatilde2]
    sigmatilde2_dot = -(2i*Delta+kappa-5i*chi+4i*chi*|mu|^2+6i*chi*sigma2)*sigmatilde2
                      + i*chi*mu^2*(1-2*sigma2)

Adapted from hill_method.py (Detroux et al. 2015 harmonic-balance / AFT structure),
but for a FIRST-ORDER complex ODE system instead of a second-order mechanical one.

State packed as n=5 real DOFs: y = [Re(mu), Im(mu), sigma2, Re(sigmatilde2), Im(sigmatilde2)]
"""

import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix
from scipy.optimize import root
from dataclasses import dataclass


@dataclass
class KerrParams:
    Delta: float
    chi: float
    kappa: float
    eps: float
    omega_d: float


@dataclass
class HBSettingsMoments:
    N_H: int = 8                # keep even harmonics 0..N_H (odd ones stay zero by parity)
    samples_per_harmonic: int = 64
    nu: int = 1
    n: int = 5                   # Re(mu), Im(mu), sigma2, Re(sigmatilde2), Im(sigmatilde2)

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H


# ----------------------------------------------------------------------
# Symbolic derivation of the nonlinear RHS + its Jacobian (done ONCE,
# lambdified for fast numeric evaluation). This avoids re-deriving the
# 5x5 Jacobian by hand -- exactly the kind of algebra that has produced
# sign/index errors earlier in this derivation.
# ----------------------------------------------------------------------
def _build_symbolic_rhs():
    m1, m2, s, t1, t2, chi, kappa, Delta = sp.symbols(
        'm1 m2 s t1 t2 chi kappa Delta', real=True
    )
    mu = m1 + sp.I * m2
    mu_c = m1 - sp.I * m2
    sig = s
    sigt = t1 + sp.I * t2

    dmu = -(sp.I * Delta + kappa / 2 - 2 * sp.I * chi) * mu \
          - sp.I * chi * (mu**2 * mu_c + 2 * mu * sig + mu_c * sigt)
    ds = kappa * (-sig) - 2 * chi * sp.im(mu_c**2 * sigt)
    dsigt = -(2 * sp.I * Delta + kappa - 5 * sp.I * chi
              + 4 * sp.I * chi * mu * mu_c + 6 * sp.I * chi * sig) * sigt \
             + sp.I * chi * mu**2 * (1 - 2 * sig)

    F = sp.Matrix([
        sp.re(dmu), sp.im(dmu), sp.expand(ds), sp.re(dsigt), sp.im(dsigt)
    ])
    y = sp.Matrix([m1, m2, s, t1, t2])

    # Lin, built EXACTLY as in MomentHillMethod._build_Lin -- must match so that
    # N = F_full - Lin @ y keeps only the genuinely nonlinear (state-bilinear/
    # cubic) part. Otherwise the linear part gets double-counted between L
    # (harmonic-balance operator) and b_nl (AFT nonlinear term) -- exactly the
    # bug that produced a spurious factor in the chi=0 test.
    a1, b1 = -kappa / 2, -(Delta - 2 * chi)
    a2, b2 = -kappa, (5 * chi - 2 * Delta)
    Lin_sym = sp.Matrix([
        [a1, -b1, 0, 0, 0],
        [b1,  a1, 0, 0, 0],
        [0,   0, -kappa, 0, 0],
        [0,   0, 0, a2, -b2],
        [0,   0, 0, b2,  a2],
    ])

    N = sp.simplify(F - Lin_sym * y)
    J = N.jacobian(y)

    params = (chi, kappa, Delta)
    args = (m1, m2, s, t1, t2) + params
    N_funcs = [sp.lambdify(args, N[i], modules='numpy') for i in range(5)]
    J_funcs = [[sp.lambdify(args, J[i, j], modules='numpy') for j in range(5)]
               for i in range(5)]
    return N_funcs, J_funcs


_F_SYM, _J_SYM = _build_symbolic_rhs()


class MomentHillMethod:
    def __init__(self, kerr: KerrParams, settings: HBSettingsMoments):
        self.p = kerr
        self.s = settings
        self.n = settings.n
        self.N_H = settings.N_H
        self.N = settings.N
        self.dim = self.n * (2 * self.N_H + 1)

        # one TRUE fundamental period of the HB basis (frequencies k*omega_d/nu):
        # T_fund = 2*pi*nu/omega_d in physical time (CLAUDE.md bug #5).
        self.T_fund = 2 * np.pi * self.s.nu / self.p.omega_d
        self.tau_j = np.linspace(0, self.T_fund, self.N, endpoint=False)

        self.Lin = self._build_Lin()
        self.L = self.build_L()
        self.Gamma = self.build_Gamma()
        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)
        self.b_ext = self.build_b_ext()

    # ---------- linear (constant-coefficient) part of the ODE --------
    def _build_Lin(self):
        p = self.p
        a1, b1 = -p.kappa / 2, -(p.Delta - 2 * p.chi)     # mu block
        a2, b2 = -p.kappa, (5 * p.chi - 2 * p.Delta)      # sigmatilde2 block
        Lin = np.array([
            [a1, -b1, 0, 0, 0],
            [b1,  a1, 0, 0, 0],
            [0,   0, -p.kappa, 0, 0],
            [0,   0, 0, a2, -b2],
            [0,   0, 0, b2,  a2],
        ])
        return Lin

    # ---------- first-order harmonic-balance linear operator ----------
    def build_L(self):
        """
        L such that: (d/dt of Fourier series) - Lin*(coeffs) <-> L @ z
        Ordering [c0, s1, c1, ..., s_NH, c_NH], each block in R^n (n=5).
        """
        n, N_H, nu = self.n, self.N_H, self.s.nu
        omega = self.p.omega_d
        dim = self.dim
        L = np.zeros((dim, dim))

        def idx_c0():  return 0
        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        i0 = idx_c0()
        L[i0 * n:(i0 + 1) * n, i0 * n:(i0 + 1) * n] = -self.Lin

        for k in range(1, N_H + 1):
            freq = k * omega / nu
            is_, ic_ = idx_sk(k), idx_ck(k)
            # derivative block (paper's nabla_k) tensor I_n, minus Lin on diagonal
            L[is_ * n:(is_ + 1) * n, is_ * n:(is_ + 1) * n] = -self.Lin
            L[is_ * n:(is_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, is_ * n:(is_ + 1) * n] = freq * np.eye(n)
            L[ic_ * n:(ic_ + 1) * n, ic_ * n:(ic_ + 1) * n] = -self.Lin

        return L

    # ---------- AFT machinery (identical structure to hill_method.py) --
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
        """Nonlinear part of the RHS, evaluated pointwise in time. X: (5,N)."""
        m1, m2, s, t1, t2 = X
        Ncol = X.shape[1]
        out = np.empty((5, Ncol))
        for i in range(5):
            val = _F_SYM[i](m1, m2, s, t1, t2, self.p.chi, self.p.kappa, self.p.Delta)
            out[i, :] = np.broadcast_to(val, (Ncol,))
        return out

    def dN_dX_blocks(self, X):
        m1, m2, s, t1, t2 = X
        Ncol = X.shape[1]
        Jarr = np.zeros((5, 5, Ncol))
        for i in range(5):
            for j in range(5):
                val = _J_SYM[i][j](m1, m2, s, t1, t2, self.p.chi, self.p.kappa, self.p.Delta)
                Jarr[i, j, :] = np.broadcast_to(val, (Ncol,))
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
        Ftime = self.N_time(X)
        return self.Gamma_pinv @ self.X_to_x_tilde(Ftime)

    def db_dz(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        Jbig = self.build_dNtilde_dx_tilde(X)
        return self.Gamma_pinv @ (Jbig @ self.Gamma)

    # ---------- forcing (kappa constant + eps drive at k=0,2) ----------
    def build_b_ext(self):
        n, N_H = self.n, self.N_H
        b = np.zeros(self.dim)

        def idx_c0():  return 0
        def idx_sk(k): return 1 + 2 * (k - 1)
        def idx_ck(k): return 2 + 2 * (k - 1)

        eps = self.p.eps
        i0 = idx_c0()
        # kappa*(sigma2 eq constant "+kappa"): physical DC value kappa -> c0 = kappa*sqrt2
        b[i0 * n + 2] += self.p.kappa * np.sqrt(2.0)
        # mu-eq forcing: -i*eps/2*(1 + exp(2i wd t))
        #   constant part -i*eps/2 -> Im(mu)-forcing at DC: c0 = -eps/2 * sqrt2
        b[i0 * n + 1] += -eps / 2 * np.sqrt(2.0)
        if N_H >= 2:
            is2, ic2 = idx_sk(2), idx_ck(2)
            # -i*eps/2*exp(2i wd t) = eps/2*sin(2 wd t) - i*eps/2*cos(2 wd t)
            b[is2 * n + 0] += eps / 2.0        # Re(mu) forcing, sin(2 wd t)
            b[ic2 * n + 1] += -eps / 2.0       # Im(mu) forcing, cos(2 wd t)
        return b

    # ---------- residual / Jacobian / solve --------------------------
    def residual(self, z):
        return self.L @ z - self.b_nl(z) - self.b_ext

    def jacobian(self, z):
        return self.L - self.db_dz(z)

    def solve(self, z0, **kwargs):
        sol = root(self.residual, z0, jac=self.jacobian, method='hybr', **kwargs)
        if not sol.success:
            raise RuntimeError(f"HB solve did not converge: {sol.message}")
        return sol.x
