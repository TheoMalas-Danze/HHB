"""
Harmonic-balance solver for the two-mode SNAIL moment equations (EXACT,
all-orders in phia, phib -- no cubic-order Taylor truncation):

    dmu_a/dt        = -i*wa*mu_a + nl_mu_a
    dmu_b/dt        = -i*wb*mu_b - i*epsd*cos(wd t) - kappab/2*mu_b + nl_mu_b
    dsigma_a2/dt    =                                                nl_sa2      (no linear decay: mode a undamped)
    dsigma_b2/dt    = -kappab*sigma_b2 + kappab +                    nl_sb2
    dtsigma_a2/dt   = -2i*wa*tsigma_a2 +                             nl_ta2
    dtsigma_b2/dt   = -(kappab+2i*wb)*tsigma_b2 +                    nl_tb2
    dc/dt           = -(kappab/2+i*wa+i*wb)*c +                      nl_c
    dd/dt           = -(kappab/2+i*wa-i*wb)*d +                      nl_d

with (X_bar, K, P_a, P_a*, P_b, P_b* all built purely from the moments):
    Xbar  = phia*(mu_a+mu_a*) + phib*(mu_b+mu_b*)
    K     = phia^2*(sigma_a2+Re(tsigma_a2)-1/2) + phib^2*(sigma_b2+Re(tsigma_b2)-1/2)
            + 2*phia*phib*(Re(c)+Re(d))
    P_a   = phia*(sigma_a2+tsigma_a2)  + phib*(c+d)
    P_a*  = phia*(sigma_a2+tsigma_a2*) + phib*(c*+d*)
    P_b   = phib*(sigma_b2+tsigma_b2)  + phia*(c+d*)
    P_b*  = phib*(sigma_b2+tsigma_b2*) + phia*(c*+d)

    nl_mu_a = 2i*EJ*sinep*phia*(exp(-K)*cos(Xbar)-1)
    nl_mu_b = 2i*EJ*sinep*phib*(exp(-K)*cos(Xbar)-1)
    nl_sa2  =  2i*EJ*sinep*phia*exp(-K)*(P_a-P_a*)*sin(Xbar)
    nl_ta2  = -2i*EJ*sinep*phia*exp(-K)*(2*P_a-phia)*sin(Xbar)
    nl_sb2  =  2i*EJ*sinep*phib*exp(-K)*(P_b-P_b*)*sin(Xbar)
    nl_tb2  = -2i*EJ*sinep*phib*exp(-K)*(2*P_b-phib)*sin(Xbar)
    nl_c    = -2i*EJ*sinep*exp(-K)*(phib*P_a+phia*P_b-phia*phib)*sin(Xbar)
    nl_d    =  2i*EJ*sinep*exp(-K)*(phib*P_a-phia*P_b*)*sin(Xbar)

    (NOTE: these covariance nonlinear terms are pure sin(Xbar) -- the cos(Xbar)
    pieces cancel exactly against the mean-field's own contribution once the
    product rule d(tsigma_a2)/dt = d<a^2>/dt - 2*mu_a*dmu_a/dt etc. is applied
    correctly. An earlier version of this file had cos(Xbar) terms left in by
    mistake -- verified numerically against direct Lindblad simulation and
    corrected.)

State packed as n=14 real DOFs:
    y = [Re(mu_a), Im(mu_a), Re(mu_b), Im(mu_b), sigma_a2, sigma_b2,
         Re(tsigma_a2), Im(tsigma_a2), Re(tsigma_b2), Im(tsigma_b2),
         Re(c), Im(c), Re(d), Im(d)]

IMPORTANT: choose HBSettingsSnail.nu to match resonance conditions. If
wb = 2*wa and wd = wb (the standard cat-qubit two-photon-dissipation setup),
mode a's natural oscillation completes half a cycle per drive period, so the
true steady state has period 2*(2*pi/wd), not 2*pi/wd -- you need nu=2 (the
fundamental becomes wd/2 = wa). Using nu=1 in that case will simply fail to
converge (a safe failure mode, not a silently wrong answer).

KNOWN LIMITATION: for large-amplitude "cat" solutions (mu_a ~ 0, <a^2> large)
under the exact 2:1 resonance, the harmonic-balance Jacobian can become
nearly singular (a "cat orientation" flat direction) and plain Newton from a
naive initial guess may fail to converge. This is a continuation/regularization
issue, not a bug in the equations -- verified that N_time matches the exact
closed forms to machine precision, and that the solver converges cleanly on
generic (non-degenerate) parameter sets.
"""

import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, block_diag as sp_block_diag
from scipy.optimize import root
from dataclasses import dataclass


@dataclass
class SnailParams:
    wa: float
    wb: float
    phia: float
    phib: float
    EJ: float
    sinep: float     # sin(epsilon_p)
    epsd: float
    wd: float
    kappab: float


@dataclass
class HBSettingsSnail:
    N_H: int = 10
    samples_per_harmonic: int = 48
    nu: int = 1
    n: int = 14   # see state ordering in module docstring

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H


# ----------------------------------------------------------------------
# Symbolic derivation of the (purely nonlinear) RHS + its Jacobian.
# N(y) here needs ONLY phia, phib, EJ, sinep -- wa, wb, epsd, wd, kappab
# live entirely in Lin / b_ext.
# ----------------------------------------------------------------------
def _build_symbolic_nonlinear():
    (ma_r, ma_i, mb_r, mb_i, sa2, sb2, ta_r, ta_i, tb_r, tb_i,
     c_r, c_i, d_r, d_i) = sp.symbols(
        'ma_r ma_i mb_r mb_i sa2 sb2 ta_r ta_i tb_r tb_i c_r c_i d_r d_i',
        real=True)
    phia, phib, EJ, sinep = sp.symbols('phia phib EJ sinep', real=True)
    I = sp.I

    mu_a, mu_ac = ma_r + I * ma_i, ma_r - I * ma_i
    mu_b, mu_bc = mb_r + I * mb_i, mb_r - I * mb_i
    ta2, ta2c = ta_r + I * ta_i, ta_r - I * ta_i
    tb2, tb2c = tb_r + I * tb_i, tb_r - I * tb_i
    c_, cc_ = c_r + I * c_i, c_r - I * c_i
    d_, dc_ = d_r + I * d_i, d_r - I * d_i

    Xbar = phia * (mu_a + mu_ac) + phib * (mu_b + mu_bc)
    K = (phia**2 * (sa2 + ta_r - sp.Rational(1, 2))
         + phib**2 * (sb2 + tb_r - sp.Rational(1, 2))
         + 2 * phia * phib * (c_r + d_r))
    expmK = sp.exp(-K)
    cosX, sinX = sp.cos(Xbar), sp.sin(Xbar)

    P_a = phia * (sa2 + ta2) + phib * (c_ + d_)
    P_as = phia * (sa2 + ta2c) + phib * (cc_ + dc_)
    P_b = phib * (sb2 + tb2) + phia * (c_ + dc_)
    P_bs = phib * (sb2 + tb2c) + phia * (cc_ + d_)

    dmu_a = 2 * I * EJ * sinep * phia * (expmK * cosX - 1)
    dmu_b = 2 * I * EJ * sinep * phib * (expmK * cosX - 1)
    dsa2 = 2 * I * EJ * sinep * phia * expmK * (P_a - P_as) * sinX
    dta2 = -2 * I * EJ * sinep * phia * expmK * (2 * P_a - phia) * sinX
    dsb2 = 2 * I * EJ * sinep * phib * expmK * (P_b - P_bs) * sinX
    dtb2 = -2 * I * EJ * sinep * phib * expmK * (2 * P_b - phib) * sinX
    dc_eq = -2 * I * EJ * sinep * expmK * (phib * P_a + phia * P_b - phia * phib) * sinX
    dd_eq = 2 * I * EJ * sinep * expmK * (phib * P_a - phia * P_bs) * sinX

    F = sp.Matrix([
        sp.re(dmu_a), sp.im(dmu_a), sp.re(dmu_b), sp.im(dmu_b),
        sp.re(dsa2), sp.re(dsb2),
        sp.re(dta2), sp.im(dta2), sp.re(dtb2), sp.im(dtb2),
        sp.re(dc_eq), sp.im(dc_eq), sp.re(dd_eq), sp.im(dd_eq),
    ])
    y = sp.Matrix([ma_r, ma_i, mb_r, mb_i, sa2, sb2,
                   ta_r, ta_i, tb_r, tb_i, c_r, c_i, d_r, d_i])
    J = F.jacobian(y)

    params = (phia, phib, EJ, sinep)
    args = tuple(y) + params
    F_funcs = [sp.lambdify(args, F[i], modules='numpy') for i in range(14)]
    J_funcs = [[sp.lambdify(args, J[i, j], modules='numpy') for j in range(14)]
               for i in range(14)]
    return F_funcs, J_funcs


_F_SYM, _J_SYM = _build_symbolic_nonlinear()


def _complex_block(c):
    """2x2 real matrix representing multiplication by complex number c."""
    return np.array([[c.real, -c.imag], [c.imag, c.real]])


class SnailHillMethod:
    def __init__(self, snail: SnailParams, settings: HBSettingsSnail):
        self.p = snail
        self.s = settings
        self.n = settings.n
        self.N_H = settings.N_H
        self.N = settings.N
        self.dim = self.n * (2 * self.N_H + 1)

        self.tau_j = np.linspace(0, 2 * np.pi * settings.nu / snail.wd, self.N, endpoint=False)

        self.Lin = self._build_Lin()
        self.L = self.build_L()
        self.Gamma = self.build_Gamma()
        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)
        self.b_ext = self.build_b_ext()

    def _build_Lin(self):
        p = self.p
        Lin = np.zeros((self.n, self.n))
        Lin[0:2, 0:2] = _complex_block(-1j * p.wa)                       # mu_a
        Lin[2:4, 2:4] = _complex_block(-1j * p.wb - p.kappab / 2)        # mu_b
        Lin[4, 4] = 0.0                                                   # sigma_a2 (undamped!)
        Lin[5, 5] = -p.kappab                                             # sigma_b2
        Lin[6:8, 6:8] = _complex_block(-2j * p.wa)                       # tsigma_a2
        Lin[8:10, 8:10] = _complex_block(-p.kappab - 2j * p.wb)          # tsigma_b2
        Lin[10:12, 10:12] = _complex_block(-p.kappab / 2 - 1j * (p.wa + p.wb))  # c
        Lin[12:14, 12:14] = _complex_block(-p.kappab / 2 - 1j * (p.wa - p.wb))  # d
        return Lin

    def build_L(self):
        n, N_H, nu = self.n, self.N_H, self.s.nu
        omega = self.p.wd
        dim = self.dim
        L = np.zeros((dim, dim))

        def idx_c0():
            return 0

        def idx_sk(k):
            return 1 + 2 * (k - 1)

        def idx_ck(k):
            return 2 + 2 * (k - 1)

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

    def build_Gamma(self):
        t, N_H, n, nu = self.tau_j, self.N_H, self.n, self.s.nu
        omega = self.p.wd
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
        args_state = list(X)
        Ncol = X.shape[1]
        out = np.empty((self.n, Ncol))
        for i in range(self.n):
            val = _F_SYM[i](*args_state, self.p.phia, self.p.phib, self.p.EJ, self.p.sinep)
            out[i, :] = np.broadcast_to(val, (Ncol,))
        return out

    def dN_dX_blocks(self, X):
        args_state = list(X)
        Ncol = X.shape[1]
        Jarr = np.zeros((self.n, self.n, Ncol))
        for i in range(self.n):
            for j in range(self.n):
                val = _J_SYM[i][j](*args_state, self.p.phia, self.p.phib, self.p.EJ, self.p.sinep)
                Jarr[i, j, :] = np.broadcast_to(val, (Ncol,))
        return [Jarr[:, :, k] for k in range(Ncol)]

    #def build_dNtilde_dx_tilde(self, X):
    #    blocks = self.dN_dX_blocks(X)
    #    Jbig = lil_matrix((self.n * self.N, self.n * self.N))
    #    for j, Jj in enumerate(blocks):
    #        rows = slice(j * self.n, (j + 1) * self.n)
    #        Jbig[rows, rows] = Jj
    #    return Jbig.tocsr()
    def build_dNtilde_dx_tilde(self, X):
        blocks = self.dN_dX_blocks(X)
        return sp_block_diag(blocks, format='csr')

    def b_nl(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        Ftime = self.N_time(X)
        return self.Gamma_pinv @ self.X_to_x_tilde(Ftime)

    def db_dz(self, z):
        X = self.x_tilde_to_X(self.Gamma @ z)
        Jbig = self.build_dNtilde_dx_tilde(X)
        return self.Gamma_pinv @ (Jbig @ self.Gamma)

    def build_b_ext(self):
        n, N_H = self.n, self.N_H
        b = np.zeros(self.dim)

        def idx_c0():
            return 0

        def idx_ck(k):
            return 2 + 2 * (k - 1)

        i0 = idx_c0()
        b[i0 * n + 5] += self.p.kappab * np.sqrt(2.0)

        if N_H >= self.s.nu:
            icn = idx_ck(self.s.nu)
            b[icn * n + 3] += -self.p.epsd
        return b

    def residual(self, z):
        return self.L @ z - self.b_nl(z) - self.b_ext

    def jacobian(self, z):
        return self.L - self.db_dz(z)

    def solve(self, z0, **kwargs):
        sol = root(self.residual, z0, jac=self.jacobian, method='hybr', **kwargs)
        if not sol.success:
            raise RuntimeError(f"HB solve did not converge: {sol.message}")
        return sol.x
