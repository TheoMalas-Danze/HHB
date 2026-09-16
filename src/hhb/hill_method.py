import sympy as sp
import numpy as np
from scipy.special import jv
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eig as scipy_eig

from dataclasses import dataclass

@dataclass
class CircuitParams:
    h_bar: float = 1.0
    e: float = 1.0
    phi_quantum: float = 0.5

    w_a: float = 25.338776456203686
    phi_a: float = 0.11
    phi_b: float = 0.204
    E_J: float = 37.12 * 2 * np.pi * 1.5
    kappa_b: float = 1 / 10.4

    @property
    def w_b(self):
        return 2 * self.w_a

    @property
    def Z_a(self):
        return 2 * self.phi_quantum**2 * self.phi_a**2

    @property
    def Z_b(self):
        return 2 * self.phi_quantum**2 * self.phi_b**2

    @property
    def kappa_b_p(self):
        return self.kappa_b / self.w_b

@dataclass
class HillSettings:
    N_H: int = 16
    samples_per_harmonic: int = 900
    omega: float = 2.0
    nu: int = 2
    n: int = 2

    @property
    def N(self):
        return self.samples_per_harmonic * self.N_H

class HillMethod:
    def __init__(self, circuit, settings):
        self.circuit = circuit
        self.settings = settings
        self.n = settings.n
        self.N_H = settings.N_H
        self.N = settings.N
        self.dim = self.n * (2*self.N_H + 1)

        self.tau_j = np.linspace(0, 100*np.pi, self.N)

        self.M = np.array([[1.0, 0.0], [0.0, 0.25]])

        self.C = np.array([[0.0, 0.0], [0.0, 0.5*self.circuit.kappa_b_p]])

        self.K = np.eye(2)

        self.L = self.build_L()

        self.Gamma = self.build_Gamma()

        self.Gamma_pinv = np.linalg.pinv(self.Gamma, rcond=1e-12)

    def build_L(self):
        """
        Build linear HB operator L for ordering:
        [c0, s1, c1, s2, c2, ..., s_NH, c_NH] with each block in R^n.
        Returns L of shape (n*(2N_H+1), n*(2N_H+1)).
        """
        M, C, K, N_H, omega, nu = self.M, self.C, self.K, self.N_H, self.settings.omega, self.settings.nu
        n = M.shape[0]
        dim = self.dim
        L = np.zeros((dim, dim), dtype=float)

        # helper to locate block indices in z
        def idx_c0():      return 0
        def idx_sk(k):     return 1 + 2*(k-1)
        def idx_ck(k):     return 2 + 2*(k-1)

        # DC: (K) acting on c0 (with your 1/sqrt2 convention, L is still K here
        # because the same scaling appears in Gamma and Gamma^+; keep L=K)
        i0 = idx_c0()
        L[i0*n:(i0+1)*n, i0*n:(i0+1)*n] = K

        # Harmonics
        for k in range(1, N_H+1):
            A = K - ((k*omega/nu)**2)*M  # n×n
            B = (k*omega/nu) * C          # n×n

            is_ = idx_sk(k)
            ic_ = idx_ck(k)

            # sine row blocks
            L[is_*n:(is_+1)*n, is_*n:(is_+1)*n] = A
            L[is_*n:(is_+1)*n, ic_*n:(ic_+1)*n] = -B

            # cosine row blocks
            L[ic_*n:(ic_+1)*n, is_*n:(is_+1)*n] = B
            L[ic_*n:(ic_+1)*n, ic_*n:(ic_+1)*n] = A

        return L

    def build_Gamma(self):
        """
        Build Gamma = I_n ⊗ Phi with omega/nu = 1.

        t   : array of shape (N,)
        N_H : number of harmonics
        n   : number of DOFs
        """
        t, N_H, n, omega, nu = self.tau_j, self.N_H, self.n, self.settings.omega, self.settings.nu
        t = np.asarray(t)
        N = t.size
        k = np.arange(1, N_H + 1)

        # N x N_H
        S = np.sin(np.outer(t, k*omega/nu))
        C = np.cos(np.outer(t, k*omega/nu))

        # Build Phi: N x (2*N_H + 1) with columns [1/sqrt2, sin(1t), cos(1t), sin(2t), cos(2t), ...]
        Phi = np.empty((N, 2 * N_H + 1), dtype=float)
        Phi[:, 0] = 1.0 / np.sqrt(2.0)
        Phi[:, 1::2] = S
        Phi[:, 2::2] = C

        # Gamma: (n*N) x (n*(2*N_H+1))
        Gamma = np.kron(Phi, np.eye(n))
        return Gamma

    def set_epsilon_p(self, epsilon_p, drive_prefactor=8.12):
        self.epsilon_p = epsilon_p
        self.g2 = jv(1,epsilon_p)*self.circuit.E_J*self.circuit.phi_a**2*self.circuit.phi_b # (Leon's paper)
        self.epsilon_d = drive_prefactor*self.g2

        self.beta_a_p, self.beta_b_p = 2*self.circuit.E_J*np.sin(epsilon_p)*self.circuit.phi_a**2/(self.circuit.w_a), 2*self.circuit.E_J*np.sin(epsilon_p)*self.circuit.phi_b**2/(self.circuit.w_b)
        self.B = np.array([self.beta_a_p, self.beta_b_p])  # shape (2,)
        self.eps_d_p = self.epsilon_d*self.circuit.phi_b/self.circuit.w_b
        self.b_ext = self.build_b_ext()



    def build_b_ext(self):
        """
        b_ext in the same ordering as z:
        [c0, s1, c1, s2, c2, ..., s_NH, c_NH], each block in R^2.
        Forcing: f_ext = -sqrt(2)*eps_d' * cos(2 tau) * e2.
        """
        N_H, eps_d_p, nu = self.N_H, self.eps_d_p, self.settings.nu
        n = 2
        b = np.zeros(self.dim, dtype=float)

        def idx_ck(k): return 2 + 2*(k-1)

        if N_H >= nu:
            ic2 = idx_ck(nu)
            # e2 = [0,1]
            b[ic2*n:(ic2+1)*n] = np.array([0.0, -np.sqrt(2.0)*eps_d_p])

        return b


    def x_tilde_to_X(self, tildex):
        # tildex is time-major: [x(t1); x(t2); ...], each x(tj) is length n
        n, N = self.n ,self.N
        return tildex.reshape(N, n).T   # -> (n, N)

    def X_to_x_tilde(self, X):
        # X is (n, N); return time-major vector
        return X.T.reshape(-1)

    def f_nl_time(self, X):
        """
        X: shape (n, N) with n=2
        returns Fnl: shape (n, N)
        """
        B = self.B
        y = np.sqrt(2.0) * (X[0, :] + X[1, :])          # shape (N,)
        g = np.cos(y) - 1.0                             # shape (N,)
        Fnl = -np.sqrt(2.0) * B[:, None] * g[None, :]    # (2,N)
        return Fnl

    def b_nl(self, z):
        Gamma, Gamma_pinv = self.Gamma, self.Gamma_pinv
        x_tilde = Gamma @ z
        X = self.x_tilde_to_X(x_tilde)
        Fnl = self.f_nl_time(X)              # (n,N)
        f_tilde = self.X_to_x_tilde(Fnl)     # (nN,)
        return Gamma_pinv @ f_tilde     # (n*(2N_H+1),)

    def residual(self, z):
        return self.L @ z + self.b_nl(z) - self.b_ext

    def dfnl_dX_blocks(self, X):
        """
        Return list of N blocks J_j = ∂f_nl/∂X at each time sample.
        Each block is shape (2, 2).
        """
        y = np.sqrt(2.0) * (X[0, :] + X[1, :])
        s = np.sin(y)

        base = np.array([
            [self.beta_a_p, self.beta_a_p],
            [self.beta_b_p, self.beta_b_p],
        ])

        J_blocks = [(2.0 * s[j]) * base for j in range(X.shape[1])]

        return J_blocks

    def build_dftilde_dx_tilde(self, X):
        """
        Build block diagonal Jacobian ∂f_tilde/∂x_tilde.

        Shape: (n*N, n*N), with one n×n block per time sample.
        Consistent with time-major stacking:
            x_tilde = [x(t1); x(t2); ...]
        """
        J_blocks = self.dfnl_dX_blocks(X)

        Jbig = lil_matrix((self.n * self.N, self.n * self.N), dtype=float)

        for j, Jj in enumerate(J_blocks):
            rows = slice(j * self.n, (j + 1) * self.n)
            cols = slice(j * self.n, (j + 1) * self.n)
            Jbig[rows, cols] = Jj

        return Jbig.tocsr()

    def db_dz(self, z):
        """
        Compute ∂b_nl/∂z using the chain rule:

            b_nl(z) = Gamma_pinv f_nl(Gamma z)

        so:

            db_nl/dz = Gamma_pinv @ J_time @ Gamma
        """
        x_tilde = self.Gamma @ z
        X = self.x_tilde_to_X(x_tilde)

        Jbig = self.build_dftilde_dx_tilde(X)

        return self.Gamma_pinv @ (Jbig @ self.Gamma)

    def jacobian(self, z):
        return self.L + self.db_dz(z)

    def newton_solve(self, z0, tol=1e-10, maxit=50, verbose=True):
        z = z0.copy()

        for it in range(maxit):
            r = self.residual(z)
            norm_r = np.linalg.norm(r)

            if norm_r < tol:
                if verbose:
                    print("Converged", it, norm_r)
                return z

            J = self.jacobian(z)
            dz = np.linalg.solve(J, -r)

            z = z + dz

            if verbose:
                print(it, norm_r, np.linalg.norm(dz))

        raise RuntimeError("Newton did not converge")

    def powell_hybrid_solve(self, z0, tol=1e-10, maxit=50, use_jac=True, verbose=True):
        """
        Powell hybrid method (MINPACK hybr) via scipy.optimize.root.
        """

        def fun(z):
            return self.residual(z)

        jac = self.jacobian if use_jac else None

        sol = root(
            fun,
            z0,
            jac=jac,
            method="hybr",
            tol=tol,
            options={"maxfev": maxit * (len(z0) + 1)},
        )

        if not sol.success:
            raise RuntimeError(f"Powell hybrid did not converge: {sol.message}")

        if verbose:
            print("Converged", sol.nfev, np.linalg.norm(sol.fun))

        return sol.x

    def plot_phase_space(self, z_solution):
        """
        z_solution: array of length 2*(2*N_H+1)
        """
        N_H = self.N_H

        # time grid (one period)
        t = np.linspace(0, 2*np.pi, 400)

        # split DOFs
        za = z_solution[::2]        # memory
        zb = z_solution[1::2]        # buffer

        # initialize
        x_a = np.zeros_like(t)
        p_a = np.zeros_like(t)
        x_b = np.zeros_like(t)
        p_b = np.zeros_like(t)

        # DC terms
        x_a += za[0] / np.sqrt(2)
        x_b += zb[0] / np.sqrt(2)

        # harmonics
        for k in range(1, N_H + 1):
            s_idx = 1 + 2*(k-1)
            c_idx = 2 + 2*(k-1)

            # memory (a)
            x_a += za[s_idx]*np.sin(k*t) + za[c_idx]*np.cos(k*t)
            p_a += k*za[s_idx]*np.cos(k*t) - k*za[c_idx]*np.sin(k*t)

            # buffer (b)
            x_b += zb[s_idx]*np.sin(k*t) + zb[c_idx]*np.cos(k*t)
            p_b += k/2*zb[s_idx]*np.cos(k*t) - k/2*zb[c_idx]*np.sin(k*t)

        a = (x_a + 1j*p_a)/np.sqrt(2)
        a_R = a*np.exp(1j*t)*1j

        b = (x_b + 1j*p_b)/np.sqrt(2)
        b_R = b*np.exp(2j*t)

        # ---- plots ----
        plt.figure()
        plt.plot(np.real(a_R), np.imag(a_R))
        plt.xlabel(r"$\Re(a_R)$")
        plt.ylabel(r"$\Im(a_R)$")
        plt.title("Phase portrait: memory (a)")
        plt.grid(True)
        plt.gca().set_aspect('equal')

        plt.figure()
        plt.plot(np.real(b_R), np.imag(b_R))
        plt.xlabel(r"$\Re(b_R)$")
        plt.ylabel(r"$\Im(b_R)$")
        plt.title("Phase portrait: buffer (b)")
        plt.grid(True)
        plt.gca().set_aspect('equal')
        plt.show()


    def build_Phi_terms(self, t=None):
        """
        Return Phi, dPhi, ddPhi for basis:
            [1/sqrt2, sin, cos, sin, cos, ...]

        Uses self.tau_j by default.
        """
        if t is None:
            t = self.tau_j

        t = np.asarray(t)
        N = t.size
        N_H = self.N_H

        Phi = np.zeros((N, 2 * N_H + 1))
        dPhi = np.zeros_like(Phi)
        ddPhi = np.zeros_like(Phi)

        Phi[:, 0] = 1.0 / np.sqrt(2.0)

        for k in range(1, N_H + 1):
            is_ = 1 + 2 * (k - 1)
            ic_ = 2 + 2 * (k - 1)

            freq = k * self.settings.omega / self.settings.nu

            Phi[:, is_] = np.sin(freq * t)
            Phi[:, ic_] = np.cos(freq * t)

            dPhi[:, is_] = freq * np.cos(freq * t)
            dPhi[:, ic_] = -freq * np.sin(freq * t)

            ddPhi[:, is_] = -(freq**2) * np.sin(freq * t)
            ddPhi[:, ic_] = -(freq**2) * np.cos(freq * t)

        return Phi, dPhi, ddPhi

    def build_Gamma_from_Phi(self, Phi):
        return np.kron(Phi, np.eye(self.n))

    def check_L_vs_Gamma(self, trials=5, tol=1e-10, verbose=True):
        """
        Check that the Fourier-domain linear operator L is consistent
        with the time-domain operator:

            M x'' + C x' + K x

        using the current Gamma convention.
        """
        Phi, dPhi, ddPhi = self.build_Phi_terms(self.tau_j)

        Gamma = self.build_Gamma_from_Phi(Phi)
        Gamma_d = self.build_Gamma_from_Phi(dPhi)
        Gamma_dd = self.build_Gamma_from_Phi(ddPhi)

        Mbig = np.kron(np.eye(self.N), self.M)
        Cbig = np.kron(np.eye(self.N), self.C)
        Kbig = np.kron(np.eye(self.N), self.K)
        # IMPORTANT: above assumes time-major stacking [x(t1); x(t2); ...]
        # If your tilde x is DOF-major [x1(t1..tN); x2(t1..tN)], use this instead:
        #Mbig = np.kron(M, np.eye(N))
        #Cbig = np.kron(C, np.eye(N))
        #Kbig = np.kron(K, np.eye(N))

        for r in range(trials):
            z = np.random.randn(self.n*(2*self.N_H+1))

            x_tilde = Gamma @ z
            dx_tilde = Gamma_d @ z
            ddx_tilde = Gamma_dd @ z

            lin_time = (
                Mbig @ ddx_tilde
                + Cbig @ dx_tilde
                + Kbig @ x_tilde
            )

            lin_freq_time = Gamma @ (self.L @ z)

            err = np.linalg.norm(lin_time - lin_freq_time) / max(
                1.0,
                np.linalg.norm(lin_time),
            )

            if verbose:
                print(f"trial {r}: relative error = {err:.3e}")

            if err > tol:
                if verbose:
                    print("❌ Inconsistent: likely ordering, sign, or scaling mismatch.")
                return False

        if verbose:
            print("✅ L is consistent with Gamma within tolerance.")

        return True

    def floquet_coefficients(self, z_sol):
        dim = self.dim

        Delta_1 = np.zeros((dim, dim), dtype=float)

        # Put C on every diagonal block
        for i in range(2 * self.N_H + 1):
            rows = slice(i * self.n, (i + 1) * self.n)
            Delta_1[rows, rows] = self.C

        # Coupling between sine/cosine blocks
        for k in range(1, self.N_H + 1):
            is_ = 2 * k - 1 #cos index
            ic_ = 2 * k #sin index

            freq = k * self.settings.omega / self.settings.nu

            s_block = slice(is_ * self.n, (is_ + 1) * self.n)
            c_block = slice(ic_ * self.n, (ic_ + 1) * self.n)

            Delta_1[s_block, c_block] = -2.0 * freq * self.M
            Delta_1[c_block, s_block] =  2.0 * freq * self.M

        Delta_2 = block_diag(*([self.M] * (2 * self.N_H + 1)))

        h_z = self.jacobian(z_sol)

        Z = np.zeros((dim, dim))
        I = np.eye(dim)

        B_1 = np.block([
            [Delta_1, h_z],
            [-I,       Z],
        ])

        B_2 = -np.block([
            [Delta_2, Z],
            [Z,       I],
        ])

        # Prefer solve over explicit inverse
        eigvals, eigvecs = scipy_eig(B_1, B_2)

        # sort by absolute value of imaginary part
        idx = np.argsort(np.abs(np.imag(eigvals)))

        # keep 2n smallest
        idx_selected      = idx[:4*self.n]
        eigvals_selected  = eigvals[idx_selected]
        eigvecs_selected  = eigvecs[:, idx_selected]

        return eigvals_selected, eigvecs_selected

    def match_eigs(self, prev_eigvals, prev_eigvecs, new_eigvals, new_eigvecs,
                    alpha=1.0, beta=5.0):
        """
        Match new eigenpairs to previous ones using both eigenvalue proximity
        and eigenvector overlap (subspace continuity). Returns a permutation
        array 'perm' such that new_eigvals[perm] aligns with prev_eigvals.
        """
        n_eigvals = len(prev_eigvals)
        cost = np.zeros((n_eigvals, n_eigvals))

        # normalize eigenvectors (columns) to unit norm, kill phase ambiguity
        # by fixing the phase of the largest-magnitude component to be real positive
        def normalize(vecs):
            vecs = vecs / np.linalg.norm(vecs, axis=0, keepdims=True)
            for j in range(vecs.shape[1]):
                k = np.argmax(np.abs(vecs[:, j]))
                phase = vecs[k, j] / np.abs(vecs[k, j])
                vecs[:, j] /= phase
            return vecs

        prev_v = normalize(prev_eigvecs.copy())
        new_v  = normalize(new_eigvecs.copy())

        # scale for eigenvalue distance so it's comparable to overlap term
        eig_scale = np.max(np.abs(prev_eigvals)) + 1e-12

        for i in range(n_eigvals):
            for j in range(n_eigvals):
                dval     = np.abs(prev_eigvals[i] - new_eigvals[j]) / eig_scale
                overlap  = np.abs(np.vdot(prev_v[:, i], new_v[:, j]))  # in [0,1]
                dvec     = 1.0 - overlap
                cost[i, j] = alpha * dval + beta * dvec

        _, perm = linear_sum_assignment(cost)
        # row_ind is just 0..n-1 in order since cost is square; perm gives new->prev mapping
        return perm
