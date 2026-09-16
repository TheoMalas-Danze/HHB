# CLAUDE.md

Guidance for working in this repository. Read this before editing code or re-deriving anything.

**Repository layout, setup, how to run things, the closure-investigation record,
and the known-issues list are in [`README.md`](README.md).** This file is the
physics reference: equations, conventions, reference parameters, and the registry
of bugs not to reintroduce. Keep it that way — don't duplicate layout or status
here.

The code is an installed package (`pip install -e .`): solvers live in
`src/hhb/{hill_method.py,cat/hhb_cat.py,kerr/hb_moments_kerr*.py}`, and every
artefact directory comes from `hhb.paths` rather than the working directory.

## What this project is

**Husimi Harmonic Balance (HHB)** — a method for finding the *periodic* non-equilibrium steady
state (NESS) of driven-dissipative bosonic systems, built from three ingredients:

1. **Lindblad → Husimi $Q$**: map the GKSL equation to a PDE for $Q(\alpha,\alpha^*,t)=\langle\alpha|\rho|\alpha\rangle/\pi$ using the standard correspondence rules.
2. **Gaussian closure**: assume $Q$ stays Gaussian and track only means and covariances, giving a closed set of ODEs with periodic coefficients.
3. **Harmonic balance**: expand each moment in a truncated Fourier series and solve the resulting algebraic system by Newton/Powell-hybrid, reusing the classical structural-dynamics HB machinery of Detroux et al. (2015).

The physical target is a **two-mode SNAIL cat qubit**: memory mode $a$ (undamped), lossy buffer
mode $b$, resonance $\omega_b=2\omega_a=\omega_d$, two-photon dissipation stabilizing a cat state
in $a$.

Selling point vs. Fock-space methods: cost is independent of photon number and of the attractor's
distance from the phase-space origin; scaling is $O(N)$ means $+\,O(N^2)$ covariances instead of
exponential. Main weakness: the Gaussian ansatz is uncontrolled exactly where the physics is most
interesting (deep Kerr-cat regime, near bifurcations, bimodal $Q$).

## The equation for the Kerr oscillator
$$
    \dot \rho = -i[\Delta a\da a+\frac{\eps}{2}a\left(1+e^{-2i\omega_d t}\right)+H.c.+\frac{\chi}{2}\left(a\da\right)^2a^2, \rho] + \kappa [a\rho a\da -\frac{1}{2}(a\da a\rho + \rho a\da a)]

$$
$$
    \dot \mu=-i\frac{\eps}{2}\left(1+e^{2i\omega_d t}\right)-(i\Delta+ \kappa/2-2i\chi)\mu -i\chi(\mu^2\mu^*+2\mu\sigma^2+\mu^*\tilde\sigma^2)\\
    \dot \sigma^2 = \kappa (1-\sigma^2)- 2\chi\textrm{Im}[(\mu^*)^2\tilde\sigma^2]\\
    \dot{\tilde\sigma}^2= -(2i\Delta+\kappa-5i\chi+4i\chi|\mu|^2+6i\chi\sigma^2)\tilde\sigma^2 +i\chi\mu^2(1-2\sigma^2)
$$

## The reference equations for the Cat (do not silently modify)

Physical model:

$$
H=\omega_a a^\dagger a+\omega_b b^\dagger b+\varepsilon_d\cos(\omega_d t)(b^\dagger+b)
-2E_J\sin\varepsilon_p\big[\sin\hat x-\hat x\big],\qquad
\hat x=\varphi_a(a+a^\dagger)+\varphi_b(b+b^\dagger),\qquad L=\sqrt{\kappa_b}\,b
$$

Tracked moments (8 complex/real objects, **14 real DOF**):
$\mu_a,\mu_b$ (complex means); $\sigma_a^2,\sigma_b^2$ (real, $\langle\delta a\,\delta a^\dagger\rangle$-type, **anti-normal ordered** so $\sigma^2\to1$ in vacuum); $\tilde\sigma_a^2,\tilde\sigma_b^2$ (complex, $\langle\delta a^2\rangle$-type); $c=\langle\delta a\,\delta b\rangle$, $d=\langle\delta a\,\delta b^\dagger\rangle$.

$$
\begin{aligned}
\dot\mu_a &= -i\omega_a\mu_a + 2iE_J\sin\varepsilon_p\,\varphi_a\big(e^{-K}\cos\bar X-1\big)\\
\dot\mu_b &= -i\omega_b\mu_b-i\varepsilon_d\cos(\omega_d t)-\tfrac{\kappa_b}2\mu_b+2iE_J\sin\varepsilon_p\,\varphi_b\big(e^{-K}\cos\bar X-1\big)\\
\dot\sigma_a^2 &= 2iE_J\sin\varepsilon_p\,\varphi_a e^{-K}(P_a-P_a^*)\sin\bar X\\
\dot\sigma_b^2 &= \kappa_b(1-\sigma_b^2)+2iE_J\sin\varepsilon_p\,\varphi_b e^{-K}(P_b-P_b^*)\sin\bar X\\
\dot{\tilde\sigma}_a^2 &= -2i\omega_a\tilde\sigma_a^2-2iE_J\sin\varepsilon_p\,\varphi_a e^{-K}(2P_a-\varphi_a)\sin\bar X\\
\dot{\tilde\sigma}_b^2 &= -(2i\omega_b+\kappa_b)\tilde\sigma_b^2-2iE_J\sin\varepsilon_p\,\varphi_b e^{-K}(2P_b-\varphi_b)\sin\bar X\\
\dot c &= -\big(i(\omega_a+\omega_b)+\tfrac{\kappa_b}2\big)c-2iE_J\sin\varepsilon_p\,e^{-K}(\varphi_bP_a+\varphi_aP_b-\varphi_a\varphi_b)\sin\bar X\\
\dot d &= \big(i(\omega_b-\omega_a)-\tfrac{\kappa_b}2\big)d+2iE_J\sin\varepsilon_p\,e^{-K}(\varphi_bP_a-\varphi_aP_b^*)\sin\bar X
\end{aligned}
$$

$$
\bar X=\varphi_a(\mu_a+\mu_a^*)+\varphi_b(\mu_b+\mu_b^*),\qquad
K=\varphi_a^2\big(\sigma_a^2+\mathrm{Re}\,\tilde\sigma_a^2-\tfrac12\big)+\varphi_b^2\big(\sigma_b^2+\mathrm{Re}\,\tilde\sigma_b^2-\tfrac12\big)+2\varphi_a\varphi_b[\mathrm{Re}(c)+\mathrm{Re}(d)]
$$
$$
P_a=\varphi_a(\sigma_a^2+\tilde\sigma_a^2)+\varphi_b(c+d),\qquad
P_a^*=\varphi_a(\sigma_a^2+\tilde\sigma_a^{2*})+\varphi_b(c^*+d^*)
$$
$$
P_b=\varphi_b(\sigma_b^2+\tilde\sigma_b^2)+\varphi_a(c+d^*),\qquad
P_b^*=\varphi_b(\sigma_b^2+\tilde\sigma_b^{2*})+\varphi_a(c^*+d)
$$

These are **exact to all orders in $\varphi_a,\varphi_b$** (no small-amplitude Taylor truncation),
because $[a,\hat x]=\varphi_a$ is a c-number and $\hat x$ involves only commuting quadratures, so
$\langle e^{i\hat x}\rangle$ is the ordinary real Gaussian characteristic function. Every formula
has been checked against direct truncated-Fock Lindblad evaluation at the $10^{-7}$–$10^{-9}$
level. The $d\leftrightarrow d^*$ asymmetry between $P_a$ and $P_b$, and the asymmetric $-1$ in
the effective means, are **real** and traceable to $C_{p_ax_b}\neq C_{p_bx_a}$ — not typos.

## Do not reintroduce these bugs

All were found numerically, none were visible by eye. Preserve them as regression tests if tests
are ever added.

1. **Sign of the SNAIL commutator term**: $+2iE_J\sin\varepsilon_p(\sin D_L-\sin D_R)Q$, not $-$. Fixed by requiring cancellation between $\sin\hat x$'s linear term and the explicit $-\hat x$ counter-term.
2. **$P_a^*$ cross term is $\varphi_b(c^*+d^*)$** — conjugate *both*. (`phib*(cc_+d_)` is the classic typo.)
3. **Drive contributes to raw $\langle bb^\dagger\rangle$**: $[b+b^\dagger,bb^\dagger]=b-b^\dagger\neq0$. It cancels only after passing to the covariance $\sigma_b^2$.
4. **Covariance nonlinear terms are pure $\sin\bar X$.** The $\cos\bar X$ pieces cancel exactly against the mean-field contribution in the product rule (e.g. $\dot{\tilde\sigma}_a^2=\frac{d}{dt}\langle a^2\rangle-2\mu_a\dot\mu_a$). Leaving them in badly overstates the covariance dynamics.
5. **`tau_j` must span the true fundamental period $2\pi\nu/\omega_d$**, not $2\pi/\omega_d$. Silently wrong for $\nu\neq1$.
6. **Drive must be placed at harmonic $k=\nu$**, not hardcoded $k=1$.
7. **$\nu$ must match the resonance structure.** With $\omega_b=2\omega_a=\omega_d$, mode $a$ completes half a cycle per drive period, so the true period is $2\times(2\pi/\omega_d)$ → use $\nu=2$. $\nu=1$ simply fails to converge (safe failure, not a wrong answer).
8. **sympy gotchas**: pass `domain=sp.EX` to `Poly()` when coefficients are transcendental (`cos(wd*t)`), or domain detection hangs; prefer `expand()` over `simplify()` on expressions mixing `cos(wd*t)`, `I` and many symbols.

## Conventions

**Fourier ordering** (both solvers, inherited from Detroux):
$z=[c_0,\,s_1,\,c_1,\,s_2,\,c_2,\dots,s_{N_H},\,c_{N_H}]$, each block in $\mathbb R^n$, with basis
$Q(t)=[1/\sqrt2,\ \sin(\omega t/\nu),\ \cos(\omega t/\nu),\dots]$. Index helpers in the code:
`idx_c0()=0`, `idx_sk(k)=1+2(k-1)`, `idx_ck(k)=2+2(k-1)`.

**Time-major stacking**: $\tilde x=[x(t_1);x(t_2);\dots]$, so $\Gamma=\Phi\otimes I_n$ and the
time-domain Jacobian is block diagonal with one $n\times n$ block per sample. (`src/hhb/hill_method.py`
documents the DOF-major alternative in a comment — don't mix the two.)

**State packing in `src/hhb/cat/hhb_cat.py`** (n=14, order matters everywhere: `Lin`, `b_ext`, sympy `F`/`J`):
```
[Re mu_a, Im mu_a, Re mu_b, Im mu_b, sigma_a2, sigma_b2,
 Re tsig_a2, Im tsig_a2, Re tsig_b2, Im tsig_b2, Re c, Im c, Re d, Im d]
```

**Architecture shared by both solvers**: split the ODE into a constant-coefficient linear part
(`Lin` → exact HB operator `L`, no truncation) plus a nonlinear remainder evaluated
pointwise in time via AFT ($\Gamma$, $\Gamma^+$) and differentiated symbolically for the Newton
Jacobian. `hhb_cat.py`'s `build_L` is the **first-order** version ($\nabla-\mathrm{Lin}$);
`hill_method.py`'s is the **second-order** mechanical version ($\nabla^2\otimes M+\nabla\otimes C+I\otimes K$).
Sign conventions on the nonlinear/external split differ between the two files
(`hhb_cat`: `residual = L@z - b_nl - b_ext`; `hill_method`: `L@z + b_nl - b_ext`) — consistent
within each file; check before copying code between them.

**Rotating frame** (for visualization): each moment picks up the sum
of its operators' phases — $\mu_a\to\mu_ae^{i\omega_a t}$, $\tilde\sigma_a^2\to\tilde\sigma_a^2e^{2i\omega_a t}$,
$c\to ce^{i(\omega_a+\omega_b)t}$, $d\to de^{i(\omega_a-\omega_b)t}$, while $\sigma_a^2,\sigma_b^2$
are invariant. Apply in the time domain after reconstructing from Fourier coefficients, using the
*bare* frequencies, not harmonic indices. A converged solution should look nearly static here.

**Physicality check**: with $n=\langle\delta a^\dagger\delta a\rangle=\sigma^2-1$ (our $\sigma^2$ is
anti-normal) and $m=\tilde\sigma^2=\langle\delta a^2\rangle$, a single-mode Gaussian state is
physical **iff $n(n+1)\ge|m|^2$**, i.e. $(\sigma^2-1)\,\sigma^2\ge|\tilde\sigma^2|^2$. Monitor
$(\sigma_a^2-1)\sigma_a^2-|\tilde\sigma_a^2|^2$ over a period: brief shallow dips suggest numerics
(raise $N_H$), deep sustained violations mean the single-Gaussian ansatz is genuinely failing
(bimodal $Q$ near the cat bifurcation).

**Do not use $\sigma^2\ge1+|\tilde\sigma^2|$** (the earlier form here, propagated into the Kerr
notebooks and sweeps). It is strictly stronger than the true bound and is *wrong*: applied to the
exact truncated-Fock Lindblad state — physical by construction, purity 0.92–0.99 — it reports a
violation of $-0.11$ already at $\chi=1/3$, whereas $n(n+1)-|m|^2=+0.007$ there. Every
"unphysical at N/400 phases" warning printed by the Kerr sweeps before 2026-09-15 is spurious.

## Reference parameters

$\omega_a\approx25.34$ (GHz$\times2\pi$), $\omega_b=2\omega_a$, $\omega_d=\omega_b$,
$\varphi_a=0.11$, $\varphi_b=0.204$, $E_J=37.12\times2\pi$, $\kappa_b=3/10.4$,
$\varepsilon_p=0.23$, $\varepsilon_d\approx0.40$.

The coupling that appears **in our equations** is $g=2E_J\sin\varepsilon_p\,\varphi_a^2\varphi_b$.
The Bessel form $J_1(\varepsilon_p)E_J\varphi_a^2\varphi_b$ used in `hill_method.py`'s
`set_epsilon_p` is a flux-pump-derived quantity from a *different* context — don't conflate them.

Known validation result at these parameters: $|\mu_a|^2\approx0.002$ (≈0, as parity requires) and
$\langle a^2\rangle\approx1.28$ — the cat signature. Adiabatic elimination predicts
$|\alpha|^2=\varepsilon_d/g\approx1.53$; the ~16% gap is expected since $g/\kappa_b\approx0.91$
violates $g\ll\kappa_b$. Treat this as the regression benchmark for the full model.

## Known discrepancies to resolve (flag, don't silently "fix")

- `hill_method.py::floquet_coefficients` keeps `idx[:4*self.n]` eigenvalues; Moore's criterion and the paper say $2n$. For $n=2$ that's 8 vs. 4. Possibly intentional margin for `match_eigs`, possibly a bug.
- `hill_method.py::floquet_coefficients` comments label `is_ = 2*k-1` as "cos index" and `ic_ = 2*k` as "sin index", reversed relative to `build_L`'s `idx_sk`/`idx_ck`. The matrix entries look consistent with $\nabla\otimes2M$; the comments are wrong.
- `hill_method.py` uses `tau_j = linspace(0, 100*pi, N)` (≈50 periods) whereas `hhb_cat.py` uses one true period with `endpoint=False`. The least-squares $\Gamma^+$ tolerates the former but it's redundant and inconsistent.
- `hill_method.py::CircuitParams` has `kappa_b = 1/10.4` and `E_J = 37.12*2*pi*1.5`, both differing from the reference parameters above.
