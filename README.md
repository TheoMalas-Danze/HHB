# Husimi Harmonic Balance (HHB)

Periodic non-equilibrium steady states (NESS) of driven-dissipative bosonic
systems, without Fock-space truncation.

> **Scope of this file.** Layout, setup, how to run things, and the record of
> what is validated and what is not. The physics — equations, conventions,
> reference parameters, and the list of bugs not to reintroduce — lives in
> [`CLAUDE.md`](CLAUDE.md) and is not duplicated here.

---

## 1. The method

Three ingredients, applied in order:

1. **Lindblad → Husimi Q.** Map the GKSL equation to a PDE for
   `Q(α,α*,t) = ⟨α|ρ|α⟩/π` via the standard correspondence rules.
2. **Moment closure.** Assume a form for `Q` and track only its low moments,
   giving a closed set of ODEs with periodic coefficients. The closure is the
   scientifically delicate step — see §6.
3. **Harmonic balance.** Expand each moment in a truncated Fourier series and
   solve the resulting algebraic system by Newton/Powell-hybrid, reusing the
   classical structural-dynamics HB machinery of Detroux et al. (2015):
   a constant-coefficient linear part handled exactly, plus a nonlinear
   remainder evaluated pointwise in time by alternating frequency/time (AFT).

**Selling point.** Cost is independent of photon number and of the attractor's
distance from the phase-space origin: `O(N)` means plus `O(N²)` covariances,
rather than exponential in the Fock cutoff.

**Weakness.** The closure is uncontrolled exactly where the physics is most
interesting. Quantifying that is what most of §6 is about.

## 2. The two systems

| | **Kerr oscillator** | **Two-mode SNAIL cat** |
|---|---|---|
| Role | numerical testbed | the physics target |
| Modes | one | memory `a` (undamped) + lossy buffer `b` |
| Reference | `dynamiqs` (exact truncated-Fock Lindblad) | none tractable |
| Code | [`src/hhb/kerr/`](src/hhb/kerr/) | [`src/hhb/cat/`](src/hhb/cat/) |

The Kerr oscillator exists so that every claim about the closure can be checked
against an exact answer. The cat is the goal: `ω_b = 2ω_a = ω_d`, two-photon
dissipation stabilising a cat state in `a`.

## 3. Layout

```
HHB/
├── pyproject.toml              installable package (pip install -e .)
├── README.md                   this file
├── CLAUDE.md                   equations, conventions, bug registry
├── LICENSE  NOTICE             Apache-2.0
│
├── src/hhb/
│   ├── paths.py                artefact directories, resolved from the package
│   ├── plot_setting.py         shared matplotlib style
│   ├── hill_method.py          ORIGINAL 2nd-order "mechanical" Hill/HB solver
│   │                           (∇²⊗M + ∇⊗C + I⊗K). Floquet stability via
│   │                           floquet_coefficients(). The ancestor of
│   │                           everything else here.
│   ├── cat/hhb_cat.py          two-mode SNAIL: 14 real DOF, exact to all
│   │                           orders in φ_a, φ_b
│   └── kerr/
│       ├── hb_moments_kerr.py             n=2  Gaussian closure       (5 DOF)
│       ├── hb_moments_kerr_3.py           n=3  cumulant closure       (9 DOF)
│       ├── hb_moments_kerr_n.py           any n, cumulant closure     ((n+1)(n+2)/2−1 DOF)
│       └── hb_moments_kerr_max_entr_n.py  max-ent closure — NOT APPLICABLE (§6)
│
├── scripts/
│   └── sweep_kerr_hellinger_n.py          the sweep driver (CLI, all orders)
│
├── notebooks/
│   ├── kerr/  husimi_hill_method_test_kerr.ipynb
│   │          sweep_kerr_hellinger.ipynb          n=2 sweep
│   │          sweep_kerr_hellinger_3.ipynb        n=3 sweep
│   │          Kerr_3rd_order_sympy.ipynb          symbolic derivation of the
│   │                                              3rd-order cumulant equations
│   ├── cat/   husimi_hill_method_test_cat.ipynb
│   │          husimi_hill_method_test_cat_simplified.ipynb
│   └── hill/  Instability_threshold_kappa_b_dependence.ipynb
│                                          κ_b-dependence of the instability
│                                          threshold
│
├── data/       TRACKED. kerr/*.npz are the regression baseline (§7);
│               cat/*.json are saved solution branches.
├── figures/    cat/ is TRACKED (paper figures); kerr/ is generated and ignored.
└── media/      NOT TRACKED — 84 MB of Wigner animations (§9).
```

Nothing resolves paths from the current working directory. Scripts and
notebooks import their artefact directories from
[`src/hhb/paths.py`](src/hhb/paths.py), so they behave identically whatever
directory Jupyter or the shell was started in.

### Module lineage

`hb_moments_kerr.py` → `_3.py` → `_n.py` is a progression, not a set of
alternatives. Each supersedes the previous; `_n.py` at `order=2` reproduces
`hb_moments_kerr.py` exactly, and at `order=3` reproduces `_3.py` exactly (both
verified by the module self-tests). The older two are kept because the
`sweep_kerr_hellinger*.ipynb` notebooks import them directly.

From `_n.py` onward the ODEs are **re-derived symbolically at import**, never
hand-transcribed — the FPE moment recursion plus the cumulant closure, with the
triangular structure of the moment→cumulant map asserted rather than assumed.
Hand transcription is where every historical bug in this project came from.

## 4. Setup

Developed against Python 3.11.14 (`/opt/homebrew/bin/python3.11`).

```bash
git clone <this repo> && cd HHB

# the solvers need only numpy/scipy/sympy/matplotlib; the extras are:
#   bench -> dynamiqs + jax, the exact Lindblad reference (sweeps, Kerr notebooks)
#   dev   -> nbstripout + pre-commit (see §8)
#   nb    -> jupyterlab + ipykernel
python3.11 -m pip install -e ".[dev,bench,nb]"

# REQUIRED once per clone: keep notebook outputs out of git (§8)
nbstripout --install --attributes .gitattributes
pre-commit install
```

The editable install is what makes `import hhb` work from any directory, and it
declares `sympy`, which every solver module needs at import time.

## 5. Running things

```bash
# module self-tests (symbolic checks + HB vs. direct time integration)
python3.11 -m hhb.kerr.hb_moments_kerr_n
python3.11 -m hhb.kerr.hb_moments_kerr_3

# Hellinger sweep against dynamiqs, any cumulant order
python3.11 scripts/sweep_kerr_hellinger_n.py --order 3                 # full χ sweep
python3.11 scripts/sweep_kerr_hellinger_n.py --order 4 --chi-max 1.2   # order 4 stops converging past ~1.32
python3.11 scripts/sweep_kerr_hellinger_n.py --order 4 --chi 1.0       # single point
```

Results go to `data/kerr/*.npz` and `figures/kerr/*.png`. The plot overlays
every order whose `.npz` it finds.

> ⚠️ **A single-point run overwrites the full-sweep baseline.** `--chi X`
> writes `data/kerr/sweep_kerr_hellinger_n{order}.npz` under the same name a
> full sweep uses, replacing a 10-point regression record with one point. Check
> `git status` afterwards and `git checkout` the file if you only wanted the
> single number. See §7.

**Derivation cost grows steeply with order.** Measured on an Apple-silicon
laptop with Python 3.11: 0.3 s (n=2), 1.6 s (n=3), 60 s (n=4), minutes at n=6.
It is cached per-process, so one sweep pays it once, but every fresh kernel pays
it again. A disk cache (`sp.srepr` + re-lambdify) is the obvious fix if order ≥ 4
becomes routine.

## 6. Where the science stands

### Validated

HB matches direct time integration of the *same* moment ODEs to 1e-12 (χ=0) and
~1e-6–1e-7 at finite χ (the `N_H=8` Fourier truncation floor, dropping to 1e-10
at `N_H=14`). That is an internal consistency check: it validates the HB/AFT
machinery, not the closure.

### Accuracy against the exact Lindblad answer

Period-averaged Hellinger distance between the reconstructed `Q_HB` and the
`dynamiqs` steady state (Δ=1, κ=1, ε=3, ω_d=1.3, 20 Fock states):

| χ | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 1.0 | 1.17 | 1.2 |
|---|---|---|---|---|---|---|---|---|
| order 2 (Gaussian)  | 0.0052 | 0.0073 | 0.0106 | 0.0150 | 0.0199 | 0.0253 | 0.0319 | — |
| order 3             | 0.0041 | 0.0035 | 0.0047 | 0.0075 | 0.0121 | 0.0185 | 0.0271 | — |
| order 4             | 0.0041 | **0.0030** | **0.0032** | **0.0052** | **0.0100** | 0.0197 | 0.0524 | 0.0470 |

The χ=0 value (0.0076, all orders) is the grid/tolerance floor, not a physics
result. Order 4 wins up to χ≈0.85, crosses order 3 near χ≈1.0, then degrades
sharply.

### Closure error vs. reconstruction error

These are separable, and separating them is the main diagnostic result. Feeding
the order-n Gram–Charlier reconstruction the **exact** cumulants (taken from the
`dynamiqs` density matrix) isolates the reconstruction:

| χ | n=2 | n=3 | n=4 |
|---|---|---|---|
| 0.33 | 0.0071 | 0.0036 | 0.0032 |
| 1.0  | 0.0168 | 0.0089 | 0.0053 |
| 1.2  | 0.0197 | 0.0110 | **0.0065** |

The reconstruction keeps improving monotonically with order everywhere tested —
even at χ=1.2, where the sweep reports 0.048. **So the bottleneck past χ≈1 is
the closure producing wrong cumulants, not the truncated representation of Q.**
At χ=1.0 the HB cumulants already deviate from truth by amounts comparable to
the cumulants themselves (|Δκ₃₀| = 0.055, |Δκ₄₀| = 0.063), which is why the
orders stop separating there.

### The order-4 pathology

Three distinct symptoms, all artefacts of truncating the hierarchy in the
*equations of motion*:

1. **Cumulant drift** — from χ≈1.0, as above.
2. **A spurious fold at χ ≈ 1.078.** With fine continuation (Δχ=0.05,
   substepping to 0.003) the order-4 periodic solution *ceases to exist*. The
   true Lindblad NESS is smooth and unimodal there. The sweep's Δχ≈0.167 step
   jumps the fold and Newton lands on a **disconnected spurious branch** whose
   cumulants are ~3× further from truth — that branch is the entire χ=1.1667
   spike, not a physics result. The fold is detectable without knowing the
   answer: σ_min of the HB Jacobian decays 0.58 → 0.23 → 0.114 → 0.073 as
   χ goes 0 → 0.85 → 1.0 → 1.05.
3. **Finite-time runaway.** Time-integrating the order-4 ODEs from vacuum blows
   up in under one period for χ ≳ 0.83; order 3 stays periodic throughout.
   Started *on* the HB orbit, order 4 stays periodic for 200+ periods, so the
   orbit is stable but its basin has collapsed — the classic quartic
   moment-closure instability.

**Practical consequence:** never initialise the order-4 solve from vacuum;
always continue from a converged neighbour.

### Closure attempts (the record)

| Approach | Positive? | Enough DOF? | Verdict |
|---|---|---|---|
| Cumulant, order 3 | tails can go negative (mass ~1e-4) | yes | **dynamically robust across all χ tested** |
| Cumulant, order 4 | same | yes | best reconstruction, pathological dynamics |
| Maximum entropy | yes, by construction | — | **impossible here** |
| Gaussian scale mixture (NVMM) | yes, by construction | **no** | **too rigid** |

- **Max-ent** (`hb_moments_kerr_max_entr_n.py`): a max-ent density `exp(-P)`,
  `deg P = n`, needs its top-degree form positive, so its tails are *lighter*
  than Gaussian and it is necessarily platykurtic. The Kerr Q is **leptokurtic
  at every χ>0** (max excess kurtosis +0.007 / +0.027 / +0.145 at χ = 1/6, 1/3,
  1). The moments lie outside the set where max-ent has a solution at all.
  It fails *silently* — the inner Newton "converges" to 1e-14 while the
  quadrature integrates a divergent integral; the tell is 51% grid dependence.
  Order 6 does not rescue it. Module retained, with a header, as a record.
- **Gaussian scale mixture** with drift (`ζ = β(s−1) + √s·L·u`, `s~Gamma`):
  positive, unimodal, leptokurtic, analytic moments — but only 6 non-mean
  parameters against 12 moments through degree 4, and its skewness is confined
  to a 2-parameter pattern inside a 4-dimensional space. It captures 25% of the
  skewness and 19% of the kurtosis, ends up only 1.3–2.4× better than a plain
  Gaussian (worse than order-3 Gram–Charlier), and — decisively — cannot even
  hold the covariance it is supposed to track. Not implemented beyond the
  feasibility test.

### Chosen next direction: polynomial transport map

Every attempt so far trades positivity against flexibility. A transport map
breaks the trade: `ζ = T(w)` with `w ~ N(0,I₂)` and `T` polynomial. The density
is a pushforward, hence **positive by construction** and unimodal whenever `T`
is a diffeomorphism (checkable from its Jacobian); moments are exact Gaussian
moments of a polynomial, hence **analytic, no quadrature, no realizability
cliff**; and a cubic `T` carries ~18 coefficients against the 12 moments needed
— surplus flexibility rather than a deficit, for the first time.

Cheaper fallback, if that proves too large a build: keep the order-4 cumulant
closure (its reconstruction is the best measured, 0.0065 at χ=1.2) and stop the
iterate leaving the realizable set — Hankel-matrix positivity as a constraint,
plus pseudo-arclength continuation and σ_min monitoring to traverse the fold
instead of jumping it.

## 7. Known issues

| # | Issue | Status / fix |
|---|---|---|
| 1 | **Solver code is duplicated into notebooks.** Cell 0 of `husimi_hill_method_test_kerr.ipynb` *is* `hb_moments_kerr.py`, and cell 0 of `husimi_hill_method_test_cat.ipynb` *is* `cat/hhb_cat.py`. They can drift apart silently. | **open** — the notebooks should `import hhb…` instead. Deliberately left for a separate pass, since the inlined copies would need diffing against the modules first. |
| 2 | **A single-point sweep overwrites the full-sweep baseline** (§5): `--chi X` writes the same `data/kerr/sweep_kerr_hellinger_n{order}.npz` a 10-point sweep does. Pre-existing, not introduced by the restructuring. | **open** — the single-point branch wants its own filename, or `--no-save`. Until then, watch `git status`. |
| 3 | `hill_method.py` has known internal discrepancies (eigenvalue count, reversed comments, `tau_j` spanning 50 periods, parameters differing from the reference set). | **open** — catalogued in `CLAUDE.md` §"Known discrepancies". |
| 4 | `husimi_hill_method_test_kerr_3.ipynb` was byte-identical to `husimi_hill_method_test_kerr.ipynb` (same md5); it never received its third-order content. | **resolved** — deleted. |
| 5 | 84 MB of GIFs tracked in a repo whose source is ~1 MB. | **resolved** — `media/` is gitignored (§9). |
| 6 | Notebooks hard-coded `sys.path.insert(0, "/Users/…")`, breaking for anyone else. | **resolved** — editable install + `hhb.paths`. |
| 7 | `Cat/Figures/` was silently gitignored by a `**/Figures/` rule, so paper figures were not version-controlled. | **resolved** — now `figures/cat/`, tracked. |
| 8 | The old venv lacked `sympy`, so every solver module failed to import in the project's own kernel. | **resolved** — declared in `pyproject.toml`. |
| 9 | A tracked `__pycache__/*.pyc` predated the `.gitignore`. | **resolved** — removed and ignored. |

## 8. Notebooks are stored without outputs

Notebook outputs are stripped by [nbstripout](https://github.com/kynan/nbstripout)
before anything reaches git. This keeps diffs readable and the repository small
— the seven notebooks were 4.1 MB with outputs and are 168 KB without.

Two independent guards, both configured in-tree:

- a **git filter**, declared in [`.gitattributes`](.gitattributes), activated
  per clone by `nbstripout --install --attributes .gitattributes`;
- a **pre-commit hook**, in [`.pre-commit-config.yaml`](.pre-commit-config.yaml),
  activated by `pre-commit install`, which catches the case where someone
  skipped the filter install.

Consequence to be aware of: **notebooks render without figures on GitHub.**
That is the reason `figures/cat/` and `data/kerr/` are tracked — the results
worth keeping live there as files, not inside notebook metadata.

## 9. Media

`media/` holds ~84 MB of Wigner-function animations (GIF and MP4). It is
deliberately **not** tracked: the source tree is ~3 MB and should stay that
way, and every animation is regenerable by re-running the cells that wrote it.
Notebooks write there via `MEDIA_DIR` from `hhb.paths`.

If these ever need to be shared, Git LFS or external storage is the route — not
a plain `git add -f`.
