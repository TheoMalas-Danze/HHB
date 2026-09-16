"""
Sweep the Kerr strength chi and measure how badly the Gaussian Husimi ansatz
violates the *exact* Husimi PDE.

For each chi: solve the Gaussian (order-2) harmonic-balance moment problem,
reconstruct the moments and their analytic derivatives over one drive period,
and evaluate the residual polynomial of hhb.kerr.gaussian_fp_residual --

    R(alpha, alpha*, t) = dQ/dt|_ansatz - RHS_exact[Q_ansatz] = Q * P(Z, Zb, t)

The diagnostic needs no exact reference: it is computable from the HB solution
alone, unlike the Hellinger sweep, which needs a dynamiqs Lindblad solve.  The
last panel of the figure overlays the two, so one can see whether this cheap
a-priori quantity tracks the measured a-posteriori error.

Cost is a few seconds for the whole sweep (no phase-space grid, no Fock space).

Run:
    python3.11 scripts/sweep_kerr_residual.py
    python3.11 scripts/sweep_kerr_residual.py --N-H 20        # tighter HB
    python3.11 scripts/sweep_kerr_residual.py --chi 1.0        # one point

Writes data/kerr/sweep_kerr_residual.npz and figures/kerr/sweep_kerr_residual.png.
"""

import argparse

import numpy as np

from hhb.paths import KERR_DATA_DIR as DATA_DIR, KERR_FIG_DIR as FIG_DIR
from hhb.kerr.hb_moments_kerr import (KerrParams, HBSettingsMoments,
                                      MomentHillMethod)
from hhb.kerr.gaussian_fp_residual import moments_from_fourier, residual_report

# ----------------------------------------------------------------------
# Fixed physical parameters -- the same ones sweep_kerr_hellinger_n.py uses,
# so the two sweeps are directly comparable
# ----------------------------------------------------------------------
DELTA = 1.0
KAPPA = 1.0
EPS = 3.0
OMEGA_D = 1.3

CHI_VALUES = np.linspace(0.0, 1.5, 10)

# HB settings.  N_H = 14 leaves a pointwise ODE-residual ('proj' below) of
# ~1e-2 at chi = 1.5; it falls by ~30x per 6 harmonics.  The reported norms
# are converged to 6 digits by N_H = 20 -- see the proj panel of the figure.
N_H = 14
SAMPLES_PER_HARMONIC = 48
N_TIMES = 256              # time samples per period for the residual

#: The eight surviving channels of P, listed one per conjugate pair (the
#: partner has the same magnitude: P(Zb, Z) is the conjugate polynomial,
#: because Q is real).  There is no Z^2 Zb^2 monomial at all -- the exact
#: dynamics pumps no isotropic fourth cumulant, only anisotropic ones.
CHANNELS = [(3, 0), (2, 1), (4, 0), (3, 1)]
CHANNEL_TEX = {(3, 0): r'$Z^3$', (2, 1): r'$Z^2\bar Z$',
               (4, 0): r'$Z^4$', (3, 1): r'$Z^3\bar Z$'}


def run_sweep(chi_values, n_h=N_H, n_times=N_TIMES, plot=True, save=True):
    n_chi = len(chi_values)
    l1_avg = np.zeros(n_chi)
    l1_max = np.zeros(n_chi)
    l2_avg = np.zeros(n_chi)
    l2_rel = np.zeros(n_chi)
    proj = np.zeros(n_chi)
    chan = np.zeros((n_chi, len(CHANNELS)))
    l1_t = np.zeros((n_chi, n_times))
    t_grid = None

    z_guess = None                       # numerical continuation in chi
    settings = HBSettingsMoments(N_H=n_h, samples_per_harmonic=SAMPLES_PER_HARMONIC)

    print(f"{'chi':>7} {'<|R|_1>':>10} {'max|R|_1':>10} {'<|R|_2>':>10} "
          f"{'relL2':>8} {'proj':>9}   dominant channel")
    for i, chi in enumerate(chi_values):
        hb = MomentHillMethod(
            KerrParams(Delta=DELTA, chi=chi, kappa=KAPPA, eps=EPS,
                       omega_d=OMEGA_D), settings)
        if z_guess is None:
            z_guess = np.zeros(hb.dim)
            z_guess[2] = np.sqrt(2.0)     # DC sigma^2 = 1 (vacuum)
        z_guess = hb.solve(z_guess)

        mom = moments_from_fourier(z_guess, N_H=hb.N_H, omega_d=OMEGA_D,
                                   nu=hb.s.nu, n_times=n_times)
        rep = residual_report(mom, DELTA, KAPPA, EPS, chi, OMEGA_D)

        t_grid = rep['t']
        l1_avg[i] = rep['l1_avg']
        l1_max[i] = rep['l1'].max()
        l2_avg[i] = rep['l2_avg']
        l2_rel[i] = rep['l2_rel_avg']
        proj[i] = rep['proj_max'].max()
        l1_t[i] = rep['l1']
        rms = rep['channel_rms']
        for j, (m, n) in enumerate(CHANNELS):
            chan[i, j] = rms[(m, n)]
            # the conjugate channel must carry the same magnitude
            assert abs(rms[(n, m)] - rms[(m, n)]) <= 1e-10 * (1 + rms[(m, n)]), \
                f"conjugate channels differ at chi={chi}: {(m, n)}"

        top = CHANNELS[int(np.argmax(chan[i]))]
        print(f"{chi:7.4f} {l1_avg[i]:10.3e} {l1_max[i]:10.3e} "
              f"{l2_avg[i]:10.3e} {l2_rel[i]:8.2e} {proj[i]:9.1e}   "
              f"Z^{top[0]}Zb^{top[1]}")

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fname = DATA_DIR / 'sweep_kerr_residual.npz'
        np.savez(fname, chi=np.asarray(chi_values), l1_avg=l1_avg,
                 l1_max=l1_max, l2_avg=l2_avg, l2_rel=l2_rel, proj=proj,
                 channels=chan, channel_labels=np.array([str(c) for c in CHANNELS]),
                 l1_t=l1_t, t=t_grid, N_H=n_h, Delta=DELTA, kappa=KAPPA,
                 eps=EPS, omega_d=OMEGA_D)
        print(f"saved {fname}")

    if plot and n_chi > 1:
        make_plot(np.asarray(chi_values), l1_avg, l1_max, l2_rel, proj, chan,
                  l1_t, t_grid, n_h)
    return dict(chi=np.asarray(chi_values), l1_avg=l1_avg, l1_max=l1_max,
                l2_avg=l2_avg, l2_rel=l2_rel, proj=proj, channels=chan,
                l1_t=l1_t, t=t_grid)


def make_plot(chi, l1_avg, l1_max, l2_rel, proj, chan, l1_t, t_grid, n_h):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    # ---- (a) the rate, and the relative PDE error ----
    ax = axes[0, 0]
    ax.plot(chi, l1_avg, 'o-', color='C0', label=r'$\langle\|R\|_1\rangle_T$')
    ax.fill_between(chi, l1_avg, l1_max, color='C0', alpha=0.18,
                    label='to worst phase')
    ax.set_xlabel(r'Kerr strength $\chi$')
    ax.set_ylabel(r'$\|R\|_1 / \kappa = \kappa^{-1}\!\int|R|\,d^2\alpha$',
                  color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(chi, 100 * l2_rel, 's--', color='C3',
             label=r'$\|R\|_2/\|\partial_tQ\|_2$')
    ax2.set_ylabel(r'$\|R\|_2\,/\,\|\partial_t Q\|_2$   [%]', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    h1, lb1 = ax.get_legend_handles_labels()
    h2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, lb1 + lb2, fontsize=8, loc='upper left')
    ax.set_title('(a)  how fast the exact dynamics leaves the Gaussian manifold',
                 fontsize=10)

    # ---- (b) per-channel: HOW the ansatz fails ----
    ax = axes[0, 1]
    keep = chan.max(axis=1) > 0        # chi = 0 is identically zero: no log point
    for j, ch in enumerate(CHANNELS):
        deg = sum(ch)
        ax.plot(chi[keep], chan[keep, j], 'o-' if deg == 3 else 's--',
                color=f'C{j}', label=f'{CHANNEL_TEX[ch]}  (deg {deg})')
    ax.set_yscale('log')
    ax.set_xlabel(r'Kerr strength $\chi$')
    ax.set_ylabel('RMS coefficient over one period')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('(b)  which cumulant channel is being pumped\n'
                 '(conjugate partner has equal magnitude)', fontsize=10)

    # ---- (c) phase resolution over one period ----
    ax = axes[1, 0]
    T = t_grid[-1] + (t_grid[1] - t_grid[0])
    show = [i for i in range(len(chi)) if i % 2 == 1] or list(range(len(chi)))
    for k, i in enumerate(show):
        ax.plot(t_grid / T, l1_t[i], color=plt.cm.viridis(k / max(len(show) - 1, 1)),
                label=rf'$\chi={chi[i]:.2f}$')
    ax.set_xlabel(r'phase $t/T$ over one HB period $T=2\pi\nu/\omega_d$')
    ax.set_ylabel(r'$\|R(t)\|_1$')
    ax.set_ylim(0.0, 1.3 * l1_t.max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc='upper center')
    ax.set_title('(c)  the residual is strongly phase-dependent', fontsize=10)

    # ---- (d) vs. the measured Hellinger error, + the HB truncation floor ----
    ax = axes[1, 1]
    ax.plot(chi, l1_avg / l1_avg.max(), 'o-', color='C0',
            label=r'$\|R\|_1$ (a priori, this sweep)')
    try:
        ref = np.load(DATA_DIR / 'sweep_kerr_hellinger.npz')
        ax.plot(ref['chi'], ref['mean'] / ref['mean'].max(), '^-', color='C2',
                label='Hellinger vs. dynamiqs (a posteriori)')
    except FileNotFoundError:
        pass
    ax.set_xlabel(r'Kerr strength $\chi$')
    ax.set_ylabel('normalised to own maximum')
    ax.set_ylim(0.0, 1.08)
    ax.grid(alpha=0.3)
    ax4 = ax.twinx()
    ax4.plot(chi, proj, 'x:', color='grey',
             label=rf'HB pointwise ODE residual ($N_H={n_h}$)')
    ax4.set_yscale('log')
    ax4.set_ylabel('max pointwise HB ODE residual', color='grey', fontsize=9)
    ax4.tick_params(axis='y', labelcolor='grey')
    h1, lb1 = ax.get_legend_handles_labels()
    h4, lb4 = ax4.get_legend_handles_labels()
    ax.legend(h1 + h4, lb1 + lb4, fontsize=8, loc='upper left')
    ax.set_title('(d)  cheap diagnostic vs. measured error\n'
                 'grey: the numerics floor, a separate absolute scale',
                 fontsize=10)

    fig.suptitle(r'Gaussian-ansatz residual on the exact Husimi equation   '
                 rf'($\Delta={DELTA:g}$, $\kappa={KAPPA:g}$, $\epsilon={EPS:g}$,'
                 rf' $\omega_d={OMEGA_D:g}$, $N_H={n_h}$)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figname = FIG_DIR / 'sweep_kerr_residual.png'
    fig.savefig(figname, dpi=150)
    print(f"saved {figname}")
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--chi', type=float, default=None,
                    help='run a single chi instead of the full sweep '
                         '(does not overwrite the sweep .npz)')
    ap.add_argument('--chi-max', type=float, default=None,
                    help='truncate the sweep at this chi')
    ap.add_argument('--N-H', type=int, default=N_H, dest='n_h',
                    help=f'harmonics kept by the HB solve (default {N_H})')
    ap.add_argument('--n-times', type=int, default=N_TIMES,
                    help=f'time samples per period (default {N_TIMES})')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    if args.chi is not None:
        # single point: report only, never touch the sweep baseline
        # (known issue #2 in README -- do not repeat it here)
        run_sweep(np.array([args.chi]), n_h=args.n_h, n_times=args.n_times,
                  plot=False, save=False)
        raise SystemExit(0)

    chis = CHI_VALUES
    if args.chi_max is not None:
        chis = np.unique(np.append(chis[chis < args.chi_max], args.chi_max))
    run_sweep(chis, n_h=args.n_h, n_times=args.n_times, plot=not args.no_plot)
