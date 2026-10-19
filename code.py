# -*- coding: utf-8 -*-
"""
Outputs (Results_Final_v11/):
    Fig1_degradation_curve.tiff
    Fig2_spectral_scaling.tiff
    Fig3_eta_scaling.tiff
    Fig4_bottom_diagnostic.tiff
    raw_results.csv
    summary.csv
    calibration_constants.json
"""

import os
import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import t as student_t

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'tiff',
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
})

COLORS = {
    'nu100': '#1f77b4',
    'nu30':  '#ff7f0e',
    'nu10':  '#d62728',
    'nu3':   '#9467bd',
    'threshold': '#7f7f7f',
    'tolerance': '#2ca02c',
    'bottom': '#8c564b',
}

RESULTS_DIR = 'Results_Final_v11'
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. Synthetic covariance  
# =============================================================================
class SyntheticCovariance:
    def __init__(self, p=100, delta=0.01, k=10,
                 signal_spacing=0.1, eps_p=1e-3, seed=42):
        self.p = p
        self.delta = delta
        self.k = k
        self.signal_spacing = signal_spacing
        self.eps_p = eps_p
        self.seed = seed
        self.Sigma = self._build()

    def _build(self):
        p, k, delta = self.p, self.k, self.delta
        rng = np.random.RandomState(self.seed)

        signal_top = 1.0
        signal_bottom = signal_top - (k - 1) * self.signal_spacing
        signal_block = np.linspace(signal_top, signal_bottom, k)

        mid_top = signal_bottom - 0.05
        mid_bottom = mid_top - self.eps_p
        n_middle = p - k - 1
        middle_block = np.linspace(mid_top, mid_bottom, n_middle)

        lambda_bottom = mid_bottom - delta

        eigenvalues = np.concatenate([signal_block, middle_block, [lambda_bottom]])
        eigenvalues = np.sort(eigenvalues)[::-1]

        # [v9 FIX] shift spectrum so that the smallest eigenvalue is > 0.
        shift = max(0.0, -eigenvalues.min() + 1e-3)
        eigenvalues = eigenvalues + shift

        A = rng.randn(p, p)
        Q, _ = np.linalg.qr(A)

        Sigma = Q @ np.diag(eigenvalues) @ Q.T
        Sigma = (Sigma + Sigma.T) / 2.0
        Sigma = Sigma / np.linalg.norm(Sigma, 2)
        return Sigma

    def true_local_gap(self):
        eigvals = np.sort(np.linalg.eigvalsh(self.Sigma))[::-1]
        return float(eigvals[-2] - eigvals[-1])

    def signal_gap(self):
        eigvals = np.sort(np.linalg.eigvalsh(self.Sigma))[::-1]
        return float(np.min(np.abs(np.diff(eigvals[:self.k]))))


# =============================================================================
# 2. Perturbation and epsilon
# =============================================================================
def sample_perturbation(p, nu, seed, normalize='fro'):
    rng = np.random.RandomState(seed)
    raw = student_t.rvs(df=nu, size=(p, p), random_state=rng)
    if nu > 2:
        raw = raw / np.sqrt(nu / (nu - 2.0))
    raw = (raw + raw.T) / 2.0
    if normalize == 'fro':
        nrm = np.linalg.norm(raw, 'fro')
        if nrm > 0:
            raw = raw / nrm
    return raw


def moment_matching_error(nu, n_samples=200000, seed=0):
    rng = np.random.RandomState(seed)
    r = student_t.rvs(df=nu, size=n_samples, random_state=rng)
    if nu > 2:
        r = r / np.sqrt(nu / (nu - 2.0))
    z = rng.randn(n_samples)
    m3_r, m4_r = np.mean(r ** 3), np.mean(r ** 4)
    m3_z, m4_z = np.mean(z ** 3), np.mean(z ** 4)
    return float(max(abs(m3_r - m3_z), abs(m4_r - m4_z)))


# =============================================================================
# 3. Frobenius deviation + bottom diagnostic   
# =============================================================================
def frobenius_deviation(Sigma, G_rad, eta, k=10, return_pert_norm=False):
    p = Sigma.shape[0]

    # ---- Original eigendecomposition ----
    eigvals_orig, eigvecs_orig = np.linalg.eigh(Sigma)
    idx_orig = np.argsort(eigvals_orig)[::-1]

    U_orig = eigvecs_orig[:, idx_orig[:k]]
    u_bottom_orig = eigvecs_orig[:, idx_orig[p - 1]]
    if u_bottom_orig[np.argmax(np.abs(u_bottom_orig))] < 0:
        u_bottom_orig = -u_bottom_orig

    # ---- Perturbation ----
    Sigma_sqrt = (eigvecs_orig
                  @ np.diag(np.sqrt(np.maximum(eigvals_orig, 0)))
                  @ eigvecs_orig.T)
    P = Sigma_sqrt @ G_rad @ Sigma_sqrt
    Sigma_pert = Sigma + eta * P
    Sigma_pert = (Sigma_pert + Sigma_pert.T) / 2.0

    # ---- Perturbed eigendecomposition ----
    eigvals_pert, eigvecs_pert = np.linalg.eigh(Sigma_pert)
    idx_pert = np.argsort(eigvals_pert)[::-1]

    U_pert = eigvecs_pert[:, idx_pert[:k]]
    u_bottom_pert = eigvecs_pert[:, idx_pert[p - 1]]

    s_b = np.sign(np.dot(u_bottom_orig, u_bottom_pert))
    if s_b == 0:
        s_b = 1.0
    u_bottom_pert = s_b * u_bottom_pert

    signs = np.sign(np.sum(U_orig * U_pert, axis=0))
    signs[signs == 0] = 1.0
    U_pert = U_pert * signs[np.newaxis, :]

    # ---- Deviations ----
    D_signal = float(np.linalg.norm(U_orig - U_pert, 'fro'))
    D_bottom = float(np.linalg.norm(u_bottom_orig - u_bottom_pert))

    cos_bottom = float(np.clip(np.abs(np.dot(u_bottom_orig, u_bottom_pert)),
                               0.0, 1.0))
    sin_bottom = float(np.sqrt(max(0.0, 1.0 - cos_bottom ** 2)))

    U_orig_all = np.column_stack([U_orig, u_bottom_orig])
    U_pert_all = np.column_stack([U_pert, u_bottom_pert])
    D_total = float(np.linalg.norm(U_orig_all - U_pert_all, 'fro'))

    if return_pert_norm:
        return (D_total, D_signal, D_bottom, sin_bottom,
                float(eta * np.linalg.norm(P, 'fro')))
    return D_total, D_signal, D_bottom, sin_bottom


# =============================================================================
# 4. Single condition
# =============================================================================
def run_single_condition(delta, nu, n_train, eta=0.1, p=100, k=10, seed=42):
    cov = SyntheticCovariance(p=p, delta=delta, k=k, seed=seed)
    Sigma = cov.Sigma
    M_hat_true = cov.true_local_gap() / np.linalg.norm(Sigma, 2)
    sig_gap = cov.signal_gap()

    G_rad = sample_perturbation(p=p, nu=nu, seed=seed + 1000, normalize='fro')
    D_total, D_signal, D_bottom, sin_bottom, pert_norm = frobenius_deviation(
        Sigma, G_rad, eta=eta, k=k, return_pert_norm=True
    )

    epsilon = moment_matching_error(nu)

    stat_term = np.sqrt(p * np.log(p)) / (eta * np.sqrt(n_train))
    moment_term = epsilon * (p ** 0.25)
    spectral_term = eta / max(M_hat_true, 1e-6)

    return {
        'delta': delta, 'nu': nu, 'n_train': n_train, 'eta': eta,
        'M_hat': M_hat_true,
        'signal_gap': sig_gap,
        'D_total': D_total,
        'D_signal': D_signal,
        'D_bottom': D_bottom,
        'sin_bottom': sin_bottom,
        'pert_norm': pert_norm,
        'epsilon': epsilon,
        'stat_term': stat_term,
        'moment_term': moment_term,
        'spectral_term': spectral_term,
        'seed': seed,
    }


# =============================================================================
# 5. Calibration
# =============================================================================
def calibrate_C1_C2(df):
    sub = df[df['delta'] >= 0.4].copy()
    sub = sub[np.isfinite(sub['D_signal'])
              & np.isfinite(sub['stat_term'])
              & np.isfinite(sub['moment_term'])]
    if len(sub) < 5:
        return 1.0, 1.0
    A = np.vstack([sub['stat_term'].values, sub['moment_term'].values]).T
    b = sub['D_signal'].values
    try:
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        C1, C2 = float(coef[0]), float(coef[1])
        if not np.isfinite(C1) or C1 <= 0:
            C1 = 1.0
        if not np.isfinite(C2) or C2 <= 0:
            C2 = 1.0
    except Exception:
        C1, C2 = 1.0, 1.0
    return C1, C2


def calibrate_C3(df, eta, baseline_delta=0.4, metric='sin_bottom'):
    ref = df[(df['n_train'] == 2000) & (df['nu'] == 100)
             & np.isclose(df['eta'], eta)]
    if len(ref) == 0:
        return np.nan

    base_rows = ref[np.isclose(ref['delta'], baseline_delta)]
    if len(base_rows) == 0:
        return np.nan
    D0 = float(np.mean(base_rows[metric].values))

    sub = ref[ref['delta'] < 0.4].copy()
    sub = sub[np.isfinite(sub['M_hat']) & (sub['M_hat'] > 1e-8)]
    sub = sub[np.isfinite(sub[metric])]
    if len(sub) < 3:
        return np.nan

    x = 1.0 / sub['M_hat'].values
    y = sub[metric].values - D0
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan

    pos = y > 0
    if np.sum(pos) < 3:
        return np.nan
    x, y = x[pos], y[pos]

    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return np.nan
    slope = float(np.dot(x, y) / denom)
    C3 = slope / eta
    if not np.isfinite(C3) or C3 <= 0:
        return np.nan
    return C3


# =============================================================================
# 6. Visualization (FINAL VERSION, ALIGNED WITH SECTION IV)
# =============================================================================
def plot_degradation_curve(df, theta, delta_op, save_path, eta_ref=0.1):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    nu_colors = {100: COLORS['nu100'], 30: COLORS['nu30'],
                 10: COLORS['nu10'], 3: COLORS['nu3']}

    sub_eta = df[np.isclose(df['eta'], eta_ref)]
    for nu in [100, 30, 10, 3]:
        sub = sub_eta[sub_eta['nu'] == nu].groupby('delta').agg(
            D_mean=('sin_bottom', 'mean'),
            D_std=('sin_bottom', 'std'),
            count=('sin_bottom', 'count')
        ).reset_index().sort_values('delta')
        if len(sub) == 0:
            continue
        sub['D_sem'] = sub['D_std'] / np.sqrt(sub['count'].clip(lower=1))
        ax.errorbar(
            sub['delta'], sub['D_mean'],
            yerr=sub['D_sem'],
            marker='o', markersize=5, linewidth=1.8,
            color=nu_colors[nu], label=f'$\\nu$ = {nu}',
            capsize=3, capthick=1.0, elinewidth=1.0,
        )

    if np.isfinite(delta_op):
        ax.axhline(delta_op, color=COLORS['tolerance'], linestyle='--',
                   linewidth=1.5, label=f'$\\delta_{{op}}$ = {delta_op}')
    if np.isfinite(theta):
        ax.axvline(theta, color=COLORS['threshold'], linestyle='--',
                   linewidth=1.5, label=f'$\\theta$ = {theta:.4f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Spectral margin $\\hat{M}(\\Sigma) = \\Delta$')
    ax.set_ylabel('$\\sin\\Theta$ of bottom eigenvector')
    ax.set_title(f'Degradation curve ($\\eta$ = {eta_ref})')
    
    #  
    ax.legend(frameon=False, ncol=2, loc='best', fontsize=10)
    
    ax.grid(True, which='both', alpha=0.2, linestyle=':')

    xmin, xmax = ax.get_xlim()
    if np.isfinite(theta):
        ax.axvspan(xmin, theta, color='red', alpha=0.05, zorder=0)
        ax.axvspan(theta, xmax, color='green', alpha=0.05, zorder=0)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_spectral_scaling(df, save_path, eta=0.1, baseline_delta=0.4):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ref = df[(df['n_train'] == 2000) & (df['nu'] == 100)
             & np.isclose(df['eta'], eta)]
    if len(ref) == 0:
        plt.close(fig)
        return

    base_rows = ref[np.isclose(ref['delta'], baseline_delta)]
    if len(base_rows) == 0:
        plt.close(fig)
        return
    D0 = float(np.mean(base_rows['sin_bottom'].values))

    sub = ref[ref['delta'] < 0.4].copy().sort_values('delta')
    x = 1.0 / sub['M_hat'].values
    y = sub['sin_bottom'].values - D0
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) == 0:
        plt.close(fig)
        return

    # 
    ax.scatter(x, y, s=70, color=COLORS['nu100'],
               edgecolor='black', linewidth=0.8, zorder=5,
               label=f'Data ($n=2000$, $\\nu=100$, $\\eta={eta}$)')

    if len(x) > 1:
        x_ref = np.array([x.min() * 0.8, x.max() * 1.2])
        logx = np.log(x)
        logy = np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        y_ref = np.exp(intercept) * x_ref ** slope
        # 
        ax.plot(x_ref, y_ref, 'k--', linewidth=1.8,
                label=f'Fit slope = {slope:.3f}')
        # 
        xm = np.exp(np.median(logx))
        ym = np.exp(np.median(logy))
        ax.plot(x_ref, ym * (x_ref / xm), ':', color='gray', linewidth=1.5,
                label='Slope 1 reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Inverse spectral margin $1 / \\hat{M}(\\Sigma) = 1/\\Delta$')
    ax.set_ylabel('Baseline-subtracted $\\sin\\Theta$ (bottom)')
    ax.set_title('Spectral scaling below the boundary')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(True, which='both', alpha=0.2, linestyle=':')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_bottom_diagnostic(df, save_path, eta_ref=0.1):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    nu_colors = {100: COLORS['nu100'], 30: COLORS['nu30'],
                 10: COLORS['nu10'], 3: COLORS['nu3']}

    sub_eta = df[np.isclose(df['eta'], eta_ref)]
    for nu in [100, 30, 10, 3]:
        sub = sub_eta[sub_eta['nu'] == nu].groupby('delta').agg(
            Db_mean=('sin_bottom', 'mean'),
            Db_std=('sin_bottom', 'std'),
        ).reset_index().sort_values('delta')
        if len(sub) == 0:
            continue
        ax.errorbar(
            sub['delta'], sub['Db_mean'],
            yerr=sub['Db_std'],
            marker='s', markersize=5, linewidth=1.6,
            color=nu_colors[nu], label=f'$\\nu$ = {nu}',
            capsize=3, capthick=1.0,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Spectral margin $\\hat{M}(\\Sigma) = \\Delta$')
    ax.set_ylabel('$\\sin\\Theta$ of bottom eigenvector')
    ax.set_title(f'Diagnostic: bottom eigenvector rotation ($\\eta$ = {eta_ref})')
    ax.legend(frameon=False, ncol=2, loc='best')
    ax.grid(True, which='both', alpha=0.2, linestyle=':')

    # 
    x_vals = sorted(df['delta'].unique())
    if len(x_vals) >= 2:
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f'{v:g}' for v in x_vals])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    y_all = df['sin_bottom'].values
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if len(y_all) > 0:
        ymin, ymax = float(y_all.min()), float(y_all.max())
        if ymax / ymin < 10:
            ymin = ymin / 3.0
            ymax = ymax * 3.0
        e_lo = int(np.floor(np.log10(ymin)))
        e_hi = int(np.ceil(np.log10(ymax)))
        if e_hi <= e_lo:
            e_hi = e_lo + 1
        y_ticks = [10.0 ** e for e in range(e_lo, e_hi + 1)]
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylim(10.0 ** (e_lo - 0.3), 10.0 ** (e_hi + 0.3))

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_eta_scaling(df, save_path, baseline_delta=0.4):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ref = df[(df['n_train'] == 2000) & (df['nu'] == 100)]
    if len(ref) == 0:
        plt.close(fig)
        return

    deltas_small = sorted([d for d in ref['delta'].unique() if d < 0.4])
    if len(deltas_small) == 0:
        plt.close(fig)
        return

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
    for i, delta in enumerate(deltas_small):
        sub = ref[np.isclose(ref['delta'], delta)].sort_values('eta')
        if len(sub) < 2:
            continue
        D0_by_eta = (ref[np.isclose(ref['delta'], baseline_delta)]
                     .groupby('eta')['sin_bottom'].mean())
        y = []
        x = []
        for _, row in sub.iterrows():
            eta = row['eta']
            if eta not in D0_by_eta.index:
                continue
            d0 = D0_by_eta.loc[eta]
            yv = row['sin_bottom'] - d0
            if yv > 0:
                x.append(eta)
                y.append(yv)
        if len(x) < 2:
            continue
        x = np.array(x)
        y = np.array(y)
        # 
        ax.scatter(x, y, s=60, marker=markers[i % len(markers)],
                   edgecolor='black', linewidth=0.8, zorder=5,
                   label=f'$\\Delta$ = {delta:g}')
        # 
        logx, logy = np.log(x), np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        x_ref = np.array([x.min() * 0.8, x.max() * 1.2])
        y_ref = np.exp(intercept) * x_ref ** slope
        ax.plot(x_ref, y_ref, '--', linewidth=1.2,
                label=f'  slope = {slope:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Perturbation strength $\\eta$')
    ax.set_ylabel('Baseline-subtracted $\\sin\\Theta$ (bottom)')
    ax.set_title('Eta-scaling of the spectral term')
    # 
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='lower right')
    ax.grid(True, which='both', alpha=0.2, linestyle=':')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# 7. Main
# =============================================================================
def main():
    np.random.seed(42)

    etas = [0.05, 0.1, 0.2, 0.4, 0.8]
    deltas = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.8]
    nus = [100, 30, 10, 3]
    n_trains = [500, 1000, 2000]
    p = 100
    k = 10
    delta_op = 5e-3
    n_repeats = 5

    records = []
    total_start = time.time()

    for eta in etas:
        for n_train in n_trains:
            for delta in deltas:
                for nu in nus:
                    for rep in range(n_repeats):
                        seed = (1000000 * etas.index(eta)
                                + 100000 * n_trains.index(n_train)
                                + 10000 * deltas.index(delta)
                                + 100 * nus.index(nu)
                                + rep)
                        print(f"[eta={eta}, n={n_train}, "
                              f"delta={delta:.3f}, nu={nu}, rep={rep}]",
                              end=' ')
                        try:
                            result = run_single_condition(
                                delta=delta, nu=nu, n_train=n_train,
                                eta=eta, p=p, k=k, seed=seed
                            )
                            result['repeat'] = rep
                            records.append(result)
                            print(f"sin_bot={result['sin_bottom']:.3e}, "
                                  f"D_bot={result['D_bottom']:.3e}, "
                                  f"D_sig={result['D_signal']:.3f}")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            traceback.print_exc()

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No results collected.")
        return

    df.to_csv(os.path.join(RESULTS_DIR, 'raw_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    for eta in etas:
        sub_eta = df[np.isclose(df['eta'], eta)]
        print(f"\n--- eta = {eta} ---")
        for delta in deltas:
            sub = sub_eta[np.isclose(sub_eta['delta'], delta)]
            if len(sub) == 0:
                continue
            print(f"  Delta = {delta:.3f}: "
                  f"median sin_bottom = {sub['sin_bottom'].median():.3e}, "
                  f"median D_bottom = {sub['D_bottom'].median():.3e}, "
                  f"median D_signal = {sub['D_signal'].median():.4f}")

    # ---- Calibration ----
    C1_list, C2_list, C3_list = [], [], []
    eta_ref = 0.1
    for cal_rep in range(5):
        df_boot = df.sample(frac=1.0, replace=True, random_state=cal_rep)
        C1, C2 = calibrate_C1_C2(df_boot)
        C3 = calibrate_C3(df_boot, eta=eta_ref, metric='sin_bottom')
        C1_list.append(C1)
        C2_list.append(C2)
        if np.isfinite(C3):
            C3_list.append(C3)

    C1_med = float(np.median(C1_list))
    C2_med = float(np.median(C2_list))
    C1_ci = (float(np.percentile(C1_list, 2.5)),
             float(np.percentile(C1_list, 97.5)))
    C2_ci = (float(np.percentile(C2_list, 2.5)),
             float(np.percentile(C2_list, 97.5)))

    if len(C3_list) == 0:
        C3_med = np.nan
        C3_ci = (np.nan, np.nan)
        print("WARNING: all C3 bootstrap fits failed.")
    else:
        C3_arr = np.array(C3_list)
        C3_med = float(np.median(C3_arr))
        C3_ci = (float(np.percentile(C3_arr, 2.5)),
                 float(np.percentile(C3_arr, 97.5)))

    if np.isfinite(C3_med):
        theta = 3.0 * C3_med * eta_ref / delta_op
    else:
        theta = np.nan

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"C1 = {C1_med:.6f}  (95% CI: [{C1_ci[0]:.6f}, {C1_ci[1]:.6f}])")
    print(f"C2 = {C2_med:.6f}  (95% CI: [{C2_ci[0]:.6f}, {C2_ci[1]:.6f}])")
    print(f"C3 = {C3_med:.6f}  (95% CI: [{C3_ci[0]:.6f}, {C3_ci[1]:.6f}])")
    print(f"delta_op = {delta_op}")
    print(f"theta = 3 C3 eta / delta_op = {theta:.6f}")

    calib = {
        'C1_median': C1_med, 'C1_CI': C1_ci,
        'C2_median': C2_med, 'C2_CI': C2_ci,
        'C3_median': C3_med, 'C3_CI': C3_ci,
        'theta': theta, 'eta_ref': eta_ref, 'delta_op': delta_op,
        'metric': 'sin_bottom',
    }
    with open(os.path.join(RESULTS_DIR, 'calibration_constants.json'), 'w') as f:
        json.dump(calib, f, indent=2, default=lambda o: None)

    summary = df.groupby(['eta', 'delta', 'nu', 'n_train']).agg(
        M_hat_mean=('M_hat', 'mean'),
        D_total_mean=('D_total', 'mean'),
        D_signal_mean=('D_signal', 'mean'),
        D_signal_std=('D_signal', 'std'),
        D_bottom_mean=('D_bottom', 'mean'),
        D_bottom_std=('D_bottom', 'std'),
        sin_bottom_mean=('sin_bottom', 'mean'),
        sin_bottom_std=('sin_bottom', 'std'),
        pert_norm_mean=('pert_norm', 'mean'),
        epsilon_mean=('epsilon', 'mean'),
        stat_term_mean=('stat_term', 'mean'),
        moment_term_mean=('moment_term', 'mean'),
        spectral_term_mean=('spectral_term', 'mean'),
    ).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)
    print(f"Saved summary: {os.path.join(RESULTS_DIR, 'summary.csv')}")

    # ---- Plots ----
    plot_degradation_curve(
        df, theta, delta_op,
        save_path=os.path.join(RESULTS_DIR, 'Fig1_degradation_curve.tiff'),
        eta_ref=eta_ref,
    )
    plot_spectral_scaling(
        df, eta=eta_ref,
        save_path=os.path.join(RESULTS_DIR, 'Fig2_spectral_scaling.tiff'),
    )
    plot_bottom_diagnostic(
        df, eta_ref=eta_ref,
        save_path=os.path.join(RESULTS_DIR, 'Fig4_bottom_diagnostic.tiff'),
    )
    plot_eta_scaling(
        df,
        save_path=os.path.join(RESULTS_DIR, 'Fig3_eta_scaling.tiff'),
    )

    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print("All experiments completed.")


if __name__ == '__main__':
    main()