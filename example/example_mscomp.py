#!/usr/bin/env python3

"""
Generate "paper-style" MS-Comp probability figures using a **synthetic** series
and the local `mscomp` package (univariate Markov-switching mean–variance component).

Outputs (saved in --outdir):
- mscomp_fit_top_synth.png       : Top panel with P[Mean Regime 1] and P[Mean Regime 3]
- mscomp_fit_bottom_synth.png    : Bottom panel with P[Mean Regime 1]
- mscomp_fit_synth_banner.png    : Composed banner (two panels stacked) — ideal for GitHub social preview

Usage (from repo root):
    python examples/generate_mscomp_synthetic_banner.py --outdir figures --seed 123

Notes:
- Uses matplotlib only, one figure per chart (no subplots), and no explicit colors/styles.
- Shaded vertical bands mark periods where the estimated P[variance=high] > 0.5.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure local package import works when running from repo root
# (i.e., when the folder "mscomp/" is in the same directory as this script's parent)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from mscomp import MSComp  # local package
except Exception as e:
    raise RuntimeError("Could not import 'mscomp'. Make sure the repo root is on PYTHONPATH.") from e


def generate_series(T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # True regimes
    mean_regime_true = np.repeat([0, 1, 2, 3, 1, 0], [40, 50, 60, 50, 50, 50])
    if len(mean_regime_true) < T:
        mean_regime_true = np.pad(mean_regime_true, (0, T - len(mean_regime_true)), mode="edge")
    mean_regime_true = mean_regime_true[:T]

    var_regime_true = np.zeros(T, dtype=int)
    i = 0
    # alternating stretches of high (1) and low (0) variance
    while i < T:
        on = rng.integers(12, 22)   # high-variance stretch
        off = rng.integers(20, 40)  # low-variance stretch
        var_regime_true[i:i+on] = 1
        i += on + off
    var_regime_true = var_regime_true[:T]

    # Parameters
    mu_vals = np.array([-0.7, 0.1, 0.8, -0.05])   # mean per mean-regime
    sigma_vals = np.array([0.4, 1.2])             # std per var-regime (low/high)
    phi = np.array([0.35, 0.15])                  # AR(2)

    y = np.zeros(T)
    for t in range(T):
        ar = (phi[0] * y[t-1] if t-1 >= 0 else 0.0) + (phi[1] * y[t-2] if t-2 >= 0 else 0.0)
        eps = rng.normal(0.0, sigma_vals[var_regime_true[t]])
        y[t] = mu_vals[mean_regime_true[t]] + ar + eps

    return y


def high_variance_bands(P_var_high: np.ndarray):
    """Return list of (start_index, end_index) where high variance prob > 0.5."""
    hv = (P_var_high > 0.5).astype(int)
    bands = []
    in_band = False
    start = 0
    T = len(hv)
    for i in range(T - 1):
        if not in_band and hv[i] == 1:
            in_band = True
            start = i
        if in_band and hv[i] == 1 and hv[i + 1] == 0:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, T - 1))
    return bands


def plot_top(years, P_mean, bands, out_path):
    fig = plt.figure(figsize=(16, 3.2), dpi=160)
    ax = plt.gca()
    for a, b in bands:
        ax.axvspan(years[a], years[b], alpha=0.2)
    ax.plot(years, P_mean[:, 0], linewidth=1.6, label="P[Mean Regime 1]")
    ax.plot(years, P_mean[:, 2], linewidth=1.6, linestyle=":", label="P[Mean Regime 3]")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(years[0], years[-1])
    ax.set_title("MSComp(4,2)-AR(2)")
    ax.set_ylabel("Probability")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bottom(years, P_mean, bands, out_path):
    fig = plt.figure(figsize=(16, 3.2), dpi=160)
    ax = plt.gca()
    for a, b in bands:
        ax.axvspan(years[a], years[b], alpha=0.2)
    ax.plot(years, P_mean[:, 0], linewidth=1.6, label="P[Mean Regime 1]")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(years[0], years[-1])
    ax.set_title("MSComp(4,2)-AR(2) — Smoothed P[Mean Regime 1]")
    ax.set_ylabel("Probability")
    ax.legend(frameon=False, ncol=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compose_banner(path_top, path_bottom, out_path):
    imgA = Image.open(path_top).convert("RGB")
    imgB = Image.open(path_bottom).convert("RGB")
    w = max(imgA.width, imgB.width)
    gutter = 12
    banner = Image.new("RGB", (w, imgA.height + imgB.height + gutter), (255, 255, 255))
    banner.paste(imgA, (0, 0))
    banner.paste(imgB, (0, imgA.height + gutter))
    banner.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MS-Comp probability figures.")
    parser.add_argument("--T", type=int, default=300, help="Length of the synthetic series")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Synthetic series
    y = generate_series(T=args.T, seed=args.seed)

    # 2) Fit MS-Comp(4,2)-AR(2)
    model = MSComp(k_mean=4, k_var=2, ar_order=2, absorbing_last_mean=False, triangular_Pm=False, random_state=42)
    res = model.fit(y, maxiter=200, tol=1e-6, method="L-BFGS-B", verbose=False)

    P_mean = res.mean_regime_probs           # (T, 4)
    P_var  = res.var_regime_probs            # (T, 2)
    years  = np.linspace(1950, 2012, len(y))

    # Bands via estimated high variance probability
    bands = high_variance_bands(P_var[:, 1])

    # 3) Plots (two separate figures)
    top_path = os.path.join(args.outdir, "mscomp_fit_top_synth.png")
    bottom_path = os.path.join(args.outdir, "mscomp_fit_bottom_synth.png")
    plot_top(years, P_mean, bands, top_path)
    plot_bottom(years, P_mean, bands, bottom_path)

    # 4) Compose the banner
    banner_path = os.path.join(args.outdir, "mscomp_fit_synth_banner.png")
    compose_banner(top_path, bottom_path, banner_path)

    print("Saved:")
    print("  ", top_path)
    print("  ", bottom_path)
    print("  ", banner_path)


if __name__ == "__main__":
    main()