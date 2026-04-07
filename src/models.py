from __future__ import annotations

import numpy as np
import pandas as pd

# Mass-to-light ratios (Solar units at 3.6 microns)
UPS_DISK = 0.5
UPS_BULGE = 0.7

A0_MOND = 1.2e-10  # m/s^2

_KPC_TO_M = 3.0856775814913673e19
_KM_TO_M = 1_000.0


def _get_radius_kpc(df: pd.DataFrame) -> np.ndarray:
    for col in ("R", "r", "Rad", "RAD", "radius", "Radius"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    raise KeyError(f"No radius column found. Available columns: {list(df.columns)}")

def _get_vbar_kms(df: pd.DataFrame) -> np.ndarray:
    """
    SPARC Rotmod_LTG typically provides Vgas, Vdisk, Vbul (km/s).
    Some files may omit Vbul.
    """
    for req in ("Vgas", "Vdisk"):
        if req not in df.columns:
            raise KeyError(f"Missing required baryon component column: {req}")

    vgas = df["Vgas"].to_numpy(dtype=float)
    vdisk = df["Vdisk"].to_numpy(dtype=float)
    vbul = df["Vbul"].to_numpy(dtype=float) if "Vbul" in df.columns else np.zeros_like(vgas)

    # Guard against tiny negatives from parsing/rounding
    v2 = np.maximum(vgas, 0.0) ** 2 + np.maximum(vdisk, 0.0) ** 2 + np.maximum(vbul, 0.0) ** 2
    return np.sqrt(v2)

def _second_derivative_uniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Deterministic second derivative on approximately uniform grid.
    Uses simple centered differences with edge replication.
    """
    n = int(x.size)
    out = np.zeros(n, dtype=float)
    if n < 3:
        return out
    # use median spacing to avoid per-galaxy tuning
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        return out
    out[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dx * dx)
    out[0] = out[1]
    out[-1] = out[-2]
    return out

def _sigma3_local_length_warp(
    r_kpc_sorted: np.ndarray,
    gbar_si_sorted: np.ndarray,
    Lc_kpc: float,
) -> np.ndarray:
    """
    Σ₃ admissible fluctuation (Option A contract):

    Inputs MUST already be:
      - finite
      - sorted by radius ascending
      - same shape

    Returns Leff_kpc aligned with that same sorted order.

      L_eff(r) = Lc * clip(1 + ETA * tanh(C_hat(r)), 1-ETA, 1+ETA)

    where C_hat is robustly normalized curvature of log(gbar) along radius.
    """
    # Fixed constants (NOT exposed as parameters)
    ETA = 0.50
    CMAX = 5.0
    EPS = 1e-30

    r = r_kpc_sorted.astype(float)
    gb = np.maximum(gbar_si_sorted.astype(float), EPS)

    n = int(r.size)
    if n < 5:
        return float(Lc_kpc) * np.ones(n, dtype=float)

    # log(gbar) and a tiny deterministic 3-pt smoothing (edge-padded)
    x = np.log(gb)
    xpad = np.concatenate(([x[0]], x, [x[-1]]))
    xs = (xpad[:-2] + xpad[1:-1] + xpad[2:]) / 3.0

    # Irregular-grid second derivative: three-point stencil
    C = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        h1 = float(r[i] - r[i - 1])
        h2 = float(r[i + 1] - r[i])
        if h1 <= 0.0 or h2 <= 0.0:
            C[i] = 0.0
            continue
        C[i] = 2.0 * (((xs[i + 1] - xs[i]) / h2) - ((xs[i] - xs[i - 1]) / h1)) / (h1 + h2)

    # Edge handling: copy nearest interior
    C[0] = C[1]
    C[-1] = C[-2]

    # Hard bound + robust normalization
    C = np.clip(C, -CMAX, CMAX)
    denom = float(np.median(np.abs(C))) + EPS
    Chat = C / denom

    scale = 1.0 + ETA * np.tanh(Chat)
    scale = np.clip(scale, 1.0 - ETA, 1.0 + ETA)

    return float(Lc_kpc) * scale


def _smooth_compact_triangular(r: np.ndarray, y: np.ndarray, Lc: float) -> np.ndarray:
    """
    Deterministic compact-support smoothing with a triangular (tent) kernel.

    For each i:
        w_ij = max(0, 1 - |r_i - r_j| / Lc)
        y_tilde[i] = sum_j w_ij y[j] / sum_j w_ij

    Properties:
    - Support radius = Lc (finite-range locality)
    - No extra parameters beyond Lc
    - Deterministic, order-invariant (uses pairwise distances)
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(r.size)
    if n == 0:
        return y.copy()
    if not np.isfinite(Lc) or Lc <= 0.0:
        return y.copy()

    # Pairwise distances (n x n)
    dr = np.abs(r[:, None] - r[None, :])
    w = 1.0 - (dr / float(Lc))
    w[w < 0.0] = 0.0

    denom = np.sum(w, axis=1)
    # Guard: if a row has zero weight (shouldn't happen unless all NaN/inf), fall back to identity
    out = np.empty_like(y, dtype=float)
    for i in range(n):
        if denom[i] <= 0.0 or not np.isfinite(denom[i]):
            out[i] = y[i]
        else:
            out[i] = float(np.sum(w[i, :] * y) / denom[i])
    return out

def dft_b_model_sigma3(df: pd.DataFrame, ac: float, Lc: float) -> np.ndarray:
    """
    Σ₃ variant of DFT-B (Option A):

    - Deterministic, bounded, baryon-derived local warp of kernel length.
    - SAME global parameters as DFT-B: (ac, Lc). No extra fit knobs.
    - Computes on a sorted/finite radius subset, then maps back to df row order.

    Output: predicted circular velocity v_dft (km/s), aligned with df rows.
    """
    r_kpc = _get_radius_kpc(df).astype(float)
    vbar_kms = _get_vbar_kms(df).astype(float)

    r_m = r_kpc * _KPC_TO_M
    vbar_ms = vbar_kms * _KM_TO_M

    # baryonic acceleration in SI: gbar = vbar^2 / r
    eps_r = 1e-30
    gbar = (vbar_ms * vbar_ms) / np.maximum(r_m, eps_r)

    # Mask finite points once, sort once
    m = np.isfinite(r_kpc) & np.isfinite(gbar)
    n_all = int(r_kpc.size)
    out_v = np.full(n_all, np.nan, dtype=float)

    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return out_v

    r0 = r_kpc[idx]
    g0 = gbar[idx]

    order = np.argsort(r0)
    idxs = idx[order]           # indices into original df order
    r = r0[order]               # sorted radii (kpc)
    g = g0[order]               # sorted gbar (SI)

    # Σ₃ local length (aligned with sorted order)
    Leff_kpc = _sigma3_local_length_warp(r, g, float(Lc))

    # Nonlocal smoothing using Li = Leff_kpc[i] per target i
    n = int(r.size)
    gtilde = np.zeros(n, dtype=float)
    for i in range(n):
        Li = float(max(Leff_kpc[i], 1e-6))
        w = np.exp(-np.abs(r - r[i]) / Li)
        sw = float(np.sum(w))
        if sw <= 0.0 or not np.isfinite(sw):
            gtilde[i] = g[i]
        else:
            gtilde[i] = float(np.sum(w * g) / sw)

    # DFT-B law: g_total = gbar + sqrt(ac * gtilde)
    acf = float(ac)
    gtot = g + np.sqrt(np.maximum(acf * np.maximum(gtilde, 0.0), 0.0))

    # v = sqrt(g_total * r)
    r_m_sorted = r * _KPC_TO_M
    v_ms = np.sqrt(np.maximum(gtot * np.maximum(r_m_sorted, eps_r), 0.0))
    v_kms_sorted = (v_ms / _KM_TO_M).astype(float)

    # Map back to original df order
    out_v[idxs] = v_kms_sorted
    return out_v

def get_g_bar(df: pd.DataFrame) -> np.ndarray:
    """Newtonian baryonic acceleration (m/s^2) from SPARC components."""
    km2m = 1e3
    kpc2m = 3.086e19
    v_bar_sq = (UPS_DISK * df["Vdisk"] ** 2 + UPS_BULGE * df["Vbul"] ** 2 + df["Vgas"] ** 2) * (km2m**2)
    r_m = df["Rad"].to_numpy(dtype=float) * kpc2m
    return v_bar_sq.to_numpy(dtype=float) / np.maximum(r_m, 1e-10)


def rar_exp_benchmark(df: pd.DataFrame) -> np.ndarray:
    """Exponential-style RAR/MOND benchmark mapping."""
    g_bar = get_g_bar(df)
    g_tot = g_bar / (1.0 - np.exp(-np.sqrt(g_bar / A0_MOND)))
    r_m = df["Rad"].to_numpy(dtype=float) * 3.086e19
    v_kms = np.sqrt(np.maximum(0.0, g_tot * r_m)) / 1000.0
    return v_kms


def dft_b_model(df: pd.DataFrame, a_c: float, L_c_kpc: float) -> np.ndarray:
    """
    DFT-B (compact-support variant):
      g_pred = g_bar + sqrt(a_c * g_tilde)
      g_tilde(r) = ∫ g_bar(r') K(r,r') dr' / ∫ K(r,r') dr'
      K = max(0, 1 - |r-r'| / L_c)   (triangular / tent kernel; finite support)

    Same parameter count as before: (a_c, L_c_kpc).
    """
    if L_c_kpc <= 0:
        raise ValueError("L_c_kpc must be > 0")

    r_kpc = df["Rad"].to_numpy(dtype=float)
    g_bar = get_g_bar(df)
    n = int(r_kpc.size)

    if n == 0:
        return np.zeros(0, dtype=float)

    # Pairwise distances
    dist = np.abs(r_kpc[:, None] - r_kpc[None, :])

    # Compact-support triangular kernel
    K = 1.0 - (dist / float(L_c_kpc))
    K[K < 0.0] = 0.0

    # 1D trapezoid weights over r_kpc
    if n == 1:
        w = np.array([1.0], dtype=float)
    else:
        dx = np.diff(r_kpc)
        w = np.empty_like(r_kpc, dtype=float)
        w[0] = float(dx[0]) / 2.0
        w[-1] = float(dx[-1]) / 2.0
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0

    num = K @ (g_bar * w)
    den = K @ w
    g_tilde = num / np.maximum(den, 1e-300)

    g_pred = g_bar + np.sqrt(np.maximum(0.0, float(a_c) * np.maximum(g_tilde, 0.0)))

    r_m = r_kpc * 3.086e19
    v_kms = np.sqrt(np.maximum(0.0, g_pred * r_m)) / 1000.0

    if not np.all(np.isfinite(v_kms)):
        raise ValueError("Non-finite v_kms encountered in dft_b_model().")

    return v_kms.astype(float)