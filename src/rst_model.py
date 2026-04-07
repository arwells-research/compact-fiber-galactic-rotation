"""
RST-Informed Galactic Rotation Model
=====================================

Derives an effective rotation-curve law from the Reciprocal System of Theory
compact fiber structure (Papers I-III, Wells 2026).

Key RST inputs used:
  1. Gravitation is 3D scalar motion inward; the space-time progression is
     universal scalar motion outward at unit speed. The net motion at any
     point is the algebraic resultant.

  2. The gravitational limit (Larson point 37): at distance r_GL from a
     mass concentration, inward gravitational speed equals the outward
     progression speed. Beyond r_GL, the net scalar motion is outward.
     This is the RST mechanism that replaces dark matter: matter beyond
     the gravitational limit is being carried outward by the progression,
     but the observable is the *rotation speed*, which includes both
     gravitational and progression components as seen from the spatial
     reference frame.

  3. Coordinate time accumulation: gravitation produces coordinate time
     at rate 3v²/c² per unit clock time (Paper III §10). This modifies
     the effective dynamics in the same way that GR's metric curvature
     does, but with a different mechanism.

  4. The covering degree d=2 from the compact fiber Z/4Z × Z/4Z × Z/8Z
     (Paper I) enters through the inter-regional ratio and the
     second-power relation r = ρ².

  5. The inter-regional ratio R = 156.444 connects natural units to
     conventional units and governs the transition between gravitational
     regimes.

Model: RST Gravitational-Limit Rotation Law
--------------------------------------------
At each radius r from a galaxy center:

  g_Newton(r) = v_bar²(r) / r       (baryonic Newtonian acceleration)

The RST predicts that gravitation is always accompanied by the scalar
progression. In the region r < r_GL, gravitation dominates and the
net effect is inward (standard Newtonian regime). In the region r > r_GL,
the progression dominates and the net outward motion maintains the
orbital speed rather than letting it fall Keplerian.

The effective acceleration including the progression effect:

  g_eff(r) = g_bar(r) + g_progression(r)

where g_progression represents the coordinate-time contribution that
standard physics attributes to dark matter. From the RST coordinate-time
mechanism (Paper III):

  g_progression(r) = (3/d) * g_bar(r) * v²(r)/c²  * f(r/r_GL)

But at galactic scales, v << c, so the direct coordinate-time term is
negligible. The dominant RST effect is the gravitational limit mechanism:

  At r ~ r_GL:  g_grav(r) ≈ g_progression(r)
  => v_rot stays roughly constant (flat rotation curve)

Implementation:
  We model the RST gravitational-limit effect as an effective additional
  acceleration that becomes significant where baryonic acceleration falls
  below a characteristic scale. The RST predicts this scale is set by the
  natural unit system and the inter-regional ratio, not by a free parameter.

  The functional form derives from the balance condition:
    g_grav = g_prog  =>  GM/r² = a_nat * f(r/r_GL)

  where a_nat is the natural acceleration unit.

  The effective total acceleration:
    g_tot = g_bar + sqrt(a_0 * g_bar) * h(r, L_GL)

  where:
    a_0 is derived from the natural units (not free in principle, but we
        allow calibration to account for incomplete derivation of G from
        natural units — Paper III §10.4 identifies this as open)
    h(r, L_GL) encodes the gravitational-limit transition, which is a
        nonlocal function of the mass distribution (the gravitational
        limit depends on the total enclosed mass profile)

RST-specific improvements over vanilla DFT-B:
  1. The nonlocal kernel is derived from the second-power relation:
     the time-region smoothing maps to r = ρ², producing a kernel that
     goes as exp(-sqrt(|r-r'|/L)) rather than exp(-|r-r'|/L).
     This "square-root exponential" kernel has heavier tails and better
     matches the long-range gravitational-limit transition.

  2. The coupling law uses the covering-degree structure: the geometric
     mean sqrt(a_0 * g_bar) arises from the degree-2 covering map
     projection, connecting the time-region (where g is harmonic) to
     extension space (where g is Coulombic) through r = ρ².

  3. A curvature-dependent length scale from the fiber transport metric:
     the per-step weight 1/N on cyclic sectors (Paper I, Theorem 11.1)
     implies that the effective coherence length scales with the local
     "sector capacity" of the mass distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_KPC_TO_M = 3.0856775814913673e19
_KM_TO_M = 1_000.0
_C_SI = 2.998e8  # m/s

# RST natural units (Larson)
S_NAT = 4.559e-8   # m
T_NAT = 1.521e-16  # s
# Natural speed = s_nat / t_nat = c (by construction)
# Inter-regional ratio
R_INTER = 156.444

# Covering degree from compact fiber
D_COVER = 2


def _get_radius_kpc(df: pd.DataFrame) -> np.ndarray:
    for col in ("R", "r", "Rad", "RAD", "radius", "Radius"):
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    raise KeyError(f"No radius column found. Available columns: {list(df.columns)}")


def _get_vbar_kms(df: pd.DataFrame) -> np.ndarray:
    UPS_DISK = 0.5
    UPS_BULGE = 0.7
    vgas = df["Vgas"].to_numpy(dtype=float)
    vdisk = df["Vdisk"].to_numpy(dtype=float)
    vbul = df["Vbul"].to_numpy(dtype=float) if "Vbul" in df.columns else np.zeros_like(vgas)

    v2 = (np.maximum(vgas, 0.0) ** 2
          + UPS_DISK * np.maximum(vdisk, 0.0) ** 2
          + UPS_BULGE * np.maximum(vbul, 0.0) ** 2)
    return np.sqrt(v2)


def rst_rotation_model(df: pd.DataFrame, a0: float, L_gl_kpc: float) -> np.ndarray:
    """
    RST gravitational-limit rotation model.

    Parameters
    ----------
    df : DataFrame with SPARC columns (Rad, Vgas, Vdisk, Vbul, Vobs, errV)
    a0 : acceleration scale (m/s²) — in principle derivable from natural units
    L_gl_kpc : gravitational-limit coherence length (kpc) — the characteristic
               scale over which the progression-gravitation balance is evaluated

    Returns
    -------
    v_pred : predicted rotation velocity (km/s)
    """
    r_kpc = _get_radius_kpc(df).astype(float)
    vbar_kms = _get_vbar_kms(df).astype(float)

    r_m = r_kpc * _KPC_TO_M
    vbar_ms = vbar_kms * _KM_TO_M
    eps_r = 1e-30

    # Baryonic acceleration
    gbar = (vbar_ms ** 2) / np.maximum(r_m, eps_r)

    # Mask and sort
    m = np.isfinite(r_kpc) & np.isfinite(gbar) & (gbar > 0)
    n_all = int(r_kpc.size)
    out_v = np.full(n_all, np.nan, dtype=float)

    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return out_v

    r0 = r_kpc[idx]
    g0 = gbar[idx]

    order = np.argsort(r0)
    idxs = idx[order]
    r = r0[order]      # sorted radii (kpc)
    g = g0[order]       # sorted gbar (SI)
    n = int(r.size)

    # === RST KERNEL: Second-power relation implies sqrt-exponential ===
    # The time-region coordinate ρ maps to extension space via r = ρ².
    # A smoothing kernel exp(-|ρ-ρ'|/L_ρ) in the time region maps to
    # exp(-|sqrt(r) - sqrt(r')|/L_ρ) in extension space.
    # This gives heavier tails than a simple exponential in r, which is
    # physically correct: the gravitational limit transition is gradual.

    # Adaptive length scale from curvature of log(gbar)
    # (fiber transport metric: 1/N per step, here N = local "sector capacity")
    log_g = np.log(np.maximum(g, 1e-30))

    # Curvature of log(gbar) profile
    curv = np.zeros(n, dtype=float)
    if n >= 3:
        for i in range(1, n - 1):
            h1 = max(r[i] - r[i-1], 1e-6)
            h2 = max(r[i+1] - r[i], 1e-6)
            curv[i] = 2.0 * ((log_g[i+1] - log_g[i])/h2 - (log_g[i] - log_g[i-1])/h1) / (h1 + h2)
        curv[0] = curv[1]
        curv[-1] = curv[-2]

    # Curvature-modulated length: where the baryonic profile curves sharply
    # (transition regions), the coherence length shortens; where it's smooth
    # (outer disk), it lengthens. This is the fiber transport metric applied
    # at galactic scale.
    ETA = 0.45  # modulation depth (fixed, not fitted)
    curv_clip = np.clip(curv, -5.0, 5.0)
    curv_norm = curv_clip / (np.median(np.abs(curv_clip)) + 1e-30)
    L_eff = float(L_gl_kpc) * np.clip(1.0 + ETA * np.tanh(curv_norm), 1.0 - ETA, 1.0 + ETA)

    # Compute nonlocal smoothed baryonic field using sqrt-exponential kernel
    # (RST second-power relation)
    sqrt_r = np.sqrt(np.maximum(r, 0.0))
    gtilde = np.zeros(n, dtype=float)

    for i in range(n):
        Li = max(float(L_eff[i]), 1e-6)
        # Second-power kernel: distance in ρ-space = |sqrt(r) - sqrt(r')|
        dr_rho = np.abs(sqrt_r - sqrt_r[i])
        w = np.exp(-dr_rho / Li)
        sw = np.sum(w)
        if sw > 0 and np.isfinite(sw):
            gtilde[i] = np.sum(w * g) / sw
        else:
            gtilde[i] = g[i]

    # === RST ACCELERATION LAW ===
    # The covering degree d=2 enters through the geometric mean:
    # g_eff = g_bar + (a0 * gtilde)^(1/d) * g_bar^(1 - 1/d)
    #       = g_bar + sqrt(a0 * gtilde)  [for d=2]
    #
    # This is the same functional form as DFT-B, but now structurally
    # motivated: the sqrt arises from the degree-2 covering projection,
    # and the nonlocal kernel shape comes from the second-power relation.

    a0f = float(a0)
    g_extra = np.sqrt(np.maximum(a0f * np.maximum(gtilde, 0.0), 0.0))
    g_tot = g + g_extra

    # v = sqrt(g_tot * r)
    r_m_sorted = r * _KPC_TO_M
    v_ms = np.sqrt(np.maximum(g_tot * np.maximum(r_m_sorted, eps_r), 0.0))
    v_kms = v_ms / _KM_TO_M

    out_v[idxs] = v_kms
    return out_v


def rst_rotation_model_v2(df: pd.DataFrame, a0: float, L_gl_kpc: float,
                           alpha_ct: float = 0.0) -> np.ndarray:
    """
    RST rotation model v2: adds coordinate-time correction.

    The coordinate-time accumulation rate 3v²/c² (Paper III) produces a
    small additional acceleration in the inner regions where v is highest.
    This is the RST equivalent of the relativistic correction and affects
    the shape of the inner rotation curve.

    Parameters
    ----------
    a0 : acceleration scale
    L_gl_kpc : gravitational-limit coherence length
    alpha_ct : coordinate-time coupling (dimensionless). In principle = 3,
               but the effective value depends on the projection geometry.
               Set to 0 to recover the base model.
    """
    r_kpc = _get_radius_kpc(df).astype(float)
    vbar_kms = _get_vbar_kms(df).astype(float)

    r_m = r_kpc * _KPC_TO_M
    vbar_ms = vbar_kms * _KM_TO_M
    eps_r = 1e-30

    gbar = (vbar_ms ** 2) / np.maximum(r_m, eps_r)

    m = np.isfinite(r_kpc) & np.isfinite(gbar) & (gbar > 0)
    n_all = int(r_kpc.size)
    out_v = np.full(n_all, np.nan, dtype=float)

    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return out_v

    r0 = r_kpc[idx]
    g0 = gbar[idx]
    vb0 = vbar_ms[idx[np.argsort(r0)]]

    order = np.argsort(r0)
    idxs = idx[order]
    r = r0[order]
    g = g0[order]
    vb = vbar_ms[idx][order]
    n = int(r.size)

    # Curvature-modulated length
    log_g = np.log(np.maximum(g, 1e-30))
    curv = np.zeros(n, dtype=float)
    if n >= 3:
        for i in range(1, n - 1):
            h1 = max(r[i] - r[i-1], 1e-6)
            h2 = max(r[i+1] - r[i], 1e-6)
            curv[i] = 2.0 * ((log_g[i+1] - log_g[i])/h2 - (log_g[i] - log_g[i-1])/h1) / (h1 + h2)
        curv[0] = curv[1]
        curv[-1] = curv[-2]

    ETA = 0.45
    curv_clip = np.clip(curv, -5.0, 5.0)
    curv_norm = curv_clip / (np.median(np.abs(curv_clip)) + 1e-30)
    L_eff = float(L_gl_kpc) * np.clip(1.0 + ETA * np.tanh(curv_norm), 1.0 - ETA, 1.0 + ETA)

    # Sqrt-exponential kernel (second-power relation)
    sqrt_r = np.sqrt(np.maximum(r, 0.0))
    gtilde = np.zeros(n, dtype=float)
    for i in range(n):
        Li = max(float(L_eff[i]), 1e-6)
        dr_rho = np.abs(sqrt_r - sqrt_r[i])
        w = np.exp(-dr_rho / Li)
        sw = np.sum(w)
        if sw > 0 and np.isfinite(sw):
            gtilde[i] = np.sum(w * g) / sw
        else:
            gtilde[i] = g[i]

    # Base RST acceleration (covering degree 2)
    a0f = float(a0)
    g_extra = np.sqrt(np.maximum(a0f * np.maximum(gtilde, 0.0), 0.0))

    # Coordinate-time correction (Paper III, Eq. for dt_c/dt = 3v²/c²)
    # This adds a small inward acceleration in the inner region
    if alpha_ct > 0:
        v_est = np.sqrt(np.maximum((g + g_extra) * r * _KPC_TO_M, 0.0))
        ct_factor = 1.0 + alpha_ct * (v_est / _C_SI) ** 2
        g_tot = (g + g_extra) * ct_factor
    else:
        g_tot = g + g_extra

    r_m_sorted = r * _KPC_TO_M
    v_ms = np.sqrt(np.maximum(g_tot * np.maximum(r_m_sorted, eps_r), 0.0))
    v_kms = v_ms / _KM_TO_M

    out_v[idxs] = v_kms
    return out_v
