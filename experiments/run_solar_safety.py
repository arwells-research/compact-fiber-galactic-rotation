"""
Solar System Safety Computation for the RST Nonlocal Term

This script reproduces the numerical results in Appendix E of:
"Galactic Rotation from the Compact Fiber: The Gravitational Limit,
R^{16}, and a Zero-Parameter Rotation-Curve Model"

It evaluates g_tilde at six Solar System positions using a Milky Way
rotation curve model, demonstrating that the nonlocal term is constant
across the Solar System to a relative precision of ~3e-8.

Usage:
    cd experiments && python run_solar_safety.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Physical constants
G = 6.674e-11          # m^3 kg^-1 s^-2
M_sun = 1.989e30       # kg
AU = 1.496e11          # m
KPC = 3.0856775814913673e19  # m

# RST derived parameters
R = 156.444
s_nat = 4.559e-8       # m
t_nat = 1.521e-16      # s
a_nat = s_nat / t_nat**2
N_m = 4
a0 = a_nat * N_m / R**16   # 6.1225e-11 m/s^2
L_GL_kpc = s_nat * R**12 / KPC  # 0.31757 kpc

# Milky Way model
V_CIRC = 220e3         # m/s (flat rotation speed)
R_SUN = 8.0            # kpc (Sun's galactic radius)


def milky_way_rotation_curve(R_kpc, v_circ=V_CIRC, R_rise=1.0):
    """Simple Milky Way model: linear rise to 1 kpc, flat beyond."""
    v = np.where(R_kpc < R_rise, v_circ * R_kpc / R_rise, v_circ)
    return v  # m/s


def compute_g_bar_galactic(R_kpc):
    """Baryonic acceleration from the Milky Way rotation curve."""
    v = milky_way_rotation_curve(R_kpc)
    return v**2 / (R_kpc * KPC)  # m/s^2


def compute_g_tilde(r_eval_kpc, R_grid_kpc, g_bar_grid):
    """Evaluate g_tilde at r_eval using the RST sqrt-exponential kernel."""
    sqrt_r = np.sqrt(r_eval_kpc)
    sqrt_R = np.sqrt(R_grid_kpc)
    K = np.exp(-np.abs(sqrt_r - sqrt_R) / L_GL_kpc)
    return np.sum(g_bar_grid * K) / np.sum(K)


def main():
    print("=" * 70)
    print("SOLAR SYSTEM SAFETY COMPUTATION")
    print("RST Nonlocal Term on Milky Way Rotation Curve")
    print("=" * 70)
    print()
    print(f"RST parameters:")
    print(f"  a0     = {a0:.4e} m/s^2")
    print(f"  L_GL   = {L_GL_kpc:.5f} kpc")
    print()
    print(f"Milky Way model:")
    print(f"  v_circ = {V_CIRC/1e3:.0f} km/s (flat for R > 1 kpc)")
    print(f"  R_sun  = {R_SUN:.1f} kpc")
    print()

    # Galactic radial grid
    R_grid = np.concatenate([
        np.linspace(0.01, 1.0, 100),
        np.linspace(1.0, 30.0, 500)
    ])
    g_bar_grid = compute_g_bar_galactic(R_grid)

    # Evaluate g_tilde at six Solar System positions
    offsets_AU = [-10, -1, 0, 1, 10, 40]
    print("Table E.1: g_tilde and g_extra at Solar System positions")
    print("-" * 70)
    print(f"{'Offset (AU)':>12} | {'R (kpc)':>16} | {'g_tilde (m/s^2)':>16} | {'g_extra (m/s^2)':>16}")
    print("-" * 70)

    g_tilde_values = []
    for offset_AU in offsets_AU:
        r_kpc = R_SUN + offset_AU * AU / KPC
        gt = compute_g_tilde(r_kpc, R_grid, g_bar_grid)
        ge = np.sqrt(a0 * gt)
        g_tilde_values.append(gt)
        print(f"{offset_AU:>+12d} | {r_kpc:>16.10f} | {gt:>16.6e} | {ge:>16.6e}")

    g_arr = np.array(g_tilde_values)
    rel_var = (g_arr.max() - g_arr.min()) / g_arr.mean()
    ge_var = np.sqrt(a0 * g_arr.max()) - np.sqrt(a0 * g_arr.min())

    print()
    print(f"Max relative variation in g_tilde across ±40 AU: {rel_var:.2e}")
    print(f"Variation in g_extra: {ge_var:.2e} m/s^2")
    print(f"Solar gravity at 1 AU: {G*M_sun/AU**2:.3e} m/s^2")
    print(f"Ratio (g_extra variation / g_solar): {ge_var / (G*M_sun/AU**2):.2e}")
    print()

    # Kernel weight decomposition
    r_eval = R_SUN
    sqrt_r = np.sqrt(r_eval)
    sqrt_R = np.sqrt(R_grid)
    K = np.exp(-np.abs(sqrt_r - sqrt_R) / L_GL_kpc)

    w_total = np.sum(K)
    regions = [
        ("R < 1 kpc (inner)", R_grid < 1.0),
        ("1-15 kpc (near Sun)", (R_grid >= 1.0) & (R_grid < 15.0)),
        ("R > 15 kpc (outer)", R_grid >= 15.0),
    ]

    print("Kernel weight decomposition at R = 8 kpc:")
    print("-" * 50)
    gt_total = compute_g_tilde(r_eval, R_grid, g_bar_grid)
    for label, mask in regions:
        w_frac = np.sum(K[mask]) / w_total
        g_contr = np.sum(g_bar_grid[mask] * K[mask]) / w_total
        print(f"  {label:<25} weight: {w_frac:.4f}  contribution: {g_contr/gt_total*100:.1f}%")

    # Galactic tidal gradient
    dR = 0.001  # kpc
    gt_plus = compute_g_tilde(R_SUN + dR, R_grid, g_bar_grid)
    gt_minus = compute_g_tilde(R_SUN - dR, R_grid, g_bar_grid)
    dg_dR = (gt_plus - gt_minus) / (2 * dR * KPC)  # m/s^2 per m
    da_extra_dR = 0.5 * np.sqrt(a0 / gt_total) * dg_dR
    tidal_40AU = da_extra_dR * 40 * AU

    print()
    print(f"Galactic tidal gradient:")
    print(f"  dg_tilde/dR = {dg_dR:.3e} m/s^2/m")
    print(f"  da_extra/dR = {da_extra_dR:.3e} m/s^2/m")
    print(f"  Tidal variation across 40 AU: {tidal_40AU:.3e} m/s^2")
    print(f"  Ratio to Solar gravity at 1 AU: {tidal_40AU/(G*M_sun/AU**2):.2e}")
    print()

    # Summary
    print("=" * 70)
    print("CONCLUSION: The RST nonlocal term is safe at Solar System scales.")
    print(f"  g_extra = {np.sqrt(a0*gt_total):.3e} m/s^2 (constant across Solar System)")
    print(f"  Tidal residual: {tidal_40AU:.1e} m/s^2 ({tidal_40AU/(G*M_sun/AU**2):.0e} of g_solar)")
    print(f"  A constant acceleration is undetectable in free fall (equivalence principle).")
    print("=" * 70)


if __name__ == "__main__":
    main()
