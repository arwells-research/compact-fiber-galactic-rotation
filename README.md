# Galactic Rotation from the Compact Fiber
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19446509.svg)](https://doi.org/10.5281/zenodo.19446509)

**Author:** A. R. Wells  
**Affiliation:** Dual-Frame Research Group  
**License:** MIT  
**Repository:** `arwells-research/compact-fiber-galactic-rotation`  
**Status:** Paper companion repository

---

## Overview

This repository accompanies the paper:

**_Galactic Rotation from the Compact Fiber: The Gravitational Limit, R¹⁶, and a Zero-Parameter Rotation-Curve Model_**

The paper derives a galactic rotation-curve model from the compact fiber framework developed in three companion papers:

- **Paper I:** The compact fiber as the canonical admissibility structure for discrete scalar motion ([doi:10.5281/zenodo.19423877](https://doi.org/10.5281/zenodo.19423877))
- **Paper II:** Atomic structure from the compact fiber ([doi:10.5281/zenodo.19426034](https://doi.org/10.5281/zenodo.19426034))
- **Paper III:** Effective quantum and gravitational structure from the compact fiber ([doi:10.5281/zenodo.19434652](https://doi.org/10.5281/zenodo.19434652))

The model uses **zero calibrated galactic parameters**. All quantities are derived from the natural unit system and the inter-regional ratio R = 156.444:

- Acceleration scale: a₀ = a_nat × N_m / R¹⁶ = 6.12 × 10⁻¹¹ m/s²
- Coherence length: L_GL = s_nat × R¹² = 0.318 kpc
- Kernel shape: exp(−|√r − √r'|/L) from the second-power relation r = ρ²
- Coupling law: g + √(a₀ · g̃) from covering degree d = 2

Tested against 171 SPARC galaxies, the model achieves slightly lower median error than a fixed MOND baseline (median χ²/pt 10.71 vs 11.25) with no fitted parameters.

---

## Repository Structure

    compact-fiber-galactic-rotation/
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── rst_sparc_results.csv          # Per-galaxy results (171 galaxies)
    ├── paper/
    │   └── galactic_rotation_compact_fiber.tex   # Paper LaTeX source
    ├── src/
    │   ├── __init__.py
    │   ├── sparc_io.py                # SPARC data loader
    │   ├── models.py                  # MOND benchmark + DFT-B comparator
    │   └── rst_model.py               # RST rotation model
    ├── experiments/
    │   ├── __init__.py
    │   ├── run_rst_final.py           # Reproduces all paper tables
    │   └── run_solar_safety.py        # Reproduces Solar System appendix
    └── data/
        └── sparc/
            ├── Rotmod_LTG.zip         # SPARC rotation curve archive
            └── Rotmod_LTG/            # Extracted data files (175 galaxies)

---

## The RST Rotation Model

The tested acceleration law takes the form:

    g(r) = g_bar(r) + sqrt( a₀ · g̃(r) )

where:

- `g_bar(r)` is the Newtonian baryonic acceleration from SPARC components
- `g̃(r)` is a nonlocal, radially smoothed baryonic field
- `a₀ = 6.1225 × 10⁻¹¹ m/s²` (derived from R¹⁶)
- `L_GL = 0.31757 kpc` (derived from R¹²)

The nonlocal field uses a sqrt-exponential kernel:

    g̃(r) = Σⱼ g_bar(rⱼ) · exp(−|√r − √rⱼ|/L_GL)
            ─────────────────────────────────────────
            Σⱼ exp(−|√r − √rⱼ|/L_GL)

The sqrt-exponential form is structurally induced by the second-power relation r = ρ² (Paper II). The coupling exponent (square root) follows from the covering degree d = 2 (Paper I).

**No parameters are calibrated on the SPARC sample.** All model ingredients are fixed before comparison.

---

## Benchmark: RAR / MOND Mapping

For comparison, the repository implements a standard exponential-style RAR/MOND mapping:

    g_total = g_bar / (1 − exp(−sqrt(g_bar / a₀_MOND)))

with `a₀_MOND = 1.2 × 10⁻¹⁰ m/s²`.

This benchmark is used **only** as a reference comparator.

---

## Dataset

The SPARC Rotmod_LTG catalog (Lelli, McGaugh, & Schombert 2016, AJ 152, 157):

- 175 late-type galaxies with observed rotation curves
- Baryonic components: gas, disk, bulge
- 171 galaxies pass the inclusion rule (≥ 5 points with errV > 0)

Stored at `data/sparc/Rotmod_LTG/`. The ZIP archive includes a SHA-256 checksum.

SPARC data are publicly available and intended for open scientific use. Users should cite the original source:

> Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016), "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves," *Astron. J.* 152, 157.

---

## Reproducing the Paper

### Requirements

    pip install numpy pandas

### Reproduce all tables

    cd experiments && python run_rst_final.py

This single command:

1. Loads 171 SPARC galaxies
2. Evaluates the R-derived RST model (zero parameters)
3. Evaluates the MOND benchmark
4. Reports headline statistics (median, percentiles, win count)
5. Runs all sensitivity scans (covering degree, kernel power, curvature modulation, M/L ratio)
6. Calibrates the DFT-B ablation comparator
7. Writes per-galaxy results to `rst_sparc_results.csv`

### Reproduce Solar System safety analysis

    cd experiments && python run_solar_safety.py

This evaluates g̃ at six Solar System positions on a Milky Way rotation curve model and computes the kernel weight decomposition and galactic tidal gradient.

---

## Headline Results (SPARC Rotmod_LTG, 171 galaxies)

| Model | Parameters | Median χ²/pt | Wins vs MOND |
|-------|-----------|-------------|-------------|
| MOND  | 1 (empirical) | 11.25 | — |
| RST   | 0 (derived)   | 10.71 | 87/171 |

Sign test p = 0.65 (population-level statistical tie).

The zero-parameter RST model achieves slightly lower median error than the one-parameter MOND baseline, with better performance in low-mass gas-rich dwarfs (100% win rate for V_flat < 50 km/s) and near-zero mean residuals in the mid-to-outer disk.

---

## Falsification Criteria

The model makes specific falsifiable predictions:

1. Any clean-interior dwarf galaxy (V_flat < 50 km/s) where RST fails with χ² > 15 would indicate a structural failure.
2. A covering degree d ≠ 2 required for good fits would falsify the compact fiber.
3. If a future larger sample shifted the kernel-power optimum away from p = 0.5, the second-power relation would be in tension.

---

## Scope Notes

This repository is limited to the galactic rotation-curve analysis. It does not address cosmological observables (CMB, BAO, lensing), elliptical galaxies, galaxy clusters, or detailed Solar System ephemeris propagation.

The galactic model is a conditional consequence of Papers I–III plus one auxiliary assumption (exponential smoothing in the time-region radial coordinate). It is not a uniquely forced derivation from first principles. See the paper for full status discipline.

---

## License

MIT. See `LICENSE`.

---

## Citation

If you use the paper or repository, please cite the Zenodo record below. If a version-specific DOI is assigned, prefer citing that specific version for reproducibility.

Recommended citation format:

A. R. Wells (2026).  
*Galactic Rotation from the Compact Fiber: The Gravitational Limit, R^16, and a Zero-Parameter Rotation-Curve Model*.  
Zenodo. https://doi.org/10.5281/zenodo.19446509

The code in this release is licensed under the MIT License.

The paper is released under Creative Commons Attribution 4.0 (CC BY 4.0).
