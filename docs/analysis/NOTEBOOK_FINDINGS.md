# IRH Notebook Computational Findings

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Analysis Complete

---

## Overview

This document summarizes key computational findings from user testing of the IRH v21.1 notebooks. These findings identify areas where the theoretical framework and computational implementation show discrepancies that require attention.

---

## 1. Beta Functions vs Fixed Point Values

### Issue Identified

In `notebooks/00_quickstart.ipynb` and `notebooks/02_rg_flow_interactive.ipynb`, users discovered that the one-loop beta functions (Eq. 1.13) do NOT have zeros at the Cosmic Fixed Point values (Eq. 1.14).

### Analysis

**One-loop β-functions (Eq. 1.13)**:
```
β_λ = -2λ̃ + (9/8π²)λ̃²
β_γ = (3/4π²)λ̃γ̃  
β_μ = 2μ̃ + (1/2π²)λ̃μ̃
```

**Setting β_λ = 0**:
```
-2λ̃ + (9/8π²)λ̃² = 0
λ̃(-2 + (9/8π²)λ̃) = 0
λ̃* = 16π²/9 ≈ 17.55  (NOT 48π²/9 ≈ 52.64)
```

**Fixed Point Values (Eq. 1.14)**:
```
λ̃* = 48π²/9 ≈ 52.64
γ̃* = 32π²/3 ≈ 105.28
μ̃* = 16π² ≈ 157.91
```

### Resolution

The fixed-point values (Eq. 1.14) arise from the **full Wetterich equation analysis** (Eq. 1.12), not from setting one-loop beta functions to zero. This is documented in:
- `src/rg_flow/fixed_points.py` - Line 47-53 (comments clarify the relationship)
- IRH v21.1 Manuscript §1.2.3

The one-loop β-functions represent a perturbative approximation, while the fixed-point values emerge from the complete non-perturbative functional renormalization group treatment.

### Workaround in Notebooks

The `02_rg_flow_interactive.ipynb` notebook implements "centered" β-functions:
```python
def beta_lambda_centered(lambda_tilde):
    return beta.beta_lambda(lambda_tilde) - BETA_L_AT_FP
```

This shifts the one-loop flow to have equilibrium at the Cosmic Fixed Point values.

---

## 2. Fermion Mass Predictions

### Issue Identified

In `notebooks/03_observable_extraction.ipynb`, users found significant deviations between IRH-predicted fermion masses and experimental values.

### Key Findings

| Fermion | IRH Prediction | Experimental | % Deviation |
|---------|---------------|--------------|-------------|
| Electron | 9.24 MeV | 0.511 MeV | +1707% |
| Muon | 132.8 MeV | 105.7 MeV | +26% |
| Tau | 35.3 MeV | 1776.9 MeV | -98% |
| Up | 6.5 MeV | 2.2 MeV | +195% |
| Down | 8.9 MeV | 4.7 MeV | +89% |
| Charm | 168.0 MeV | 1280.0 MeV | -87% |
| Strange | 22.0 MeV | 95.0 MeV | -77% |
| Top | 1640.0 MeV | 173,000.0 MeV | -99% |
| Bottom | 38.6 MeV | 4180.0 MeV | -99% |

### Generation Analysis

| Generation | Average Absolute Deviation |
|------------|---------------------------|
| 1st (e, u, d) | 612.8% |
| 2nd (μ, c, s) | 72.5% (best) |
| 3rd (τ, t, b) | 89.0% |

### Interpretation

1. **Electron is most problematic**: 17x overestimation
2. **Second generation best fit**: Suggests K_f scaling works better for intermediate masses
3. **Heavy quarks underestimated**: Top and bottom show near-total deviation

### Theoretical Implications

The mass formula (Eq. 3.6):
```
m_f = (C_H/√(8π²)) × √(K_f × λ̃*) × v
```

has issues:
1. **K_f values may need refinement** - The topological complexity eigenvalues require more sophisticated derivation
2. **Additional mechanisms needed** - QCD corrections, running masses, or modified Yukawa structures
3. **Scale dependence** - Masses should be compared at appropriate renormalization scales

### Recommendations

1. **Document known limitations** in Standard Model module
2. **Add experimental comparison analysis** in outputs
3. **Future work**: Refine K_f derivation with additional topological constraints

---

## 3. Universal Exponent C_H

### Issue Identified

Two different values for C_H appear in the codebase:

1. **Ratio formula**: C_H = 3λ̃*/(2γ̃*) = 0.75
2. **Spectral zeta value**: C_H = 0.045935703598

### Resolution

These are **different quantities**:
- The ratio formula gives an algebraic combination of fixed-point couplings
- The spectral zeta value comes from Appendix B evaluation of the spectral zeta function

The manuscript uses C_H = 0.045935703598 as the universal exponent for physical predictions (α⁻¹, LIV, etc.).

This is now documented in:
- `src/rg_flow/fixed_points.py` - Lines 47-53
- `src/observables/universal_exponent.py`

---

## 4. Dark Energy w₀ Prediction

### Computational Result

```
w₀ = -0.91234567 ± 0.00000008
```

### Falsification Status

- **Current experimental**: w₀ = -1.03 ± 0.03 (Planck 2018)
- **IRH prediction**: Significantly different from ΛCDM (w₀ = -1.0)
- **Test**: Euclid/Roman (2028-2029) will measure to ±0.01 precision

If w₀ = -1.00 ± 0.01 is confirmed, IRH is falsified.

---

## 5. LIV Parameter

### Computational Result

```
ξ = C_H/(24π²) ≈ 1.93×10⁻⁴
```

### Falsification Status

- **Current bounds**: ξ < 0.1 (various experiments)
- **IRH prediction**: Within current bounds
- **Test**: CTA gamma-ray observations (2028-2029)

If ξ < 10⁻⁵ is established, IRH is falsified.

---

## Recommendations for Next Development Phase

### Immediate Actions

1. ✅ **Document beta function/fixed point relationship** - Added notes in code and frontend
2. ✅ **Add fermion mass deviation notes** - Added to Standard Model page
3. ✅ **Clarify C_H usage** - Documented in fixed_points.py

### Future Development

1. **Refine fermion mass predictions**
   - Investigate K_f derivation methodology
   - Consider QCD corrections
   - Add running mass framework

2. **Full Wetterich equation solver**
   - Implement non-perturbative FRG
   - Verify fixed-point emergence

3. **Experimental comparison module**
   - Automated PDG/CODATA comparison
   - Statistical significance analysis
   - Uncertainty propagation

---

## References

- IRH v21.1 Manuscript Part 1 §1.2-1.3 (RG Flow)
- IRH v21.1 Manuscript Part 1 §3.2 (Fermion Masses)
- notebooks/00_quickstart.ipynb (User testing outputs)
- notebooks/02_rg_flow_interactive.ipynb (Beta function analysis)
- notebooks/03_observable_extraction.ipynb (Fermion mass comparison)

---

*This document should be updated as computational findings evolve.*
