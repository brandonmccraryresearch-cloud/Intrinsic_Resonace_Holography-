# IRH v21.4 Interactive Notebooks

Complete collection of Jupyter notebooks demonstrating all aspects of the Intrinsic Resonance Holography framework.

**All notebooks updated**: December 27, 2025
- ✅ v21.4 manuscript references (fixed v21.1 references)
- ✅ ML surrogate model integration
- ✅ Enhanced transparency logging
- ✅ Failure detection and analysis

---

## 🚀 Featured: Exascale Full Repository Ultra

**File**: `exascale_full_repo_ultra.ipynb`

**Open in Colab**: [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/exascale_full_repo_ultra.ipynb)

The ultimate demonstration notebook with complete computational power.

### Features

- **Ultra-verbose transparency logging** - FULL verbosity level, every computation traced
- **ML surrogate models** - 10,000× speedup using neural networks (Phase 4.3)
- **Automatic failure detection** - Logs all failures to `../io/failures/`
- **Gemini AI integration** - AI-assisted code refactoring (Google Colab only)
- **Complete coverage** - All 8 implementation phases in one notebook
- **Performance metrics** - Real-time benchmarking

### Sections

1. Setup and Configuration
2. Ultra-Verbose Configuration
3. ML Surrogate Models Setup
4. Core RG Flow Computation (§1.2-1.3)
5. Observable Extraction (§3)
6. Standard Model Emergence (§3.1-3.4)
7. Cosmology and Predictions (§6-7)
8. Performance Summary
9. ⚠️ Failure Analysis and Gemini Refactoring
10. Final Summary and Results Export

**Runtime**: 10-15 minutes (2-3 minutes with ML surrogates)

---

## 📚 Core Notebooks

### 00_quickstart.ipynb

**Purpose**: Fast introduction to IRH framework  
**Runtime**: ~5 minutes  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/00_quickstart.ipynb)

**Topics**:
- Fixed point computation
- Basic observable extraction
- Minimal dependencies

---

### 01_group_manifold_visualization.ipynb

**Purpose**: Visualize G_inf = SU(2) × U(1)_φ manifold  
**Runtime**: ~3 minutes  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/01_group_manifold_visualization.ipynb)

**Topics**:
- Quaternionic representation of SU(2)
- QNCD metric visualization
- Group geodesics

**Theoretical Reference**: IRH v21.4 Part 1 §1.1

---

### 02_rg_flow_interactive.ipynb

**Purpose**: Interactive RG flow explorer  
**Runtime**: ~5 minutes  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/02_rg_flow_interactive.ipynb)

**Topics**:
- β-functions (Eq. 1.13)
- Fixed point search (Eq. 1.14)
- Trajectory visualization
- **NEW**: ML surrogate comparison

**Theoretical Reference**: IRH v21.4 Part 1 §1.2-1.3

---

### 03_observable_extraction.ipynb

**Purpose**: Extract physical constants from fixed point  
**Runtime**: ~10 minutes  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/03_observable_extraction.ipynb)

**Topics**:
- Fine structure constant α⁻¹ (Eq. 3.4-3.5)
- Universal exponent C_H (Eq. 1.16)
- Dark energy w₀ (Eq. 2.21-2.23)
- LIV parameter ξ (Eq. 2.24)
- **NEW**: ML-accelerated parameter scans

**Theoretical Reference**: IRH v21.4 Part 1 §3

---

### 04_falsification_analysis.ipynb

**Purpose**: Testable predictions and falsification criteria  
**Runtime**: ~8 minutes  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/04_falsification_analysis.ipynb)

**Topics**:
- LIV in gamma-ray astronomy
- Muon g-2 predictions (Appendix J.3)
- Gravitational wave sidebands (Appendix J.2)
- Neutrino mass hierarchy
- Dark energy evolution

**Theoretical Reference**: IRH v21.4 Part 2 §8, Appendices J-K

---

### 05_full_stack_execution.ipynb

**Purpose**: Complete end-to-end demonstration with scale selection  
**Runtime**: Configurable (30s to 1+ hour)  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/05_full_stack_execution.ipynb)

**Topics**:
- All phases: RG flow → observables → SM → cosmology
- Computation scale selector (quick/standard/full/exascale)
- Performance benchmarking

**Theoretical Reference**: Complete IRH v21.4 manuscript

---

### 05b_exascale_ml.ipynb

**Purpose**: ML surrogate models demonstration (Phase 4.3)  
**Runtime**: ~15 minutes (training included)  

[![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/05b_exascale_ml.ipynb)

**Topics**:
- RG flow surrogate training
- Uncertainty quantification (ensemble + MC dropout)
- Bayesian parameter optimization
- Performance benchmarking (10,000× speedup)

**Theoretical Reference**: IRH v21.4 Part 1 §1.2-1.3 + Phase 4.3 implementation

---

## 🔧 Technical Details

### Dependencies

All notebooks auto-install required dependencies in Colab. For local use:

```bash
pip install numpy scipy matplotlib jupyter
```

Optional (for ML features):
```bash
pip install scikit-learn  # ML surrogates (Phase 4.3)
```

### Data Files

Notebooks generate results in:
- **Colab**: `/content/drive/MyDrive/IRH_Results/` (if Drive mounted) or `/content/irh_results/`
- **Local**: `../results/` relative to notebook directory

### Failure Logs

All computational failures are logged to `../io/failures/` as JSON files with:
- Error details
- Parameters that caused failure
- Stack trace
- Suggested fixes (pattern-based)
- Gemini AI suggestions (Colab only)

### Transparency Logging

All notebooks use `TransparencyEngine` with configurable verbosity:
- **SILENT**: No output (not recommended)
- **MINIMAL**: Only results
- **STANDARD**: Results + validation
- **DETAILED**: + step-by-step derivation
- **FULL**: + all intermediate values (exascale_full_repo_ultra.ipynb)

---

## 📊 Update History

### December 27, 2025 - Major Update

**Changes applied to all notebooks**:
- ✅ Fixed v21.1 → v21.4 manuscript references (147 cells updated)
- ✅ Added ML surrogate model integration
- ✅ Added transparency logging infrastructure
- ✅ Added failure detection and logging
- ✅ Created `exascale_full_repo_ultra.ipynb` (new)
- ✅ Removed redundant `05_full_stack_execution_corrected.ipynb` from root

**Files modified**:
- `00_quickstart.ipynb` - 35 cells updated
- `01_group_manifold_visualization.ipynb` - 15 cells updated
- `02_rg_flow_interactive.ipynb` - 15 cells updated
- `03_observable_extraction.ipynb` - 34 cells updated
- `04_falsification_analysis.ipynb` - 18 cells updated
- `05_full_stack_execution.ipynb` - 17 cells updated
- `05b_exascale_ml.ipynb` - 13 cells updated

See `UPDATE_SUMMARY.json` for complete details.

### Computational Findings

**Important**: See [`docs/NOTEBOOK_FINDINGS.md`](../docs/NOTEBOOK_FINDINGS.md) for documented computational discrepancies and their theoretical explanations:

1. **Beta functions vs Fixed Point**: One-loop β-functions (Eq. 1.13) don't have zeros at Cosmic Fixed Point values (Eq. 1.14) - this is theoretically correct; the fixed point arises from the full Wetterich equation.

2. **Fermion Mass Predictions**: Current K_f derivation shows significant deviations from experiment, especially for first-generation fermions (electron: ~1700% deviation). Future work needed.

3. **Universal Exponent C_H**: Two values exist - ratio formula (0.75) and spectral zeta (0.045935703598). The spectral value is used for physical predictions.

---

## 🤝 Contributing

To add or modify notebooks:

1. Follow IRH v21.4 manuscript references
2. Use `TransparencyEngine` for logging
3. Add Colab badge at top
4. Test in both Colab and local Jupyter
5. Update this README

---

## 📖 References

- **IRH v21.4 Manuscript Part 1**: Foundation & Framework (Sections 1-4)
- **IRH v21.4 Manuscript Part 2**: QM, Cosmology, Predictions (Sections 5-8 + Appendices)
- **GitHub**: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography
- **Technical Docs**: `../docs/TECHNICAL_REFERENCE.md`

---

**Last Updated**: December 27, 2025  
**Total Notebooks**: 8 (including exascale ultra)  
**Total Cells Updated**: 147+ cells across 7 existing notebooks
