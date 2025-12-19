# IRH Interactive Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/00_quickstart.ipynb)

Interactive Jupyter notebooks for exploring Intrinsic Resonance Holography (IRH) v21.1.

## Quick Start

Click any "Open in Colab" button to run notebooks directly in Google Colab - no installation required!

## Available Notebooks

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| **00_quickstart** | Quick introduction to IRH | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/00_quickstart.ipynb) |
| **01_group_manifold_visualization** | Visualize G_inf = SU(2)×U(1)_φ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/01_group_manifold_visualization.ipynb) |
| **02_rg_flow_interactive** | Explore RG flow and β-functions | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/02_rg_flow_interactive.ipynb) |
| **03_observable_extraction** | Extract physical constants | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/03_observable_extraction.ipynb) |
| **04_falsification_analysis** | Falsifiable predictions | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/04_falsification_analysis.ipynb) |
| **05_full_stack_execution** | Complete end-to-end demo with scale selection | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/notebooks/05_full_stack_execution.ipynb) |

## Computational Findings

**Important**: See [`docs/NOTEBOOK_FINDINGS.md`](../docs/NOTEBOOK_FINDINGS.md) for documented computational discrepancies and their theoretical explanations:

1. **Beta functions vs Fixed Point**: One-loop β-functions (Eq. 1.13) don't have zeros at Cosmic Fixed Point values (Eq. 1.14) - this is theoretically correct; the fixed point arises from the full Wetterich equation.

2. **Fermion Mass Predictions**: Current K_f derivation shows significant deviations from experiment, especially for first-generation fermions (electron: ~1700% deviation). Future work needed.

3. **Universal Exponent C_H**: Two values exist - ratio formula (0.75) and spectral zeta (0.045935703598). The spectral value is used for physical predictions.

## Topics Covered

### 00_quickstart
- Installation and setup
- Core concepts overview
- Computing the Cosmic Fixed Point
- Deriving physical constants
- Standard Model emergence

### 01_group_manifold_visualization
- SU(2) as the 3-sphere (quaternions)
- U(1) phase circle
- Product manifold G_inf = SU(2) × U(1)_φ
- QNCD metric visualization

### 02_rg_flow_interactive
- β-functions (Eq. 1.13)
- Cosmic Fixed Point (Eq. 1.14)
- Flow trajectories in coupling space
- Stability analysis
- Spectral dimension flow

### 03_observable_extraction
- Universal exponent C_H (Eq. 1.16)
- Fine-structure constant α⁻¹ (Eqs. 3.4-3.5)
- Standard Model predictions
- Comparison with experiment

### 04_falsification_analysis
- Dark energy equation of state w₀
- Lorentz Invariance Violation
- Neutrino mass hierarchy
- Muon g-2 anomaly
- Falsification timeline (2026-2030)

### 05_full_stack_execution (NEW)
- **Complete end-to-end demonstration**
- **Configurable computation scale**: quick, standard, full, exascale
- RG flow integration
- Observable extraction
- Standard Model emergence
- Cosmological predictions
- ML surrogate model training
- Full report generation

## Running Locally

```bash
# Navigate to notebooks directory
cd notebooks/

# Launch Jupyter Lab
jupyter lab

# Or launch specific notebook
jupyter notebook 00_quickstart.ipynb
```

## Requirements

For local execution:
- Python 3.10+
- NumPy, SciPy, Matplotlib
- IRH modules in Python path

Colab notebooks auto-install dependencies.

## Theoretical Reference

All notebooks reference the canonical manuscript:
- [IRH v21.1 Manuscript Part 1](../Intrinsic_Resonance_Holography-v21.1-Part1.md) (Sections 1-4)
- [IRH v21.1 Manuscript Part 2](../Intrinsic_Resonance_Holography-v21.1-Part2.md) (Sections 5-8 + Appendices)

## Citation

```bibtex
@software{IRH_v21_computational_2025,
  title={Intrinsic Resonance Holography v21.1: Computational Framework},
  author={McCrary, Brandon D.},
  year={2025},
  url={https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-}
}
```
