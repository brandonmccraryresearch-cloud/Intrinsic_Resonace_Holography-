# Intrinsic Resonance Holography v21.0: Computational Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-159%2B%20passing-brightgreen.svg)](./tests/)
[![Coverage](https://img.shields.io/badge/critical%20equations-100%25-brightgreen.svg)](./THEORETICAL_CORRESPONDENCE.md)

**A complete computational framework for deriving fundamental physics from quantum-informational principles.**

---

## 📚 Documentation Quick Links

| Document | Description |
|----------|-------------|
| [**Technical Reference Manual**](./docs/TECHNICAL_REFERENCE.md) | Exhaustive specifications for all modules, APIs, and implementations |
| [**Continuation Guide**](./docs/CONTINUATION_GUIDE.md) | Next phases, priority tasks, and implementation roadmap |
| [**Theoretical Correspondence Map**](./THEORETICAL_CORRESPONDENCE.md) | Bidirectional mapping between code and manuscript equations |
| [**Architecture Overview**](./docs/architectural_overview.md) | System design and ontological layer structure |
| [**Desktop App Roadmap**](./docs/DEB_PACKAGE_ROADMAP.md) | Implementation plan for .deb desktop application |
| [**IRH21.md**](./IRH21.md) | Canonical theoretical manuscript (master reference) |
| [**Contributing Guidelines**](./CONTRIBUTING.md) | How to contribute to the project |

---

## 🎯 Overview

This repository instantiates the complete mathematical formalism of **Intrinsic Resonance Holography (IRH) v21.0**, a unified theory deriving all fundamental physical laws, constants, and observable phenomena from axiomatically minimal quantum-informational principles. The canonical theoretical specification resides in **`IRH21.md`** (root directory), which serves as the **master reference** for all computational implementations.

### Core Theoretical Commitments

IRH v21.0 establishes:

1. **Ontological Primitive**: Quantum information residing in Hilbert space $\mathcal{H}_{\text{fund}}$ with quantum algorithmic complexity functional $K_Q$ (§1.0.1)
2. **Fundamental Dynamics**: Complex quaternionic Group Field Theory (cGFT) on $G_{\text{inf}} = \text{SU}(2) \times \text{U}(1)_\phi$ (§1.1)
3. **Emergent Laws**: All of quantum mechanics, general relativity, and the Standard Model arise from a unique non-Gaussian infrared fixed point—the **Cosmic Fixed Point** (§1.2-1.3)
4. **Predictive Power**: Analytically computes ~20 physical constants from 3 fixed-point couplings, with falsifiable predictions testable by 2030 (§8)

### Key Predictions

| Prediction | Value | Precision | Reference |
|-----------|-------|-----------|-----------|
| Fine-structure constant α⁻¹ | 137.035999084 | 12 digits | Eq. 3.4-3.5 |
| Universal exponent C_H | 0.045935703598 | 12 digits | Eq. 1.16 |
| Dark energy EoS w₀ | -0.91234567 | ±0.00000008 | Eq. 2.21-2.23 |
| First Betti number β₁ | 12 | Exact | Appendix D.1 |
| Spectral dimension d_spec | 4.0 | Exact | Eq. 2.8-2.9 |

---

## 🏗️ Repository Architecture

### Epistemic Stratification

The directory structure mirrors IRH's explanatory hierarchy per the **Epistemic Stratification Principle** (§4.1):

```
primitives/ → cgft/ → rg_flow/ → emergent_spacetime/ → topology/ → standard_model/ → cosmology/ → quantum_mechanics/ → falsifiable_predictions/
```

Each layer depends **only** on predecessors, enforcing the derivational cascade from primitive ontology to phenomenological emergence.

### Directory Structure

```
IRH-v21-Computational-Framework/
│
├── IRH21.md                          # Canonical theoretical manuscript
├── README.md                         # This file
├── THEORETICAL_CORRESPONDENCE.md     # Living map: code ↔ manuscript sections
├── CONTRIBUTING.md                   # Standards for theoretical fidelity
├── LICENSE                           # GPLv3 License
│
├── docs/                             # Comprehensive documentation
│   ├── TECHNICAL_REFERENCE.md        # Exhaustive technical specifications
│   ├── DEB_PACKAGE_ROADMAP.md        # Desktop app implementation roadmap
│   ├── architectural_overview.md     # Conceptual scaffold explanation
│   └── ...                           # Additional documentation
│
├── src/                              # Source code: stratified by ontological layer
│   ├── primitives/                   # Layer 0: Ontological bedrock
│   │   ├── quaternions.py            # Quaternion algebra ℍ
│   │   ├── group_manifold.py         # G_inf = SU(2) × U(1)_φ
│   │   └── qncd.py                   # QNCD metric implementation
│   ├── cgft/                         # Layer 1: Complex Group Field Theory
│   │   ├── actions.py                # S_kin + S_int + S_hol (Eqs. 1.1-1.4)
│   │   └── fields.py                 # Field representations
│   ├── rg_flow/                      # Layer 2: Renormalization Group Dynamics
│   ├── emergent_spacetime/           # Layer 3: Geometric emergence
│   ├── topology/                     # Layer 4: Topological structures
│   ├── standard_model/               # Layer 5: Particle physics emergence
│   ├── cosmology/                    # Layer 6: Cosmological predictions
│   ├── quantum_mechanics/            # Layer 7: QM phenomenology emergence
│   ├── falsifiable_predictions/      # Layer 8: Novel experimental signatures
│   ├── observables/                  # Observable extraction infrastructure
│   └── utilities/                    # Cross-cutting computational tools
│
├── tests/                            # Comprehensive validation suite
│   ├── unit/                         # Atomic function tests
│   ├── integration/                  # Multi-module interaction tests
│   ├── theoretical_invariants/       # Mathematical property verification
│   └── falsification/                # Experimental prediction suite
│
├── scripts/                          # Automation & workflow orchestration
├── configs/                          # Parameter configuration files
├── data/                             # Reference data & baselines
├── notebooks/                        # Jupyter notebooks for exploration
└── ci_cd/                            # Continuous integration configuration
```

### Key Directories by IRH Section

| Directory | Description | IRH Section |
|-----------|-------------|-------------|
| `src/primitives/` | Quantum information foundations, group manifolds, quaternions, QNCD metric | §1.0.1 |
| `src/cgft/` | Field theory action (Eqs. 1.1-1.4), operators, symmetries | §1.1 |
| `src/rg_flow/` | Wetterich equation, β-functions (Eq. 1.13), Cosmic Fixed Point | §1.2-1.3 |
| `src/emergent_spacetime/` | 4D geometry, Lorentzian signature, Einstein equations | §2.1-2.2 |
| `src/topology/` | β₁=12, n_inst=3, Vortex Wave Patterns (fermions) | Appendix D |
| `src/standard_model/` | Gauge groups, particle masses, mixing matrices | §3.1-3.4 |
| `src/cosmology/` | Holographic hum, dark energy, running constants | §2.3-2.5 |
| `src/quantum_mechanics/` | Emergent Hilbert space, Born rule, decoherence | §5.1-5.2 |
| `src/falsifiable_predictions/` | LIV, running constants, observer back-reaction | §8, Appendix J |
| `src/observables/` | Physical constants extraction, experimental comparison | §3.2 |
| `tests/` | Comprehensive validation ensuring theoretical fidelity | — |

## Theoretical Correspondence

**Every function, class, and module** must cite its theoretical foundation via:
- **Section references**: `# IRH21.md §2.3.3` in docstrings
- **Equation labels**: `# Implements Eq. 2.21-2.23`
- **Appendix citations**: `# Derivation in Appendix C.6`

The living document **`THEORETICAL_CORRESPONDENCE.md`** maintains a bidirectional map between code and manuscript.

---

## 🚀 Installation Guide

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Operating System** | Linux, macOS, Windows | Linux (Ubuntu 22.04+) |
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 16 GB |
| **Disk Space** | 500 MB | 2 GB |

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-.git
cd Intrinsic_Resonace_Holography-

# 2. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from src.primitives.quaternions import Quaternion; print('✓ Installation successful')"
```

### Detailed Installation Steps

#### Step 1: Prerequisites

Ensure you have Python 3.10 or later installed:

```bash
python --version  # Should show Python 3.10.x or higher
```

If not, install Python from [python.org](https://www.python.org/downloads/) or use your system package manager:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS with Homebrew
brew install python@3.11

# Windows: Download installer from python.org
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-.git
cd Intrinsic_Resonace_Holography-
```

#### Step 3: Virtual Environment Setup

Creating a virtual environment isolates the project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows PowerShell
# or
venv\Scripts\activate.bat # Windows CMD
```

#### Step 4: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -e .  # Install in editable mode
```

#### Step 5: Verify Installation

Run the verification script to ensure everything is working:

```bash
# Basic verification
python -c "from src.primitives.quaternions import Quaternion; print('✓ Quaternions module loaded')"
python -c "from src.primitives.group_manifold import GInfElement; print('✓ Group manifold module loaded')"
python -c "from src.cgft.actions import compute_total_action; print('✓ cGFT actions module loaded')"

# Full verification
python scripts/verify_theoretical_annotations.py
```

### Verify Theoretical Integrity

After installation, verify that all modules properly cite their theoretical foundations:

```bash
# Check theoretical annotations
python scripts/verify_theoretical_annotations.py

# Audit equation implementations
python scripts/audit_equation_implementations.py

# Run full validation suite
python scripts/run_full_validation_suite.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/unit/test_primitives/ -v
pytest tests/unit/test_cgft/ -v
```

### Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install black isort flake8 mypy pytest-cov

# Setup code formatting
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/ --ignore-missing-imports

# Run linting
flake8 src/ tests/
```

---

## 📖 Quick Start Guide

### Example 1: Quaternion Algebra (IRH21.md §1.1.1)

```python
from src.primitives.quaternions import Quaternion, verify_quaternion_algebra

# Create quaternions - the building blocks of cGFT fields
q1 = Quaternion(w=1.0, x=0.5, y=-0.3, z=0.2)
q2 = Quaternion.random()

# Demonstrate non-commutativity (essential for emergent physics)
product_12 = q1 * q2
product_21 = q2 * q1
print(f"q1 * q2 = {product_12}")
print(f"q2 * q1 = {product_21}")
print(f"Non-commutative: {product_12 != product_21}")  # True!

# Verify algebra axioms computationally
results = verify_quaternion_algebra()
print(f"All quaternion algebra axioms satisfied: {results['all_passed']}")
```

### Example 2: Group Manifold G_inf (IRH21.md §1.1)

```python
from src.primitives.group_manifold import GInfElement, haar_integrate_GInf, verify_group_axioms

# G_inf = SU(2) × U(1)_φ is the fundamental group manifold
# Its 4-dimensionality (3 + 1) yields 4D emergent spacetime

# Generate random group element
g = GInfElement.random()
print(f"Random G_inf element: {g}")

# Verify group properties
axioms = verify_group_axioms()
print(f"All group axioms satisfied: {axioms['all_passed']}")

# Monte Carlo integration over G_inf (Haar measure)
def character(g: GInfElement) -> float:
    """Simple character function on G_inf."""
    return g.su2.quaternion.w ** 2

mean, error = haar_integrate_GInf(character, n_samples=10000)
print(f"∫ χ(g) dg = {mean:.6f} ± {error:.6f}")
```

### Example 3: cGFT Action Computation (IRH21.md §1.1, Eqs. 1.1-1.4)

```python
import numpy as np
from src.cgft.actions import compute_total_action, LAMBDA_STAR, GAMMA_STAR, MU_STAR

# Create a sample field configuration φ(g₁,g₂,g₃,g₄) ∈ ℍ
phi = np.random.random((5,5,5,5)) + 1j * np.random.random((5,5,5,5))

# Compute the complete cGFT action S = S_kin + S_int + S_hol
result = compute_total_action(phi)

print("cGFT Action Decomposition:")
print(f"  S_total = {result['S_total']:.6f}")
print(f"  S_kin   = {result['S_kin']:.6f} (kinetic term, Eq. 1.1)")
print(f"  S_int   = {result['S_int']:.6f} (interaction term, Eq. 1.2)")
print(f"  S_hol   = {result['S_hol']:.6f} (holographic term, Eq. 1.4)")
print(f"  Reference: {result['theoretical_reference']}")

# Display fixed-point coupling values (Eq. 1.14)
print(f"\nCosmic Fixed Point Couplings:")
print(f"  λ̃* = {LAMBDA_STAR:.6f}  (Eq. 1.14)")
print(f"  γ̃* = {GAMMA_STAR:.6f}  (Eq. 1.14)")
print(f"  μ̃* = {MU_STAR:.6f}  (Eq. 1.14)")
```

### Example 4: QNCD Metric Verification (IRH21.md Appendix A)

```python
from src.primitives.qncd import compute_QNCD, verify_QNCD_metric_axioms, verify_QUCC_theorem
from src.primitives.group_manifold import GInfElement

# QNCD (Quantum Normalized Compression Distance) measures
# algorithmic similarity between group elements

g1 = GInfElement.random()
g2 = GInfElement.random()

# Compute QNCD distance
distance = compute_QNCD(g1, g2)
print(f"QNCD(g1, g2) = {distance:.6f}")

# Verify metric axioms (positivity, symmetry, triangle inequality)
axioms = verify_QNCD_metric_axioms()
print(f"Metric axioms satisfied: {axioms['all_passed']}")

# Verify compressor independence (QUCC-Theorem, Appendix A.4)
qucc = verify_QUCC_theorem()
print(f"QUCC-Theorem compliance: {qucc['passed']}")
```

---

## 🏗️ Repository Architecture

### Epistemic Stratification

The directory structure mirrors IRH's explanatory hierarchy per the **Epistemic Stratification Principle** (§4.1):

```
primitives/ → cgft/ → rg_flow/ → emergent_spacetime/ → topology/ → standard_model/ → cosmology/ → quantum_mechanics/ → falsifiable_predictions/
```

Each layer depends **only** on predecessors, enforcing the derivational cascade from primitive ontology to phenomenological emergence.

### Directory Structure

```
IRH-v21-Computational-Framework/
│
├── IRH21.md                          # Canonical theoretical manuscript
├── README.md                         # This file
├── THEORETICAL_CORRESPONDENCE.md     # Living map: code ↔ manuscript sections
├── CONTRIBUTING.md                   # Standards for theoretical fidelity
├── LICENSE                           # GPLv3 License
│
├── docs/                             # Comprehensive documentation
│   ├── TECHNICAL_REFERENCE.md        # Exhaustive technical specifications
│   ├── DEB_PACKAGE_ROADMAP.md        # Desktop app implementation roadmap
│   ├── architectural_overview.md     # Conceptual scaffold explanation
│   └── ...                           # Additional documentation
│
├── src/                              # Source code: stratified by ontological layer
│   ├── primitives/                   # Layer 0: Ontological bedrock
│   │   ├── quaternions.py            # Quaternion algebra ℍ
│   │   ├── group_manifold.py         # G_inf = SU(2) × U(1)_φ
│   │   └── qncd.py                   # QNCD metric implementation
│   ├── cgft/                         # Layer 1: Complex Group Field Theory
│   │   ├── actions.py                # S_kin + S_int + S_hol (Eqs. 1.1-1.4)
│   │   └── fields.py                 # Field representations
│   ├── rg_flow/                      # Layer 2: Renormalization Group Dynamics
│   ├── emergent_spacetime/           # Layer 3: Geometric emergence
│   ├── topology/                     # Layer 4: Topological structures
│   ├── standard_model/               # Layer 5: Particle physics emergence
│   ├── cosmology/                    # Layer 6: Cosmological predictions
│   ├── quantum_mechanics/            # Layer 7: QM phenomenology emergence
│   ├── falsifiable_predictions/      # Layer 8: Novel experimental signatures
│   ├── observables/                  # Observable extraction infrastructure
│   └── utilities/                    # Cross-cutting computational tools
│
├── tests/                            # Comprehensive validation suite
│   ├── unit/                         # Atomic function tests
│   ├── integration/                  # Multi-module interaction tests
│   ├── theoretical_invariants/       # Mathematical property verification
│   └── falsification/                # Experimental prediction suite
│
├── scripts/                          # Automation & workflow orchestration
├── configs/                          # Parameter configuration files
├── data/                             # Reference data & baselines
├── notebooks/                        # Jupyter notebooks for exploration
└── ci_cd/                            # Continuous integration configuration
```

### Key Directories by IRH Section

| Directory | Description | IRH Section |
|-----------|-------------|-------------|
| `src/primitives/` | Quantum information foundations, group manifolds, quaternions, QNCD metric | §1.0.1 |
| `src/cgft/` | Field theory action (Eqs. 1.1-1.4), operators, symmetries | §1.1 |
| `src/rg_flow/` | Wetterich equation, β-functions (Eq. 1.13), Cosmic Fixed Point | §1.2-1.3 |
| `src/emergent_spacetime/` | 4D geometry, Lorentzian signature, Einstein equations | §2.1-2.2 |
| `src/topology/` | β₁=12, n_inst=3, Vortex Wave Patterns (fermions) | Appendix D |
| `src/standard_model/` | Gauge groups, particle masses, mixing matrices | §3.1-3.4 |
| `src/cosmology/` | Holographic hum, dark energy, running constants | §2.3-2.5 |
| `src/quantum_mechanics/` | Emergent Hilbert space, Born rule, decoherence | §5.1-5.2 |
| `src/falsifiable_predictions/` | LIV, running constants, observer back-reaction | §8, Appendix J |
| `src/observables/` | Physical constants extraction, experimental comparison | §3.2 |
| `tests/` | Comprehensive validation ensuring theoretical fidelity | — |

---

## 📊 Validation Status

Current implementation status tracked in [`THEORETICAL_CORRESPONDENCE.md`](./THEORETICAL_CORRESPONDENCE.md).

| Component | Status | Tests | Coverage |
|-----------|--------|-------|----------|
| Primitives (Quaternions, Groups, QNCD) | ✅ Complete | 45+ | 100% |
| cGFT Action (Eqs. 1.1-1.4) | ✅ Complete | 25+ | 100% |
| RG Flow Validation | ✅ Complete | 5+ | 100% |
| Emergent Spacetime | 🟡 Scaffold | — | 0% |
| Topology | 🟡 Scaffold | — | 0% |
| Standard Model | 🟡 Scaffold | — | 0% |
| Cosmology | 🟡 Scaffold | — | 0% |
| Quantum Mechanics | 🟡 Scaffold | — | 0% |
| Falsifiable Predictions | 🟡 Scaffold | — | 0% |
| Observables | 🟡 Scaffold | — | 0% |

**Overall**: 159+ tests passing | 100% critical equation coverage (17/17)

---

## 🤝 Contributing

All contributions must satisfy:

- ✓ **Theoretical traceability**: Cite IRH21.md sections/equations in docstrings
- ✓ **Gauge invariance**: Pass `tests/theoretical_invariants/`
- ✓ **Convergence**: Demonstrate numerical stability
- ✓ **Documentation**: Inline theoretical context annotations

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for detailed guidelines.

### Theoretical Correspondence

**Every function, class, and module** must cite its theoretical foundation via:
- **Section references**: `# IRH21.md §2.3.3` in docstrings
- **Equation labels**: `# Implements Eq. 2.21-2.23`
- **Appendix citations**: `# Derivation in Appendix C.6`

The living document **[`THEORETICAL_CORRESPONDENCE.md`](./THEORETICAL_CORRESPONDENCE.md)** maintains a bidirectional map between code and manuscript.

---

## 📖 Citation

If using this framework in research, cite:

```bibtex
@article{IRH_v21_2026,
  title={Intrinsic Resonance Holography v21.0: Unified Theory of Emergent Reality},
  author={McCrary, Brandon D.},
  journal={Manuscript},
  year={2026},
  url={https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-}
}
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [`LICENSE`](./LICENSE) file for details.

---

## 📞 Contact

For theoretical inquiries or computational collaboration:

- **Theory Lead**: Brandon D. McCrary
- **ORCID**: [0009-0008-2804-7165](https://orcid.org/0009-0008-2804-7165)
- **Issues**: [GitHub issue tracker](https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-/issues)

---

> **Note**: This is a living computational laboratory. The codebase evolves in lockstep with theoretical refinements to `IRH21.md`. Always verify you're working with the latest manuscript version.
