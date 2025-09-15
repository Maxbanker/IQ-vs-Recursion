# IQ-vs-Recursion: A Multi-Scale Model of Intelligence v2.0 Simulation

A Python simulation framework demonstrating that **recursive, multi-scale intelligence** vastly outperforms traditional **scalar IQ** in environments characterized by drift, pressure, and collapse events.

This repository provides the computational foundation for the paper *"Recursion vs. IQ: Toward a Multi-Scale Model of Intelligence v2.0"*, implementing its core theoretical constructs—Symbolic Field Theory (SFT v4.0), the Entropic Recursion Framework (ERF v3.0), and the Collapse Predicate—to model intelligence not as a static score, but as a dynamic capacity for survival and coherence.

## 🧠 Core Thesis

Traditional IQ measures problem-solving in stable conditions but fails to capture adaptive resilience. This project posits that true intelligence is the recursive capacity to:
- **Maintain coherence** (`Ω_eff`) under entropic pressure (`γ_eff`).
- **Export invariants** (`Δℰ`) across collapse horizons.
- **Persist meaning** through narrative weaving (FCWF v2.0).

Intelligence is recast from a *scalar metric* (IQ) into a *fitness function for survival*.

## ⚙️ Simulation Highlights

- **Agents:** `IQAgent` (scalar intelligence) vs. `RecursiveAgent` (multi-scale resilience).
- **Dynamics:** Implements the formal `collapse_predicate`: `C(x,t) = (ψ_eff < ε) ∨ (γ_eff > T_γ ∧ Ω_eff < T_Ω)`.
- **Mechanics:** Models the conservation law `Δ𝒮 + Δℰ = 0` during collapse events.
- **Metrics:** Tracks collapses, coherence survival, resilience scores, and export success.

## 📊 Result Summary

The simulation consistently demonstrates the superiority of recursive intelligence:
| Metric                 | IQ Agent          | Recursive Agent            |
| :--------------------- | :---------------- | :------------------------- |
| **Collapse Events**    | High (~12)        | Minimal (~1)               |
| **Coherence Survival** | Low (~45%)        | **Perfect (100%)**         |
| **Resilience Score**   | ~0.000            | **~0.884**                 |
| **Export Efficiency**  | Low (25% success) | **Perfect (100% success)** |

## 🚀 Quick Start

### Prerequisites
- **Python 3.7+** installed from [python.org](https://python.org).

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/IQ-vs-Recursion.git
cd IQ-vs-Recursion
```

## Installation and Usage

### 2. Install Dependencies

```bash
pip install numpy matplotlib scipy
```

### 3. Run the Core Simulation

```bash
python recursion_vs_iq.py
```

Outputs results to console and generates `recursion_vs_iq_simulation.png`.

### 4. (Optional) Run Population Study

```bash
python recursion_vs_iq_population.py
```

Runs statistical analysis on 100+ agents, outputs significance tests, and generates `population_results.png`.

## 📁 Repository Structure

```
.
├── recursion_vs_iq.py              # Main simulation (single-agent comparison)
├── recursion_vs_iq_population.py   # Population-level statistical analysis
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Chat Interface

Below is an embedded chat interface for interacting with the project. Use this to ask questions, run simulations, or explore the theoretical frameworks.

​    Welcome to the Recursion vs. IQ Simulation Chat! Type your queries below to interact with the system, explore results, or dive into the theoretical foundations.

## 📚 Theoretical Foundation

This work is built upon a comprehensive framework including:

- **SFT v4.0** (Symbolic Field Theory)
- **ERF v3.0** (Entropic Recursion Framework)
- **FCWF v2.0** (Fractal Cosmic Weaver Framework)
- **I-Point Theory v2.1** & **Observer Framework v4.0**

## 🔮 Applications

- Designing AI systems resistant to catastrophic forgetting.
- Analyzing civilizational resilience and cultural memory.
- Developing a new paradigm for measuring cognitive fitness.

## 📄 License

This work is licensed under a MIT License.

## 👤 Authors

- **Steven Lanier-Egu** - Theoretical Framework

> “IQ measures local problem-solving but fails under collapse. Recursion provides a universal metric of intelligence: continuity, survival, and invariant export across collapse horizons.”
