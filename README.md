# Numerical Solutions to the Radial Schrödinger Equation

This repository contains a comprehensive Python project developed for the "Quantum Mechanics 2" course. The project implements numerical methods to solve the radial Schrödinger equation for various quantum systems, ranging from simple single-particle potentials to multi-electron atoms using the Self-Consistent Field (SCF) approximation.

## 🚀 Project Overview

The code solves the eigenvalue problem:
$$-\frac{1}{2} \frac{d^2 u}{dr^2} + W_{eff}(r) u(r) = E u(r)$$
Where $W_{eff}(r)$ includes the physical potential and the centrifugal barrier.

### Key Features Implemented:
1.  **3D Harmonic Oscillator:** Comparison between Finite Difference and Numerov methods.
2.  **Hydrogen Atom:** Convergence analysis and error scaling.
3.  **Singularity Correction:** Handling the $1/r$ singularity at the origin using first and second-order corrections to the Numerov scheme.
4.  **Box Radius Analysis:** Investigation of the trade-off between confinement error and grid resolution.
5.  **Multi-Electron Atoms (Hartree-Fock):**
    * **Helium ($Z=2$):** Implementation of the SCF loop, calculating electron density $\rho(r)$ and direct repulsion potential $V_{ee}(r)$.
    * **Oxygen ($Z=8$):** Handling shell structure ($1s, 2s, 2p$) and demonstrating the lifting of degeneracy between $2s$ and $2p$ orbitals due to penetration and shielding effects.

## 🛠️ Methods
* **Finite Difference Method:** Matrix diagonalization.
* **Numerov Method:** High-precision $\mathcal{O}(h^4)$ integration scheme.
* **Self-Consistent Field (SCF):** Iterative algorithm to solve the non-linear Hartree equation for multi-electron systems.

## 📂 Files
* `Numerical Solutions of the Radial Schrodinger Equation.py`: The main script containing all solver functions and task executions.
* `Numerical Solutions of the Radial Schrodinger Equation.pdf`: Detailed scientific report explaining the physics, methodology, and results.

## 📊 Results
The code generates visualizations for:
* Wavefunctions ($u(r)$) comparisons.
* Log-Log convergence plots.
* SCF convergence history (Density and Potential).
* Effective Nuclear Charge ($Z_{eff}$) profiles.

## 💻 How to Run
Prerequisites:
```bash
pip install numpy scipy matplotlib
```

Run:
```bash
python "Numerical Solutions of the Radial Schrodinger Equation.py"
```
