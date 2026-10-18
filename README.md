[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23444569.svg)](https://doi.org/10.5281/zenodo.23444569)

# Depolarization beyond energy shifts: pathway factors, multisubband coupling and dynamic screening in resonant intersubband second-harmonic generation

**Author:** Gracen Ownby (University of Oklahoma)  
**Date:** October 2026  
**Manuscript PDF:** [`NL05_paper.pdf`](./NL05_paper.pdf)  
**LaTeX Source:** [`paper_nl05/`](./paper_nl05/)  
**Study Code & JSON Proofs:** [`studies/nl05/`](./studies/nl05/)  

## Overview
This repository is the permanent, self-contained mathematical and computational archive for **NL05**: *Depolarization beyond energy shifts: pathway factors, multisubband coupling and dynamic screening in resonant intersubband second-harmonic generation*. It includes:
1. The complete reviewed paper (`NL05_paper.pdf`) and its LaTeX source (`paper_nl05/`).
2. The complete `qwkit` (v0.7.0) Python package (`qwkit/`) and its 69 automated validation tests (`tests/`), which verify analytic limits, BenDaniel-Duke finite-well convergence, exact mirror-symmetry oddness $\chi^{(2)}(-F) = -\chi^{(2)}(F)$ to $10^{-11}$, two-band Kane orthogonality ($10^{-13}$), energy-conserving RK4 coupled-mode propagation, implicit-differentiation tangents, and RPA sum rules.
3. The study scripts and precomputed JSON datasets (`studies/nl05/`) that generate every table, bound, and figure in the manuscript.

## Quick Verification (Mathematical & Numerical Proof)

Install `qwkit` in editable mode (requires Python >= 3.10, `numpy`, `scipy`, `matplotlib`):
```bash
pip install -e .
```

Run the single-command verification suite (runs all 69 unit tests, verifies the study numbers, and regenerates the manuscript figures):
```bash
python prove_paper.py
```

## License
Code is released under the MIT License (`LICENSE`). The manuscript and figures are (c) 2026 Gracen Ownby.
