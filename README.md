# Horizon Radiation Tomography

Simulating and reconstructing simplified Hawking/Unruh radiation through
Gaussian quantum-state tomography.

## Overview

Horizons (black hole event horizons) are predicted to generate thermal particle radiation (Hawking and
Unruh radiation) from correlations between an observable mode and a
partner mode hidden behind the horizon. This project builds a toy model of
that process. It finds out, given only finite, noisy measurements of the accessible mode, how much can be
recovered about the underlying quantum state?

The horizon is modeled as a two-mode squeezed Gaussian state: one mode
represents radiation an outside observer can measure, the other an
inaccessible partner mode beyond the horizon. Tracing out the hidden mode
leaves the observer with a thermal mixed state - this is the toy-model
analogue of Hawking radiation appearing thermal despite arising from a
globally pure, entangled state.

The pipeline:

1. Generate the exact covariance matrix of the accessible mode from a squeezing parameter `r`.
2. Simulate realistic, noisy quadrature measurements of that mode under
   different detection schemes.
3. Reconstruct the covariance matrix from finite samples.
4. Estimate physical quantities - squeezing parameter, particle number,
   entanglement entropy - from the reconstruction, and quantify the
   recovery error against ground truth.
5. Sweep over squeezing strength, sample size, noise level, and detection
   scheme to characterize how estimation accuracy degrades under realistic
   experimental constraints.

## Physics background

A two-mode squeezed vacuum state has covariance matrix (vacuum
normalized to identity):

```text
V(r) = [ cosh(2r)·I₂      sinh(2r)·Z  ]
       [ sinh(2r)·Z       cosh(2r)·I₂ ]
```

Tracing out the hidden mode gives the reduced state of the accessible
mode, `V_A = cosh(2r)·I₂` - a thermal state whose properties are fully
determined by its symplectic eigenvalue `ν = sqrt(det(V_A))`:

- **Particle number:** `N = (Tr(V_A) − 2) / 4`
- **Entanglement entropy:** `S = g(ν)`, the standard thermal-state entropyfunction of a Gaussian mode

Two detection schemes are modeled:

- **Joint** - both quadratures measured exactly every shot. Not physically
  realizable (X and P do not commute); used as the idealized benchmark.
- **Heterodyne** - both quadratures measured every shot, but with an
  extra unit of vacuum noise added - a fundamental measurement penalty,
  not a technical limitation. This is the realistic scheme used for the
  rest of the pipeline, since it yields the 2D point cloud the topological
  analysis needs.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The project
pins Python 3.12 (`.python-version`) and exact package versions
(`uv.lock`), so everyone gets the same environment. After installing uv:

```bash
uv sync
```

This downloads Python 3.12 if needed and creates `.venv` with all
dependencies, including the Jupyter kernel for the notebooks. In VS Code,
select `.venv` as the interpreter / notebook kernel.

To add a package, use `uv add <package>` (not `pip install`), then commit
`pyproject.toml` and `uv.lock`, and refresh `requirements.txt` with
`uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt`.

Without uv, `requirements.txt` holds the same pinned versions:

```bash
python -m venv .venv   # Python 3.12
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

## Running

```bash
uv run streamlit run app/app.py     # interactive app
uv run python src/make_figures.py   # figures for Section 3.1, written to figures/
```

`python app/app.py` also works and relaunches itself under Streamlit.

The Bayesian inference notebooks are in `src/`:
`bayesian_inference.ipynb` (original) and
`bayesian_inference_two_schemes.ipynb` (updated for the joint + heterodyne
simulator). The latter also infers `r` from the Betti curves alone
(`src/topological_inference.py`: a synthetic likelihood trained on simulated
datasets) and compares it with the raw-quadrature posterior and the
Cramér–Rao bound. A full run takes a few minutes, mostly for ripser.

## Code layout

| File | Role |
| --- | --- |
| `src/simulate.py` | Two-mode squeezed state, measurement schemes, detector noise, calibration (Section 3.1) |
| `src/estimate.py`, `src/metrics.py` | Covariance and squeezing estimators, Cramér–Rao bounds |
| `src/homology.py` | Persistence diagrams and Betti curves (Section 3.2) |
| `src/topological_inference.py` | Bayesian inference of `r` from Betti curves (synthetic likelihood) |
| `src/bayesian_inference*.ipynb` | Bayesian inference with numpyro (Section 4) |
| `src/make_figures.py` | Figures for Section 3.1 |
| `app/app.py` | Interactive Streamlit demo |
