# f-TRFu: feasible trust-region funnel method for grey-box optimisation

This repository reproduces every result of **Chapter 6** of the thesis: Tables
6.1, 6.2 and 6.3 and Figures 6.2 and 6.3. They compare four methods:

| Method | What it is | Implementation |
|---|---|---|
| **f-TRFu** | the feasible trust-region funnel method, Algorithm 6.1 | Pyomo's trust-region solver, `globalization_strategy='funnel'` |
| **s-TRFi** | the simplified trust-region filter method of Yoshio and Biegler | Pyomo's trust-region solver, `globalization_strategy='filter'` |
| **c-TRFu** | the classical trust-region funnel method (Chapter 5) | TRF-Solver in `trf_solver/`, `algorithm_type=0`, funnel |
| **c-TRFi** | the classical trust-region filter method (Chapter 3) | TRF-Solver in `trf_solver/`, `algorithm_type=0`, filter |

f-TRFu is part of Pyomo from release 6.10.1 (`pyomo.contrib.trustregion`), so no
copy of it is kept here. It builds the corrected Taylor surrogate of Section 6.1
from one value and one exact derivative per truth model, and poses the trust
region on the decision variables only. The classical methods are run with linear,
quadratic or Taylor surrogates.

## Layout

```
problems/
  ftrfu.py             the problems for f-TRFu and s-TRFi (analytic gradients)
  classical.py         the problems for c-TRFu and c-TRFi
experiments/
  run_ftrfu.py         f-TRFu and s-TRFi on P1, P2 and the eight benchmark problems
  run_classical.py     c-TRFu and c-TRFi, every run in its own process
  make_tables.py       Tables 6.1, 6.2, 6.3
  make_figures.py      Figures 6.2, 6.3
trf_solver/            TRF-Solver (classical methods), with the AMPL gjh executable
results/reference/     the reported results (see "Reference results")
environment_ftrfu.yml  environment for f-TRFu and s-TRFi (Python 3.12, Pyomo 6.10.1)
environment.yml        environment for c-TRFu and c-TRFi (Python 3.8, Pyomo 6.2)
```

## Requirements

Two environments are used, because the classical code was written for Pyomo 6.2
and f-TRFu needs Pyomo 6.10.1:

```
conda env create -f environment_ftrfu.yml      # creates "ftrfu"
conda env create -f environment.yml            # creates "trf-classical"
```

Both need the IPOPT executable (the reported runs used IPOPT 3.14.13 with MUMPS,
built with MinGW-w64 under MSYS2). On Linux and macOS the environment files
install it; on Windows the conda-forge package provides only the library, so put
an `ipopt.exe` on the `PATH`, for example from the COIN-OR Ipopt releases on
GitHub.

The classical methods obtain derivatives from AMPL's `gjh`. `trf_solver/gjh.exe`
is the Windows build used for the reported runs and is used first; on other
platforms, or if it is removed, the solver looks on the `PATH` and otherwise
downloads gjh from AMPL's package index on first use.

All commands below are run from the repository root.

## Reproducing Chapter 6

### 1. f-TRFu and s-TRFi (Table 6.1, f-TRFu and s-TRFi columns of Tables 6.2 and 6.3, Figures 6.2 and 6.3)

```
conda activate ftrfu
python experiments/run_ftrfu.py
python experiments/make_figures.py
```

`run_ftrfu.py` solves P1, P2 and the eight problems of Tables 6.2 and 6.3 with
both methods (20 runs, about three minutes, most of it s-TRFi on P1 and
Himmelblau). It writes `results/ftrfu_runs.csv` and
the solver output of every run to `results/logs/ftrfu/`. `make_figures.py` draws
Figures 6.2 (P1) and 6.3 (P2) from those logs into `results/figures/`.

### 2. c-TRFu and c-TRFi (classical rows of Table 6.1, classical columns of Tables 6.2 and 6.3)

```
conda activate trf-classical
python experiments/run_classical.py
```

36 runs: P1 and P2 with the linear surrogate under both strategies (Table 6.1);
the eight problems with the Taylor surrogate under both strategies (Table 6.2);
the eight problems with the linear and the quadratic surrogate under the funnel
(Table 6.3). Each run is made in a fresh process. Results are appended to
`results/classical_runs.csv` and runs already recorded are skipped, so an
interrupted sweep can be restarted; the solver output goes to
`results/logs/classical/`. The sweep takes a few minutes; most of it is the runs
that reach the iteration limit.

### 3. Tables

```
python experiments/make_tables.py
```

Any of the two environments will do. It writes `results/tables/table_6_1.csv` to
`table_6_3.csv`. To build them from the reported results instead of new runs:

```
python experiments/make_tables.py --results results/reference
python experiments/make_figures.py --logs results/reference/logs/ftrfu
```

## Settings of the runs

**Initial trust-region radius.** Every method starts from the same radius on a
given problem, as stated in Tables 6.2 and 6.3:

| Problem | Initial radius Δ₀ |
|---|---|
| P1, P2 | 1 |
| Eason, Wing Weight, Loeppky, Welded Beam | 1 |
| Six_BB_Chain, High_dim_sep_10 | 1 |
| Himmelblau, Colville | 500 |

On Himmelblau and Colville the classical methods were given a larger radius,
500, at which they converge with the Taylor surrogate under both strategies; the
same radius is used for all four methods on these two problems.

**f-TRFu and s-TRFi.** All other parameters are the defaults of Pyomo's
trust-region solver (Table A.1, column Ch. 6). The decision variables of each
problem are listed in `problems/ftrfu.py`.

**c-TRFu and c-TRFi.** Iteration limit 500. Each problem keeps the solver
settings stored with it in `problems/classical.py`: the expansion factor
γ_e = 10 for Colville and Welded Beam and 15 for Himmelblau, Loeppky and Wing
Weight, and the criticality threshold 0.1 for Colville; the other problems use
the solver's defaults.

**Formulations.** Each method is run on the formulation used for the reported
results:

- P1 for f-TRFu and s-TRFi is Problem B.1.1 (one truth model,
  t(w) = w³ + w² − w, both variables unbounded); for the classical methods the
  truth model is split into w³ and w² and both variables lie in [0, 1].
- P2 for f-TRFu and s-TRFi is Problem B.1.3 (w₁ free, w₂ ≥ −2); for the
  classical methods both variables lie in [−2, 10].
- Eason has no bounds for f-TRFu and s-TRFi and all variables in [−10, 10]
  for the classical methods.

On P1 and P2 the classical methods stop in the compatibility step at the first
iteration (Table 6.1, “Fail”).

## What is counted

- **Iterations** (Table 6.1): the index of the last iteration, the starting point
  being iteration 0.
- **Evaluations** (Tables 6.2 and 6.3): the number of calls to the truth-model
  functions, repeated points included. For f-TRFu and s-TRFi every call returns
  a value and the exact derivative is supplied by a separate function, whose
  calls are recorded as `grad_evals`.
- A classical run that reaches the iteration limit is reported as `Max_iter`, and
  one that stops in the compatibility step as `Fail`.

The evaluation columns of Table 6.1 were counted differently and are not
regenerated; the iteration counts of Table 6.1 are.

## Reference results

`results/reference/` holds the reported results:

- `ftrfu_runs.csv`, `classical_runs.csv`: all 56 runs;
- `logs/ftrfu/`: the solver output of the f-TRFu and s-TRFi runs, from which
  Figures 6.2 and 6.3 are drawn;
- `tables/`: Tables 6.1 to 6.3 as built by `make_tables.py`.

In the reference environment (Windows 11, IPOPT 3.14.13, the bundled gjh)
`run_ftrfu.py` and `run_classical.py` reproduce every row of these files, and the
P1 and P2 trajectories agree, iteration by iteration, with the runs from which
the thesis figures were drawn.

## Citation

Release 1.0.0 of this repository accompanies G. Hameed, *Rigorous Trust-Region Algorithms for Surrogate-based Grey-box Optimisation*, PhD thesis, University of Surrey, 2026; publications that use later releases are listed here as they appear. Please cite the release you used with the metadata in `CITATION.cff`; each release has its own Zenodo DOI, shown on the repository page.

## Licence and attribution

The code in this repository is released under the MIT licence (`LICENSE`), except the third-party components listed below, which keep their own terms.

`trf_solver/` is TRF-Solver (MIT licence, see `trf_solver/LICENSE`); its files
`TRF.py`, `PyomoInterface.py`, `Logger.py`, `readgjh.py`, `GeometryGenerator.py`
and `cache.py` derive from the trust-region code distributed with Pyomo (3-clause
BSD licence, © 2017 National Technology and Engineering Solutions of Sandia, LLC)
and retain its notice. `trf_solver/gjh.exe` is AMPL's gjh, distributed by AMPL
Optimization Inc.
