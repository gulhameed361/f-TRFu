# -*- coding: utf-8 -*-
"""Run f-TRFu (Algorithm 6.1) and s-TRFi on the problems of Chapter 6.

Both are Pyomo's trust-region solver (pyomo.contrib.trustregion, Pyomo 6.10.1):
f-TRFu with globalization_strategy='funnel', s-TRFi with 'filter'. Every problem
starts from its initial trust-region radius (1, or 500 for Himmelblau and Colville);
all other parameters are the solver's defaults (Table A.1, column Ch. 6).

Results go to results/ftrfu_runs.csv and the solver output of every run to
results/logs/ftrfu/<problem>_<method>.log, from which make_figures.py draws
Figures 6.2 and 6.3.

    python experiments/run_ftrfu.py
    python experiments/run_ftrfu.py --problems P1 P2
"""
import argparse
import contextlib
import csv
import io
import os
import re
import sys

import pyomo.environ as pyo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'problems'))
from ftrfu import PROBLEMS  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'ftrfu_runs.csv')
LOGS = os.path.join(ROOT, 'results', 'logs', 'ftrfu')
RADIUS = {'Himmelblau': 500.0, 'Colville': 500.0}       # 1.0 otherwise
MAX_IT = {'P1': 1000, 'P2': 1000, 'Loeppky': 1000, 'Himmelblau': 1000,
          'Wing Weight': 10000}                          # 500 otherwise
METHOD = {'funnel': 'f-TRFu', 'filter': 's-TRFi'}
FIELDS = ['problem', 'method', 'status', 'iterations', 'bb_evals', 'grad_evals',
          'final_obj', 'trust_radius', 'error']


def run(problem, glob):
    build, dof_idx, basis = PROBLEMS[problem]
    m, dof_all, bb_fns, grad_fns = build()
    radius = RADIUS.get(problem, 1.0)
    solver = pyo.SolverFactory('trustregion', maximum_iterations=MAX_IT.get(problem, 500),
                               globalization_strategy=glob, trust_radius=radius,
                               verbose=True)
    kwargs = {'ext_fcn_surrogate_map_rule': basis} if basis else {}
    buf = io.StringIO()
    row = dict(problem=problem, method=METHOD[glob], trust_radius=radius, error='')
    try:
        with contextlib.redirect_stdout(buf):
            solver.solve(m, [dof_all[i] for i in dof_idx], **kwargs)
        its = [int(k) for k in re.findall(r'Iteration (\d+)', buf.getvalue())]
        row.update(status='OK', iterations=max(its) if its else 0,
                   bb_evals=sum(f.calls for f in bb_fns),
                   grad_evals=sum(f.calls for f in grad_fns), final_obj=pyo.value(m.obj))
    except Exception as e:
        row.update(status='FAIL', error='%s: %s' % (type(e).__name__, e))
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, '%s_%s.log' % (problem.replace(' ', '_'), METHOD[glob])),
              'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', nargs='*', default=list(PROBLEMS))
    a = ap.parse_args()
    rows = []
    for p in a.problems:
        for g in ('funnel', 'filter'):
            r = run(p, g)
            rows.append(r)
            print('%-16s %-7s %-4s iterations %-5s evaluations %s' %
                  (p, r['method'], r['status'], r.get('iterations', ''), r.get('bb_evals', '')))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
