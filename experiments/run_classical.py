# -*- coding: utf-8 -*-
"""Run the classical methods of Chapter 6, c-TRFu (funnel) and c-TRFi (filter).

Runs, each in its own process (TRF-Solver keeps state between solves in one process):

  Table 6.1  P1, P2                  linear surrogate   filter and funnel
  Table 6.2  the eight problems      Taylor surrogate   filter and funnel
  Table 6.3  the eight problems      linear, quadratic  funnel

Every run starts from the problem's initial trust-region radius (1, or 500 for
Himmelblau and Colville) with an iteration limit of 500. Results are appended to
results/classical_runs.csv; runs already recorded there are skipped, so an
interrupted sweep can be restarted. Solver output goes to results/logs/classical/.

    python experiments/run_classical.py                 # all 36 runs
    python experiments/run_classical.py --problems P1 Eason
"""
import argparse
import contextlib
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'classical_runs.csv')
LOGS = os.path.join(ROOT, 'results', 'logs', 'classical')

RADIUS = {'Himmelblau': 500.0, 'Colville': 500.0}       # 1.0 otherwise
MAX_IT = 500
EIGHT = ['Eason', 'Wing Weight', 'Loeppky', 'Welded Beam', 'Himmelblau', 'Colville',
         'Six_BB_Chain', 'High_dim_sep_10']
FIELDS = ['problem', 'method', 'surrogate', 'globalization', 'status', 'exit_status',
          'iterations', 'bb_evals', 'final_obj', 'trust_radius', 'error']
METHOD = {'funnel': 'c-TRFu', 'filter': 'c-TRFi'}


def runs(problems):
    out = []
    for p in ('P1', 'P2'):
        out += [(p, 'linear', g) for g in ('filter', 'funnel')]
    for p in EIGHT:
        out += [(p, 'ts', g) for g in ('filter', 'funnel')]
        out += [(p, s, 'funnel') for s in ('linear', 'quadratic')]
    return [r for r in out if not problems or r[0] in problems]


def solve_one(problem, surrogate, glob):
    """One run in this process; prints a single tab-separated RESULT line."""
    sys.path.insert(0, os.path.join(ROOT, 'trf_solver'))
    sys.path.insert(0, os.path.join(ROOT, 'problems'))
    from pyomo.environ import value
    from TRF import TrustRegionSolver
    from classical import PROBLEMS

    model, eflist, base = PROBLEMS[problem]()
    kwargs = dict(base)
    kwargs.pop('sample_radius', None)
    kwargs.update(max_it=MAX_IT, algorithm_type=0,
                  globalization_strategy=0 if glob == 'filter' else 1,
                  reduced_model_type=surrogate,
                  trust_radius=RADIUS.get(problem, 1.0))
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, '%s_%s_%s.log' % (problem.replace(' ', '_'), surrogate, glob))
    row = dict(problem=problem, method=METHOD[glob], surrogate=surrogate, globalization=glob,
               trust_radius=kwargs['trust_radius'], status='OK', error='')
    with open(log, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
        try:
            res = TrustRegionSolver(**kwargs).solve(model, eflist, problem_name=problem) or {}
            row.update(exit_status=res.get('status', ''), iterations=res.get('iterations'),
                       bb_evals=res.get('bb_evals'),
                       final_obj=res.get('final_obj', value(model.obj)))
        except Exception as e:
            row.update(status='FAIL', error='%s: %s' % (type(e).__name__, str(e).strip()))
    print('RESULT\t' + '\t'.join(str(row.get(k, '')) for k in FIELDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', nargs='*')
    ap.add_argument('--one', nargs=3, metavar=('PROBLEM', 'SURROGATE', 'GLOB'),
                    help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        return solve_one(*a.one)

    done = set()
    if os.path.exists(OUT):
        done = {(r['problem'], r['surrogate'], r['globalization'])
                for r in csv.DictReader(open(OUT, encoding='utf-8'))}
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(FIELDS)
    for p, s, g in runs(a.problems):
        if (p, s, g) in done:
            continue
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--one', p, s, g],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('RESULT\t')]
        vals = line[0].split('\t')[1:] if line else \
            [p, METHOD[g], s, g, 'FAIL', '', '', '', '', RADIUS.get(p, 1.0), 'process crashed']
        with open(OUT, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(vals)
        print('%-16s %-9s %-6s %s' % (p, s, g, ' '.join(vals[4:9])))


if __name__ == '__main__':
    main()
