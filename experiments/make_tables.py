# -*- coding: utf-8 -*-
"""Tables 6.1, 6.2 and 6.3 from the runs.

Reads results/ftrfu_runs.csv (run_ftrfu.py) and results/classical_runs.csv
(run_classical.py) and writes results/tables/table_6_1.csv to table_6_3.csv.
A classical run that stops in the compatibility step is reported as Fail, and
one that reaches the iteration limit (500) as Max_iter.

    python experiments/make_tables.py [--results results/reference]
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'tables')
MAX_IT = 500
EIGHT = [('Eason', '(2,1,3)'), ('Wing Weight', '(2,1,8)'), ('Loeppky', '(3,1,4)'),
         ('Welded Beam', '(4,1,0)'), ('Himmelblau', '(3,2,5)'), ('Colville', '(4,4,1)'),
         ('Six_BB_Chain', '(6,6,0)'), ('High_dim_sep_10', '(10,10,0)')]


def load(results):
    f = {(r['problem'], r['method']): r
         for r in csv.DictReader(open(os.path.join(results, 'ftrfu_runs.csv'), encoding='utf-8'))}
    c = {(r['problem'], r['surrogate'], r['globalization']): r
         for r in csv.DictReader(open(os.path.join(results, 'classical_runs.csv'), encoding='utf-8'))}
    return f, c


def classical(r):
    if r['status'] != 'OK':
        return 'Fail'
    if int(float(r['iterations'])) > MAX_IT:
        return 'Max_iter'
    return str(int(float(r['bb_evals'])))


def radius(r):
    return '%g' % float(r['trust_radius'])


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', path)
    for row in [header] + rows:
        print('   ' + ' | '.join(str(x) for x in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    f, c = load(ap.parse_args().results)

    rows = []
    for method, surr in (('f-TRFu', 'Taylor'), ('s-TRFi', 'Taylor')):
        rows.append([method, surr] + [f[(p, method)]['iterations'] for p in ('P1', 'P2')])
    for method, glob in (('c-TRFu', 'funnel'), ('c-TRFi', 'filter')):
        rows.append([method, 'Linear'] + [classical(c[(p, 'linear', glob)]) for p in ('P1', 'P2')])
    write('table_6_1.csv', ['Method', 'Surrogate', 'P1 iterations', 'P2 iterations'], rows)

    rows = []
    for p, size in EIGHT:
        rows.append([p, size, radius(f[(p, 'f-TRFu')]), f[(p, 'f-TRFu')]['bb_evals'],
                     f[(p, 's-TRFi')]['bb_evals'], classical(c[(p, 'ts', 'funnel')]),
                     classical(c[(p, 'ts', 'filter')])])
    write('table_6_2.csv', ['Problem', '(w,y,z)', 'Initial TR radius', 'f-TRFu', 's-TRFi',
                            'c-TRFu', 'c-TRFi'], rows)

    rows = []
    for p, _ in EIGHT:
        rows.append([p, radius(f[(p, 'f-TRFu')]), f[(p, 'f-TRFu')]['bb_evals'],
                     classical(c[(p, 'linear', 'funnel')]),
                     classical(c[(p, 'quadratic', 'funnel')])])
    write('table_6_3.csv', ['Problem', 'Initial TR radius', 'f-TRFu (Taylor)',
                            'c-TRFu (Linear)', 'c-TRFu (Quadratic)'], rows)


if __name__ == '__main__':
    main()
