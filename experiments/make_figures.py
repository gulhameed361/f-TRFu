# -*- coding: utf-8 -*-
"""Figures 6.2 and 6.3: convergence of (a) f-TRFu and (b) s-TRFi on P1 and P2.

Per iteration, from the solver output written by run_ftrfu.py: infeasibility,
trust-region radius and step norm (left, logarithmic axis) and the objective
(right axis). The step norm is zero at iteration 0 and is not drawn on the
logarithmic axis.

    python experiments/make_figures.py [--logs results/reference/logs/ftrfu]

Writes results/figures/figure_6_2.png/.svg and figure_6_3.png/.svg.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'results', 'logs', 'ftrfu')
OUT = os.path.join(ROOT, 'results', 'figures')
RECORD = re.compile(r'Iteration (\d+) \*+\s*objectiveValue = (\S+)\s*feasibility = (\S+)'
                    r'\s*trustRadius = (\S+)\s*stepNorm = (\S+)')

# figure: problem, left-axis floor, objective axis, log iteration axis for s-TRFi
FIGURES = {'figure_6_2': ('P1', 1e-15, 'log', True),
           'figure_6_3': ('P2', 1e-10, (47.8, 49.0), False)}


def trajectory(path):
    with open(path, encoding='utf-8') as f:
        rows = np.array([[float(v) for v in r] for r in RECORD.findall(f.read())])
    it, obj, theta, radius, step = rows.T
    pos = lambda v: np.where(v > 0, v, np.nan)
    return it, obj, pos(theta), radius, pos(step)


def draw(name, problem, floor, obj_axis, log_x, logs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    for ax, (label, method) in zip(axes, (('(a)', 'f-TRFu'), ('(b)', 's-TRFi'))):
        it, obj, theta, radius, step = trajectory(
            os.path.join(logs, '%s_%s.log' % (problem, method)))
        h = [ax.plot(it, theta, '-', color='green', lw=1.5, label='Infeasibility Measure')[0]]
        h.append(ax.plot([], [], ':', color='red', lw=1.5, label='Objective')[0])
        h.append(ax.plot(it, radius, '--', color='gray', lw=1.5, label='Trust-Region Radius')[0])
        h.append(ax.plot(it, step, '-.', color='skyblue', lw=1.5, label='Step Norm')[0])
        ax.set_yscale('log')
        ax.set_ylim(bottom=floor)
        if log_x and method == 's-TRFi':
            ax.set_xscale('log')
            ax.set_xlabel('Iteration (Log Scale)')
        else:
            ax.set_xlabel('Iteration')
        ax.set_ylabel('Infeasibility, Radius, Norm')
        ax.set_title(label, fontweight='bold')
        ax.grid(True, which='major', linestyle=':', alpha=0.6)
        ax2 = ax.twinx()
        ax2.plot(it, obj, ':', color='red', lw=1.5)
        if obj_axis == 'log':
            ax2.set_yscale('log')
        else:
            ax2.set_ylim(*obj_axis)
        ax2.set_ylabel('Objective')
    fig.legend(h, [x.get_label() for x in h], loc='lower center', ncol=4,
               frameon=True, shadow=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'svg'):
        path = os.path.join(OUT, '%s.%s' % (name, ext))
        fig.savefig(path, dpi=300)
        print('wrote', path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', default=LOGS)
    a = ap.parse_args()
    for name, spec in FIGURES.items():
        draw(name, *spec, logs=a.logs)


if __name__ == '__main__':
    main()
