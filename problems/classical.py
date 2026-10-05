# -*- coding: utf-8 -*-
"""Problems for the classical methods c-TRFu and c-TRFi (TRF-Solver, algorithm_type 0).

Each builder returns (model, eflist, base_kwargs). The builders are copied unchanged
from TRF-Solver's benchmark suite (benchmarks/paper3_numerical_benchmarks.py,
BenchmarkProblems.py, extra_benchmarks.py); base_kwargs are each problem's stored
solver settings, of which experiments/run_classical.py overrides only the initial
trust-region radius and the iteration limit.

P1 and P2 here are the formulations of the classical runs of Table 6.1: P1 with the
truth model split into a^3 and a^2 and both variables in [0, 1]; P2 with both
variables in [-2, 10].
"""
import math
from functools import partial

from pyomo.environ import *

_KW = dict(solver='ipopt')

def biegler():
    """Biegler problem: 2 vars, 2 black boxes (a^3, a^2), 1 equality
    constraint."""
    m = ConcreteModel()
    m.x = Var(range(2), domain=Reals, bounds=(0, 1))
    m.x[0] = 0
    m.x[1] = 0

    def blackbox1(a):
        return a**3
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a):
        return a**2
    bb2 = ExternalFunction(blackbox2)

    m.obj = Objective(expr=(m.x[0])**2 + (m.x[1])**2)

    m.c1 = Constraint(expr=bb1(m.x[0]) + bb2(m.x[0]) - m.x[1] + 1 == 0)

    return m, [bb1, bb2], dict(solver='ipopt', max_it=500)


def yoshio():
    """Yoshio problem: 2 vars, 1 black box (a^2 + b^2), 1 equality
    constraint."""
    m = ConcreteModel()
    m.x1 = Var(bounds=(-2.0, 10), initialize=0)
    m.x2 = Var(bounds=(-2.0, 10), initialize=0)

    def blackbox(a, b):
        return a**2 + b**2
    bb = ExternalFunction(blackbox)

    m.obj = Objective(expr=(m.x1 - 1)**2 + (m.x2 - 3)**2 + bb(m.x1, m.x2)**2)

    m.c1 = Constraint(expr=2 * m.x1 + m.x2 + 10.0 == bb(m.x1, m.x2))

    return m, [bb], dict(solver='ipopt', max_it=250)


def eason():
    """Eason problem: 5 vars, 1 black box (sin), 2 equality constraints."""
    m = ConcreteModel()
    m.x = Var(range(5), domain=Reals, initialize=2.0, bounds=(-10, 10))
    m.x[4] = 1.0

    def blackbox(a, b):
        return sin(a - b)
    bb = ExternalFunction(blackbox)

    m.obj = Objective(
        expr=(m.x[0] - 1.0)**2 + (m.x[0] - m.x[1])**2 + (m.x[2] - 1.0)**2
             + (m.x[3] - 1.0)**4 + (m.x[4] - 1.0)**6)

    m.c1 = Constraint(
        expr=m.x[3] * m.x[0]**2 + bb(m.x[3], m.x[4]) == 2 * sqrt(2.0))
    m.c2 = Constraint(
        expr=m.x[2]**4 * m.x[1]**2 + m.x[1] == 8 + sqrt(2.0))

    return m, [bb], dict(solver='ipopt')


def build_A1_colville():
    """Model A1 - Colville Function (4 black boxes)."""
    m = ConcreteModel()
    m.x1 = Var(initialize=78, bounds=(78, 102))
    m.x2 = Var(initialize=33, bounds=(33, 45))
    m.x3 = Var(initialize=30, bounds=(27, 45))
    m.x4 = Var(initialize=45, bounds=(27, 45))
    m.x5 = Var(initialize=37, bounds=(27, 45))

    def blackbox1(a, b):
        return 0.8357 * a * b
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b, c):
        return 0.00002584 * a * b - 0.00006663 * c * b
    bb2 = ExternalFunction(blackbox2)

    def blackbox3(a, b, c):
        return 2275.1327 * ((a * b) ** (-1)) - 0.2668 * c * ((b) ** (-1))
    bb3 = ExternalFunction(blackbox3)

    def blackbox4(a, b, c):
        return 1330.3294 * ((a * b) ** (-1)) - 0.42 * c * ((b) ** (-1))
    bb4 = ExternalFunction(blackbox4)

    m.c1 = Constraint(expr=bb2(m.x3, m.x5, m.x2) - 0.0000734 * m.x1 * m.x4 - 1 <= 0)
    m.c2 = Constraint(expr=0.000853007 * m.x2 * m.x5 + 0.00009395 * m.x1 * m.x4 - 0.00033085 * m.x3 * m.x5 - 1 <= 0)
    m.c3 = Constraint(expr=bb4(m.x2, m.x5, m.x1) - 0.30586 * ((m.x2 * m.x5) ** (-1)) * m.x3 ** 2 - 1 <= 0)
    m.c4 = Constraint(expr=0.00024186 * m.x2 * m.x5 + 0.00010159 * m.x1 * m.x2 + 0.00007379 * m.x3 ** 2 - 1 <= 0)
    m.c5 = Constraint(expr=bb3(m.x3, m.x5, m.x1) - 0.40584 * ((m.x5) ** (-1)) * m.x4 - 1 <= 0)
    m.c6 = Constraint(expr=0.00029955 * m.x3 * m.x5 + 0.00007992 * m.x1 * m.x2 + 0.00012157 * m.x3 * m.x4 - 1 <= 0)

    m.obj = Objective(expr=5.3578 * m.x3 ** 2 + bb1(m.x1, m.x5) + 37.2392 * m.x1)

    # algorithm_type and globalization_strategy are set by the driver.
    base_kwargs = dict(solver='ipopt', max_it=25000, trust_radius=1000, sample_radius=100,
                       gamma_e=10, criticality_check=0.1)
    return m, [bb1, bb2, bb3, bb4], base_kwargs


def build_A2_himmelblau():
    """Model A2 - Himmelblau's Problem."""
    model = ConcreteModel()
    model.x1 = Var(bounds=(78, 102), initialize=100)
    model.x2 = Var(bounds=(33, 45), initialize=40)
    model.x3 = Var(bounds=(27, 45), initialize=40)
    model.x4 = Var(bounds=(27, 45), initialize=35)
    model.x5 = Var(bounds=(27, 45), initialize=30)

    model.g1 = Var(bounds=(0, 92), initialize=30)
    model.g2 = Var(bounds=(90, 110), initialize=100)
    model.g3 = Var(bounds=(20, 25), initialize=20)

    def blackbox1(a):
        return a ** 2
    bb1 = ExternalFunction(blackbox1)

    def blackbox2(a, b):
        return a * b
    bb2 = ExternalFunction(blackbox2)

    model.obj = Objective(expr=5.3578547 * bb1(model.x3) + 0.8356891 * model.x1 * model.x5
                          + 37.2932239 * model.x1 - 40792.141, sense=minimize)

    model.c1 = Constraint(expr=model.g1 == 85.334407 + 0.0056858 * bb2(model.x2, model.x5)
                          + 0.00026 * model.x1 * model.x4 - 0.0022053 * model.x3 * model.x5)
    model.c2 = Constraint(expr=model.g2 == 80.51249 + 0.0071317 * bb2(model.x2, model.x5)
                          + 0.0029955 * model.x1 * model.x2 - 0.0021813 * (model.x3 ** 2))
    model.c3 = Constraint(expr=model.g3 == 9.300961 + 0.0047026 * model.x3 * model.x5
                          + 0.0012547 * model.x1 * model.x3 - 0.0019085 * model.x3 * model.x4)

    base_kwargs = dict(solver='ipopt', max_it=1000, trust_radius=10000, sample_radius=1000,
                       gamma_e=15)
    return model, [bb1, bb2], base_kwargs


def build_A3_loeppky():
    """Model A3 - Loeppky Function."""
    m = ConcreteModel()
    m.I = RangeSet(1, 7)
    m.x = Var(m.I, bounds=(0, 1), initialize=0.5)

    def blackbox1(a, b, c):
        return 3 * a * b + 2.2 * a * c
    bb1 = ExternalFunction(blackbox1)

    m.obj = Objective(
        expr=6 * m.x[1] + 4 * m.x[2] + 5.5 * m.x[3] + bb1(m.x[1], m.x[2], m.x[3])
        + 1.4 * m.x[2] * m.x[3] + m.x[4] + 0.5 * m.x[5] + 0.2 * m.x[6] + 0.1 * m.x[7],
        sense=minimize,
    )

    base_kwargs = dict(solver='ipopt', max_it=1000, trust_radius=1, sample_radius=0.1,
                       gamma_e=15)
    return m, [bb1], base_kwargs


def build_A4_wing_weight():
    """Model A4 - Wing Weight Function."""
    m = ConcreteModel()
    m.Sw = Var(bounds=(150.0, 200.0), initialize=175.0)
    m.Wfw = Var(bounds=(220.0, 300.0), initialize=260.0)
    m.A = Var(bounds=(6.0, 10.0), initialize=8.0)
    m.Lam = Var(bounds=(-10.0, 10.0), initialize=0.0)
    m.q = Var(bounds=(16.0, 45.0), initialize=30.0)
    m.lam = Var(bounds=(0.5, 1.0), initialize=0.75)
    m.tc = Var(bounds=(0.08, 0.18), initialize=0.13)
    m.Nz = Var(bounds=(2.5, 6.0), initialize=4.0)
    m.Wdg = Var(bounds=(1700.0, 2500.0), initialize=2100.0)
    m.Wp = Var(bounds=(0.025, 0.08), initialize=0.05)

    def blackbox1(a, b):
        return a * b
    bb1 = ExternalFunction(blackbox1)

    m.obj = Objective(
        expr=0.036
        * m.Sw ** 0.758
        * m.Wfw ** 0.0035
        * (m.A / (cos(m.Lam * (22 / 7) / 180.0) ** 2)) ** 0.6
        * m.q ** 0.006
        * m.lam ** 0.04
        * ((100.0 * m.tc / cos(m.Lam * (22 / 7) / 180.0)) ** -0.3)
        * (m.Nz * m.Wdg) ** 0.49
        + bb1(m.Sw, m.Wp),
        sense=minimize,
    )

    base_kwargs = dict(solver='ipopt', max_it=10000, trust_radius=1, sample_radius=0.1,
                       gamma_e=15)
    return m, [bb1], base_kwargs


def build_A5_welded_beam():
    """Model A5 - Welded Beam."""
    model = ConcreteModel()
    model.c1 = Param(default=0.10471)
    model.c2 = Param(default=0.04811)
    model.P = Param(default=6000.0)
    model.L = Param(default=14.0)
    model.E = Param(default=30e6)
    model.G = Param(default=12e6)
    model.tau_max = Param(default=13600.0)
    model.sigma_max = Param(default=30000.0)
    model.delta_max = Param(default=0.25)

    model.x1 = Var(bounds=(0.125, 5), initialize=1.0)
    model.x2 = Var(bounds=(0.1, 10), initialize=5.0)
    model.x3 = Var(bounds=(0.1, 10), initialize=5.0)
    model.x4 = Var(bounds=(0.1, 5), initialize=1.0)

    def blackbox(a, b, c, d):
        return (1 + 0.10471) * a ** 2 * b + 0.04811 * c * d * (14 + b)
    bb = ExternalFunction(blackbox)

    model.obj = Objective(expr=bb(model.x1, model.x2, model.x3, model.x4), sense=minimize)

    def tau(model):
        R = sqrt((model.x2 ** 2) / 4 + ((model.x1 + model.x3) / 2) ** 2)
        M = model.P * (model.L + model.x2 / 2)
        J = 2 * (sqrt(2) * model.x1 * model.x2) * ((model.x2 ** 2) / 12 + ((model.x1 + model.x3) / 2) ** 2)
        t1 = model.P / (sqrt(2) * model.x1 * model.x2)
        t2 = M * R / J
        return sqrt(t1 ** 2 + 2 * t1 * t2 * model.x2 / (2 * R) + t2 ** 2)
    model.tau_expr = Expression(rule=tau)

    def sigma(model):
        return (6 * model.P * model.L) / (model.x4 * model.x3 ** 2)
    model.sigma_expr = Expression(rule=sigma)

    def delta(model):
        return (4 * model.P * model.L ** 3) / (model.E * model.x3 ** 3 * model.x4)
    model.delta_expr = Expression(rule=delta)

    def Pc(model):
        return ((4.013 * model.E * sqrt(model.x3 ** 2 * model.x4 ** 6 / 36)) / (model.L ** 2)) \
            * (1 - (model.x3 / (2 * model.L)) * sqrt(model.E / (4 * model.G)))
    model.pc_expr = Expression(rule=Pc)

    model.g1 = Constraint(expr=model.tau_expr <= model.tau_max)
    model.g2 = Constraint(expr=model.sigma_expr <= model.sigma_max)
    model.g3 = Constraint(expr=model.x1 - model.x4 <= 0)
    model.g4 = Constraint(expr=model.c1 * model.x1 ** 2 * model.x2 + model.c2 * model.x3 * model.x4 * (model.L + model.x2) - 5 <= 0)
    model.g5 = Constraint(expr=model.delta_expr <= model.delta_max)
    model.g6 = Constraint(expr=model.P - model.pc_expr <= 0)

    base_kwargs = dict(solver='ipopt', max_it=50, gamma_e=10)
    return model, [bb], base_kwargs


def six_bb_chain():
    """J=6 coupled BB sub-models sharing inputs in a chain."""
    m = ConcreteModel()
    m.x = Var(range(6), initialize=1.0, bounds=(0.2, 3.0))
    m.y = Var(range(6), initialize=1.0, bounds=(-50, 50))

    coeff = [1.0, 0.5, 2.0, 0.8, 1.5, 0.3]
    targets = [1.2, 0.9, 2.5, 1.1, 1.8, 0.6]

    def make_bb(i):
        c = coeff[i]
        def bb(a, b):
            return c * (a * b + 0.3 * a**2)
        return bb
    bbs = [ExternalFunction(make_bb(i)) for i in range(6)]

    def chain_rule(m, i):
        return m.y[i] == bbs[i](m.x[i], m.x[(i + 1) % 6])
    m.chain = Constraint(range(6), rule=chain_rule)
    m.obj = Objective(expr=sum((m.y[i] - targets[i])**2 for i in range(6))
                      + 0.1 * sum((m.x[i] - 1.0)**2 for i in range(6)),
                      sense=minimize)
    return m, list(bbs), dict(_KW)


def highdim_separable(n):
    """n vars, n smooth one-input BBs; well-conditioned separable objective."""
    m = ConcreteModel()
    m.x = Var(range(n), bounds=(-5.0, 5.0), initialize=2.0)
    m.y = Var(range(n), bounds=(-50.0, 200.0), initialize=0.0)
    targets = [1.0 + (i % 3) * 0.5 for i in range(n)]

    def make_bb(i):
        def bb(a):
            return math.exp(0.1 * a) - 1.0 + 0.5 * a**2
        return bb
    bbs = [ExternalFunction(make_bb(i)) for i in range(n)]

    def link(m, i):
        return m.y[i] == bbs[i](m.x[i])
    m.link = Constraint(range(n), rule=link)
    m.obj = Objective(expr=sum((m.x[i] - targets[i])**2 for i in range(n))
                      + 0.1 * sum(m.y[i] for i in range(n)), sense=minimize)
    return m, list(bbs), dict(_KW)


# label: builder
PROBLEMS = {
    'P1': biegler,
    'P2': yoshio,
    'Eason': eason,
    'Wing Weight': build_A4_wing_weight,
    'Loeppky': build_A3_loeppky,
    'Welded Beam': build_A5_welded_beam,
    'Himmelblau': build_A2_himmelblau,
    'Colville': build_A1_colville,
    'Six_BB_Chain': six_bb_chain,
    'High_dim_sep_10': partial(highdim_separable, 10),
}
