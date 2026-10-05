# -*- coding: utf-8 -*-
"""Problems for f-TRFu and s-TRFi (Pyomo's trustregion solver, funnel and filter).

Each builder returns (model, decision_variables, black_box_functions,
gradient_functions). Every black box has an analytic gradient, so the corrected
Taylor surrogate is built from exact sensitivities, and each function counts its
calls. The builders are copied unchanged from the scripts that produced the reported
runs, except P2, which is Problem B.1.3 as reported in Table 6.1 (w1 free, w2 >= -2).

P1 is Problem B.1.1: one truth model t(w) = w^3 + w^2 - w and both variables unbounded.
"""
import math

import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, Var, Reals, RangeSet, ExternalFunction,
                           sin, sqrt, cos, Constraint, Objective, maximize, minimize)

def _make_eason():
    def eason_ext(a, b):
        eason_ext.calls += 1
        return sin(a - b)
    eason_ext.calls = 0

    def eason_grad(args, fixed):
        eason_grad.calls += 1
        a, b = args[:2]
        return [math.cos(a - b), -math.cos(a - b)]
    eason_grad.calls = 0

    m = ConcreteModel()
    m.z = Var(range(3), domain=Reals, initialize=2.0)
    m.x = Var(range(2), initialize=2.0)
    m.x[1] = 1.0
    m.ext_fcn = ExternalFunction(eason_ext, eason_grad)
    m.obj = Objective(expr=(m.z[0] - 1.0) ** 2 + (m.z[0] - m.z[1]) ** 2
                       + (m.z[2] - 1.0) ** 2 + (m.x[0] - 1.0) ** 4
                       + (m.x[1] - 1.0) ** 6)
    m.c1 = Constraint(expr=m.x[0] * m.z[0] ** 2 + m.ext_fcn(m.x[0], m.x[1]) == 2 * sqrt(2.0))
    m.c2 = Constraint(expr=m.z[2] ** 4 * m.z[1] ** 2 + m.z[1] == 8 + sqrt(2.0))
    return m, [m.z[0], m.z[1], m.z[2]], [eason_ext], [eason_grad]


def _make_biegler():
    def bieg_ext(a):
        bieg_ext.calls += 1
        return a ** 3 + a ** 2 - a
    bieg_ext.calls = 0

    def bieg_grad(args, fixed):
        bieg_grad.calls += 1
        a = args[0]
        return [3 * a ** 2 + 2 * a - 1]
    bieg_grad.calls = 0

    m = ConcreteModel()
    m.x = Var(range(2), domain=Reals, initialize=0.0)
    m.ext_fcn = ExternalFunction(bieg_ext, bieg_grad)
    m.obj = Objective(expr=(m.x[0]) ** 2 + (m.x[1]) ** 2)
    m.c1 = Constraint(expr=m.ext_fcn(m.x[0]) + m.x[0] + 1 == m.x[1])
    return m, [m.x[0]], [bieg_ext], [bieg_grad]


def _make_himmelblau():
    def bb1(a):
        bb1.calls += 1
        return a ** 2
    bb1.calls = 0

    def bb1_grad(args, fixed):
        bb1_grad.calls += 1
        return [2 * args[0]]
    bb1_grad.calls = 0

    def bb2(a, b):
        bb2.calls += 1
        return a * b
    bb2.calls = 0

    def bb2_grad(args, fixed):
        bb2_grad.calls += 1
        a, b = args[:2]
        return [b, a]
    bb2_grad.calls = 0

    m = ConcreteModel()
    m.x1 = Var(bounds=(78, 102), initialize=100)
    m.x2 = Var(bounds=(33, 45), initialize=40)
    m.x3 = Var(bounds=(27, 45), initialize=40)
    m.x4 = Var(bounds=(27, 45), initialize=35)
    m.x5 = Var(bounds=(27, 45), initialize=30)
    m.g1 = Var(bounds=(0, 92), initialize=30)
    m.g2 = Var(bounds=(90, 110), initialize=100)
    m.g3 = Var(bounds=(20, 25), initialize=20)
    m.bb1 = ExternalFunction(bb1, bb1_grad)
    m.bb2 = ExternalFunction(bb2, bb2_grad)
    m.obj = Objective(expr=5.3578547 * m.bb1(m.x3) + 0.8356891 * m.x1 * m.x5
                       + 37.2932239 * m.x1 - 40792.141, sense=minimize)
    m.c1 = Constraint(expr=m.g1 == 85.334407 + 0.0056858 * m.bb2(m.x2, m.x5)
                       + 0.00026 * m.x1 * m.x4 - 0.0022053 * m.x3 * m.x5)
    m.c2 = Constraint(expr=m.g2 == 80.51249 + 0.0071317 * m.bb2(m.x2, m.x5)
                       + 0.0029955 * m.x1 * m.x2 - 0.0021813 * (m.x3 ** 2))
    m.c3 = Constraint(expr=m.g3 == 9.300961 + 0.0047026 * m.x3 * m.x5
                       + 0.0012547 * m.x1 * m.x3 - 0.0019085 * m.x3 * m.x4)
    # DOF = 8 vars - 3 eq constraints = 5: x1..x5 are independent.
    return m, [m.x1, m.x2, m.x3, m.x4, m.x5], [bb1, bb2], [bb1_grad, bb2_grad]


def _make_loeppky():
    def bb1(a, b, c):
        bb1.calls += 1
        return 3 * a * b + 2.2 * a * c
    bb1.calls = 0

    def bb1_grad(args, fixed):
        bb1_grad.calls += 1
        a, b, c = args[:3]
        return [3 * b + 2.2 * c, 3 * a, 2.2 * a]
    bb1_grad.calls = 0

    m = ConcreteModel()
    m.x = Var(RangeSet(1, 7), bounds=(0, 1), initialize=0.5)
    m.bb1 = ExternalFunction(bb1, bb1_grad)
    m.obj = Objective(
        expr=6 * m.x[1] + 4 * m.x[2] + 5.5 * m.x[3] + m.bb1(m.x[1], m.x[2], m.x[3])
        + 1.4 * m.x[2] * m.x[3] + m.x[4] + 0.5 * m.x[5] + 0.2 * m.x[6] + 0.1 * m.x[7],
        sense=minimize)
    # pyomo.contrib.trustregion's DOF scan only walks Constraint objects, not
    # the Objective -- since every var here appears ONLY in the objective, it
    # would compute DOF=0 without this trivially non-binding inequality to
    # trigger the scan (doesn't change the feasible region: bounds already
    # cap the sum well under 1e10).
    m.dof_scan = Constraint(expr=sum(m.x[i] for i in range(1, 8)) <= 1e10)
    # No original equality constraints -> all 7 vars are decision variables.
    return m, [m.x[i] for i in range(1, 8)], [bb1], [bb1_grad]


def _make_wing_weight():
    def bb1(a, b):
        bb1.calls += 1
        return a * b
    bb1.calls = 0

    def bb1_grad(args, fixed):
        bb1_grad.calls += 1
        a, b = args[:2]
        return [b, a]
    bb1_grad.calls = 0

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
    m.bb1 = ExternalFunction(bb1, bb1_grad)
    m.obj = Objective(
        expr=0.036 * m.Sw ** 0.758 * m.Wfw ** 0.0035
        * (m.A / (cos(m.Lam * (22 / 7) / 180.0) ** 2)) ** 0.6
        * m.q ** 0.006 * m.lam ** 0.04
        * ((100.0 * m.tc / cos(m.Lam * (22 / 7) / 180.0)) ** -0.3)
        * (m.Nz * m.Wdg) ** 0.49
        + m.bb1(m.Sw, m.Wp),
        sense=minimize)
    dof_vars = [m.Sw, m.Wfw, m.A, m.Lam, m.q, m.lam, m.tc, m.Nz, m.Wdg, m.Wp]
    # Same DOF-scan trigger as Loeppky -- see comment there.
    m.dof_scan = Constraint(expr=sum(dof_vars) <= 1e10)
    # No original equality constraints -> all 10 vars are decision variables.
    return m, dof_vars, [bb1], [bb1_grad]


def _make_welded_beam():
    def bb(a, b, c, d):
        bb.calls += 1
        return (1 + 0.10471) * a ** 2 * b + 0.04811 * c * d * (14 + b)
    bb.calls = 0

    def bb_grad(args, fixed):
        bb_grad.calls += 1
        a, b, c, d = args[:4]
        return [2 * 1.10471 * a * b, 1.10471 * a ** 2 + 0.04811 * c * d,
                0.04811 * d * (14 + b), 0.04811 * c * (14 + b)]
    bb_grad.calls = 0

    m = ConcreteModel()
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    tau_max, sigma_max, delta_max = 13600.0, 30000.0, 0.25
    c1p, c2p = 0.10471, 0.04811
    m.x1 = Var(bounds=(0.125, 5), initialize=1.0)
    m.x2 = Var(bounds=(0.1, 10), initialize=5.0)
    m.x3 = Var(bounds=(0.1, 10), initialize=5.0)
    m.x4 = Var(bounds=(0.1, 5), initialize=1.0)
    m.bb = ExternalFunction(bb, bb_grad)
    m.obj = Objective(expr=m.bb(m.x1, m.x2, m.x3, m.x4), sense=minimize)

    R = sqrt((m.x2 ** 2) / 4 + ((m.x1 + m.x3) / 2) ** 2)
    Mm = P * (L + m.x2 / 2)
    J = 2 * (sqrt(2) * m.x1 * m.x2) * ((m.x2 ** 2) / 12 + ((m.x1 + m.x3) / 2) ** 2)
    t1 = P / (sqrt(2) * m.x1 * m.x2)
    t2 = Mm * R / J
    tau_expr = sqrt(t1 ** 2 + 2 * t1 * t2 * m.x2 / (2 * R) + t2 ** 2)
    sigma_expr = (6 * P * L) / (m.x4 * m.x3 ** 2)
    delta_expr = (4 * P * L ** 3) / (E * m.x3 ** 3 * m.x4)
    pc_expr = ((4.013 * E * sqrt(m.x3 ** 2 * m.x4 ** 6 / 36)) / (L ** 2)) \
        * (1 - (m.x3 / (2 * L)) * sqrt(E / (4 * G)))

    m.g1 = Constraint(expr=tau_expr <= tau_max)
    m.g2 = Constraint(expr=sigma_expr <= sigma_max)
    m.g3 = Constraint(expr=m.x1 - m.x4 <= 0)
    m.g4 = Constraint(expr=c1p * m.x1 ** 2 * m.x2 + c2p * m.x3 * m.x4 * (L + m.x2) - 5 <= 0)
    m.g5 = Constraint(expr=delta_expr <= delta_max)
    m.g6 = Constraint(expr=P - pc_expr <= 0)
    # No original equality constraints -> all 4 vars are decision variables.
    return m, [m.x1, m.x2, m.x3, m.x4], [bb], [bb_grad]


def _make_colville():
    def bb1(a, b):
        bb1.calls += 1
        return 0.8357 * a * b
    bb1.calls = 0

    def bb1_grad(args, fixed):
        bb1_grad.calls += 1
        a, b = args[:2]
        return [0.8357 * b, 0.8357 * a]
    bb1_grad.calls = 0

    def bb2(a, b, c):
        bb2.calls += 1
        return 0.00002584 * a * b - 0.00006663 * c * b
    bb2.calls = 0

    def bb2_grad(args, fixed):
        bb2_grad.calls += 1
        a, b, c = args[:3]
        return [0.00002584 * b, 0.00002584 * a - 0.00006663 * c, -0.00006663 * b]
    bb2_grad.calls = 0

    def bb3(a, b, c):
        bb3.calls += 1
        return 2275.1327 * ((a * b) ** (-1)) - 0.2668 * c * (b ** (-1))
    bb3.calls = 0

    def bb3_grad(args, fixed):
        bb3_grad.calls += 1
        a, b, c = args[:3]
        return [-2275.1327 / (a ** 2 * b),
                -2275.1327 / (a * b ** 2) + 0.2668 * c / b ** 2,
                -0.2668 / b]
    bb3_grad.calls = 0

    def bb4(a, b, c):
        bb4.calls += 1
        return 1330.3294 * ((a * b) ** (-1)) - 0.42 * c * (b ** (-1))
    bb4.calls = 0

    def bb4_grad(args, fixed):
        bb4_grad.calls += 1
        a, b, c = args[:3]
        return [-1330.3294 / (a ** 2 * b),
                -1330.3294 / (a * b ** 2) + 0.42 * c / b ** 2,
                -0.42 / b]
    bb4_grad.calls = 0

    m = ConcreteModel()
    m.x1 = Var(initialize=78, bounds=(78, 102))
    m.x2 = Var(initialize=33, bounds=(33, 45))
    m.x3 = Var(initialize=30, bounds=(27, 45))
    m.x4 = Var(initialize=45, bounds=(27, 45))
    m.x5 = Var(initialize=37, bounds=(27, 45))
    m.bb1 = ExternalFunction(bb1, bb1_grad)
    m.bb2 = ExternalFunction(bb2, bb2_grad)
    m.bb3 = ExternalFunction(bb3, bb3_grad)
    m.bb4 = ExternalFunction(bb4, bb4_grad)
    m.c1 = Constraint(expr=m.bb2(m.x3, m.x5, m.x2) - 0.0000734 * m.x1 * m.x4 - 1 <= 0)
    m.c2 = Constraint(expr=0.000853007 * m.x2 * m.x5 + 0.00009395 * m.x1 * m.x4
                       - 0.00033085 * m.x3 * m.x5 - 1 <= 0)
    m.c3 = Constraint(expr=m.bb4(m.x2, m.x5, m.x1) - 0.30586 * ((m.x2 * m.x5) ** (-1)) * m.x3 ** 2 - 1 <= 0)
    m.c4 = Constraint(expr=0.00024186 * m.x2 * m.x5 + 0.00010159 * m.x1 * m.x2
                       + 0.00007379 * m.x3 ** 2 - 1 <= 0)
    m.c5 = Constraint(expr=m.bb3(m.x3, m.x5, m.x1) - 0.40584 * (m.x5 ** (-1)) * m.x4 - 1 <= 0)
    m.c6 = Constraint(expr=0.00029955 * m.x3 * m.x5 + 0.00007992 * m.x1 * m.x2
                       + 0.00012157 * m.x3 * m.x4 - 1 <= 0)
    m.obj = Objective(expr=5.3578 * m.x3 ** 2 + m.bb1(m.x1, m.x5) + 37.2392 * m.x1)
    # No original equality constraints -> all 5 vars are decision variables.
    # (Every var also appears directly in at least one inequality constraint,
    # so the DOF-scan-only-walks-Constraints quirk doesn't bite here.)
    m._colville_efs = (m.bb1, m.bb2, m.bb3, m.bb4)
    return m, [m.x1, m.x2, m.x3, m.x4, m.x5], [bb1, bb2, bb3, bb4], [bb1_grad, bb2_grad, bb3_grad, bb4_grad]


def _colville_basis(component, ef_expr):
    # Each black box's own closed form as its basis -- all four are simple
    # algebraic functions being treated as black boxes for benchmarking, and
    # each appears exactly once (unlike WilliamsOtto), so identifying by
    # function identity is unambiguous. Avoids the default b(w)=0 rule
    # forcing a ~2412-magnitude mismatch on bb1 at the initial point.
    fcn = ef_expr._fcn
    mbb1, mbb2, mbb3, mbb4 = component.model()._colville_efs
    a = ef_expr.arg(0)
    b = ef_expr.arg(1)
    if fcn is mbb1:
        return 0.8357 * a * b
    c = ef_expr.arg(2)
    if fcn is mbb2:
        return 0.00002584 * a * b - 0.00006663 * c * b
    if fcn is mbb3:
        return 2275.1327 * ((a * b) ** (-1)) - 0.2668 * c * (b ** (-1))
    if fcn is mbb4:
        return 1330.3294 * ((a * b) ** (-1)) - 0.42 * c * (b ** (-1))
    return 0


def _make_six_bb_chain():
    coeff = [1.0, 0.5, 2.0, 0.8, 1.5, 0.3]
    targets = [1.2, 0.9, 2.5, 1.1, 1.8, 0.6]

    bb_fns, grad_fns = [], []

    def make_bb(c):
        def bb(a, b):
            bb.calls += 1
            return c * (a * b + 0.3 * a ** 2)
        bb.calls = 0

        def grad(args, fixed):
            grad.calls += 1
            a, b = args[:2]
            return [c * (b + 0.6 * a), c * a]
        grad.calls = 0
        return bb, grad

    m = ConcreteModel()
    m.x = Var(range(6), initialize=1.0, bounds=(0.2, 3.0))
    m.y = Var(range(6), initialize=1.0, bounds=(-50, 50))
    efs = []
    for i in range(6):
        bb, grad = make_bb(coeff[i])
        bb_fns.append(bb)
        grad_fns.append(grad)
        efs.append(ExternalFunction(bb, grad))
        setattr(m, f'ef{i}', efs[-1])

    def chain_rule(m, i):
        return m.y[i] == efs[i](m.x[i], m.x[(i + 1) % 6])
    m.chain = Constraint(range(6), rule=chain_rule)
    m.obj = Objective(expr=sum((m.y[i] - targets[i]) ** 2 for i in range(6))
                       + 0.1 * sum((m.x[i] - 1.0) ** 2 for i in range(6)),
                       sense=minimize)
    # DOF = 12 vars - 6 eq constraints = 6: x[0..5] are independent.
    return m, [m.x[i] for i in range(6)], bb_fns, grad_fns


def _make_highdim_sep10():
    n = 10
    targets = [1.0 + (i % 3) * 0.5 for i in range(n)]
    bb_fns, grad_fns = [], []

    def make_bb():
        def bb(a):
            bb.calls += 1
            return math.exp(0.1 * a) - 1.0 + 0.5 * a ** 2
        bb.calls = 0

        def grad(args, fixed):
            grad.calls += 1
            a = args[0]
            return [0.1 * math.exp(0.1 * a) + a]
        grad.calls = 0
        return bb, grad

    m = ConcreteModel()
    m.x = Var(range(n), bounds=(-5.0, 5.0), initialize=2.0)
    m.y = Var(range(n), bounds=(-50.0, 200.0), initialize=0.0)
    efs = []
    for i in range(n):
        bb, grad = make_bb()
        bb_fns.append(bb)
        grad_fns.append(grad)
        efs.append(ExternalFunction(bb, grad))
        setattr(m, f'ef{i}', efs[-1])

    def link(m, i):
        return m.y[i] == efs[i](m.x[i])
    m.link = Constraint(range(n), rule=link)
    m.obj = Objective(expr=sum((m.x[i] - targets[i]) ** 2 for i in range(n))
                       + 0.1 * sum(m.y[i] for i in range(n)), sense=minimize)
    # DOF = 20 vars - 10 eq constraints = 10: x[0..9] are independent.
    return m, [m.x[i] for i in range(n)], bb_fns, grad_fns


def _yoshio_basis(component, ef_expr):
    # r(w) = w1^2 - w2, per the paper's P2 basis function.
    x = ef_expr.arg(0)
    y = ef_expr.arg(1)
    return x ** 2 - y


def _make_yoshio():
    """P2, Problem B.1.3: w1 free, w2 >= -2, start at the origin."""
    def yosh_ext(a, b):
        yosh_ext.calls += 1
        return a ** 2 + b ** 2
    yosh_ext.calls = 0

    def yosh_grad(args, fixed):
        yosh_grad.calls += 1
        a, b = args[:2]
        return [2 * a, 2 * b]
    yosh_grad.calls = 0

    m = ConcreteModel()
    m.x1 = Var(initialize=0)
    m.x2 = Var(bounds=(-2.0, None), initialize=0)
    m.EF = ExternalFunction(yosh_ext, yosh_grad)
    m.c1 = Constraint(expr=2 * m.x1 + m.x2 + 10.0 == m.EF(m.x1, m.x2))
    m.obj = Objective(expr=(m.x1 - 1) ** 2 + (m.x2 - 3) ** 2 + m.EF(m.x1, m.x2) ** 2)
    return m, [m.x1], [yosh_ext], [yosh_grad]


# label: (builder, indices of the decision variables, surrogate basis or None)
PROBLEMS = {
    'P1': (_make_biegler, [0], None),
    'P2': (_make_yoshio, [0], _yoshio_basis),
    'Eason': (_make_eason, [0, 1, 2], None),
    'Wing Weight': (_make_wing_weight, list(range(10)), None),
    'Loeppky': (_make_loeppky, list(range(7)), None),
    'Welded Beam': (_make_welded_beam, [0, 1, 2, 3], None),
    'Himmelblau': (_make_himmelblau, [0, 1, 2, 3, 4], None),
    'Colville': (_make_colville, [0, 1, 2, 3, 4], _colville_basis),
    'Six_BB_Chain': (_make_six_bb_chain, list(range(6)), None),
    'High_dim_sep_10': (_make_highdim_sep10, list(range(10)), None),
}
