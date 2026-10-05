#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from math import pow

import time
import sys

from pyomo.common.dependencies import numpy as np

from pyomo.environ import Var, value
from pyomo.common.config import (
    ConfigBlock, ConfigValue, PositiveInt, PositiveFloat, NonNegativeFloat, In)

from filterMethod import (
    FilterElement, Filter)
from funnelMethod import Funnel
# Importing GJHPseudoSolver also registers the 'local.gjh' SolverFactory entry,
# so any entry script that reaches TRF() gets it automatically. ensure_gjh()
# resolves/installs the gjh executable (PATH > bundled repo copy > download).
from GJHPseudoSolver import ensure_gjh
from helper import (cloneXYZ, packXYZ, system_report, set_value_novalidate)
from Logger import Logger
from PyomoInterface import (
    PyomoInterface, ROMType, ALGType, ROM_NAME_TO_TYPE)

def TRF(m, eflist, config):
    """The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    eflist is a list of ExternalFunction objects that should be
    treated with the trust region

    config is the persistent set of variables defined 
    in the ConfigBlock class object

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR" TODO: reverse
    the transformation.
    """

    # Whole-run timers: wall clock (perf_counter) and process CPU time.
    _t_wall0 = time.perf_counter()
    _t_cpu0 = time.process_time()

    # Make sure the gjh executable is available before anything else runs:
    # existing PATH installation > bundled repo copy > Pyomo config dir >
    # automatic download (then prepended to PATH for this process).
    ensure_gjh(verbose=True)

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m, eflist, config)

    # ---- startup banner: configuration + problem size --------------------
    # x = black-box INPUT vars, y = black-box OUTPUT (surrogate/tear) vars,
    # z = remaining glass-box decision vars; total = x + y + z.
    _lx0, _ly0, _lz0 = problem.lx, problem.ly, problem.lz
    _neq0, _nineq0 = problem.count_constraints()
    _strat = {0: 'Filter', 1: 'Funnel'}.get(config.globalization_strategy, '?')
    print("========== TRF SOLVER ==========")
    print(f"Surrogate (ROM)       : {config.reduced_model_type}")
    print(f"Globalization         : {config.globalization_strategy} ({_strat})")
    print(f"Algorithm type        : {config.algorithm_type}")
    print(f"Variables             : {_lx0 + _ly0 + _lz0} total  "
          f"(x|y|z = {_lx0}|{_ly0}|{_lz0}) "
          f"[x=black-box inputs, y=black-box outputs, z=remaining]")
    print(f"Black-box functions   : {_ly0}")
    print(f"Constraints           : {_neq0 + _nineq0}  "
          f"(equality {_neq0} | inequality {_nineq0})")
    print(f"Tolerances            : ep_i={config.ep_i:.1e}, "
          f"ep_chi={config.ep_chi:.1e}, ep_delta={config.ep_delta:.1e}, "
          f"max_it={config.max_it}")
    print("================================\n")

    x, y, z = problem.getInitialValue()
    iteration = -1

    romParam, yr, FEs = problem.buildROM(x, config.sample_radius)
    #y = yr
    rebuildROM = False
    
    ROMAccuracy = False
    
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    EV = None
    FEs = None
    thetak = np.linalg.norm(yr - yk,1)
    IT = None
    objk = problem.evaluateObj(x, y, z)
    
    stepNorm = 1e10
    
    funnel = Funnel(phi_init=thetak,
                f_best_init=objk,
                phi_min=config.phi_min,
                kappa_f=config.kappa_f,
                kappa_r=config.kappa_r,
                alpha=config.alpha,
                beta=config.beta,
                mu_s=config.mu_s,
                eta=config.eta)

    exit_msg = "EXIT: terminated"

    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
            #print(xk)
        start_time = time.time()
        # increment iteration counter
        iteration = iteration + 1
        if iteration > config.max_it:
            print("EXIT: Maxmium iterations\n")
            exit_msg = "EXIT: Maximum iterations"
            break

        ######  Why is this here ###########
        if iteration == 1:
            config.sample_region = False
        ################################

        # Keep Sample Region within Trust Region
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust*config.trust_radius,
                config.delta_min)
            rebuildROM = True

        #Generate a RM r_k (x) that is kappa-fully linear on sigma k
        if(rebuildROM):
            # config.reduced_model_type is a name: 'linear'/'quadratic'/
            # 'quadratic_simp'/'ts'. For the polynomial surrogates we drop the
            # fidelity as the trust radius shrinks; 'ts' is used as configured.
            if config.reduced_model_type in ('linear', 'quadratic', 'quadratic_simp'):
                if config.trust_radius < 1e-3:
                    problem.romtype = ROMType.linear
                # Added this switching / Gul
                elif config.trust_radius >= 1e-3 and config.trust_radius < 1e-1:
                    problem.romtype = ROMType.quadratic_simp
                else:
                    problem.romtype = ROM_NAME_TO_TYPE[config.reduced_model_type]
            elif config.reduced_model_type == 'ts':
                problem.romtype = ROMType.ts

            romParam, yr, FEs = problem.buildROM(x, config.sample_radius)
            
            FEs = FEs + (1*len(romParam))
        else:
            FEs = 0 + (1*len(romParam))
        
        g, J, H, varlist, conlist, HM, EV, H_abs, EV_abs, H_clamped, EV_clamped = problem.grad_hess_calc(x, y, z, romParam)
        
        if config.algorithm_type == 0:
            problem.algtype == ALGType.Normal_TR
        elif config.algorithm_type == 1:
            problem.algtype == ALGType.Simple_Diagonal_Loading
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(HM)
            x_eig, y_eig, z_eig = problem.create_ordered_eigenvalue_lists(EV)
        elif config.algorithm_type == 2:
            problem.algtype == ALGType.Hessian_Clamped_Eigenvalues
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_clamped)
        elif config.algorithm_type == 3:
            problem.algtype == ALGType.Hessian_Absolute_Eigenvalues
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_abs)
        elif config.algorithm_type == 4:
            if ROMAccuracy == False:
                problem.algtype == ALGType.Hessian_Absolute_Eigenvalues
                x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_abs)
            elif ROMAccuracy == True:
                problem.algtype == ALGType.Hessian_Clamped_Eigenvalues
                x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_clamped)
        else:
            raise('The selected variant of the algorithm is not supported!')
            
        
        # print(x_hess_rows)
        # print(y_hess_rows)
        # print(z_hess_rows)
        
        # print(x_eig)
        # print(y_eig)
        # print(z_eig)
        
        # Criticality Check
        if iteration > 0:
            flag, chik = problem.criticalityCheck(x, y, z, romParam, g, J, varlist, conlist)
            # print(flag)
            # print(chik)
            # print(EV)
            if (not flag):
                raise Exception("Criticality Check fails!\n")

        # Save the iteration information to the logger
        logger.newIter(iteration,xk,yk,zk,thetak,objk,chik,list(EV.values()),FEs,IT,
                       config.print_variables)

        # Check for Termination
        if (thetak < config.ep_i and
            chik < config.ep_chi and
            config.sample_radius < config.ep_delta):
            print("EXIT: OPTIMAL SOLUTION FOUND")
            exit_msg = "EXIT: OPTIMAL SOLUTION FOUND"
            break

        # Possibly a local optimum, added this termication condition / Gul
        if (thetak < config.ep_i and
            stepNorm < config.ep_s):
            print("EXIT: POSSIBLY AN OPTIMAL SOLUTION IS FOUND")
            exit_msg = "EXIT: POSSIBLY AN OPTIMAL SOLUTION IS FOUND"
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if (config.trust_radius <= config.delta_min and thetak < config.ep_i):
            if subopt_flag:
                print("EXIT: FEASIBLE SOLUTION FOUND")
                exit_msg = "EXIT: FEASIBLE SOLUTION FOUND"
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        # New criticality phase
        if not config.sample_region:
            config.sample_radius = config.trust_radius/2.0
            if config.sample_radius > chik * config.criticality_check:
                config.sample_radius = config.sample_radius/10.0
            config.trust_radius = config.sample_radius*2
        else:
            config.sample_radius = max(min(config.sample_radius,
                                   chik*config.criticality_check),
                               config.delta_min)

        logger.setCurIter(trustRadius=config.trust_radius,
                          sampleRadius=config.sample_radius)

        # Compatibility Check (Definition 2)
        # radius=max(kappa_delta*config.trust_radius*min(1,kappa_mu*config.trust_radius**mu),
        #            delta_min)
        radius = max(config.kappa_delta*config.trust_radius *
                     min(1,
                         config.kappa_mu*pow(config.trust_radius,config.mu)),
                     config.delta_min)

        try:
            if config.algorithm_type == 0:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, None, None, None, None, None, None, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            elif config.algorithm_type == 1:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, x_eig, y_eig, z_eig, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            elif config.algorithm_type == 2 or config.algorithm_type == 3 or config.algorithm_type == 4:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, None, None, None, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            else:
                raise('Please choose one of the available variants (0-4)!')
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")


        theNorm = np.linalg.norm(x - xk, 2)**2 + np.linalg.norm(z - zk, 2)**2
        if (obj - config.compatibility_penalty*theNorm >
            config.ep_compatibility):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            
            theta = np.linalg.norm(yr - y, 1)

            logger.iterlog.restoration = True

            if config.globalization_strategy == 0: 
                fe = FilterElement(
                     objk - config.gamma_f*thetak,
                     (1 - config.gamma_theta)*thetak)
                filteR.addToFilter(fe)
            elif config.globalization_strategy == 1:
                pass

            rhok = 1 - ((theta - config.ep_i)/max(thetak, config.ep_i))
            if rhok < config.eta1:
                config.trust_radius = max(config.gamma_c*config.trust_radius,
                                  config.delta_min)
                ROMAccuracy = False
            elif rhok >= config.eta2:
                config.trust_radius = min(config.gamma_e*config.trust_radius,
                    config.radius_max)
                ROMAccuracy = True
            elif rhok >= config.eta1 and rhok < config.eta2:
                ROMAccuracy = False

            obj = problem.evaluateObj(x, y, z)

            # stepNorm = min(np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf), config.step_max)
            stepNorm = np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf)
            logger.setCurIter(stepNorm=stepNorm)
            
            

        else:
            # Solve TRSP_k
            if config.algorithm_type == 0:
                flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, None, None, None, None, None, None, 
                                          romParam, config.trust_radius, config.algorithm_type)
            elif config.algorithm_type == 1:
                 flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, x_eig, y_eig, z_eig, 
                                           romParam, config.trust_radius, config.algorithm_type)  
            elif config.algorithm_type == 2 or config.algorithm_type == 3 or config.algorithm_type == 4:
                flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, None, None, None,
                                          romParam, config.trust_radius, config.algorithm_type)             
            else:
                raise('Please add other variants of the algorithm')
            
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)
            
            # stepNorm = min(np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf), config.step_max)
            stepNorm = np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf)
            logger.setCurIter(stepNorm=stepNorm)

            # theta = Trial theta, thetak = current theta
            theta = np.linalg.norm(yr - y, 1)
            
            # Calculate rho for theta step trust region update
            rhok = 1 - ((theta - config.ep_i) /
                    max(thetak, config.ep_i))
            
            # Filter method
            if config.globalization_strategy == 0:
            
                fe = FilterElement(obj, theta)

                if not filteR.checkAcceptable(fe, config.theta_max) and iteration > 0:
                    logger.iterlog.rejected = True
                    config.trust_radius = max(config.gamma_c*stepNorm,
                                  config.delta_min)
                    # config.trust_radius = max(config.gamma_c*config.trust_radius,
                    #                   config.delta_min)
                    rebuildROM = False
                    ROMAccuracy = False
                
                    x, y, z = cloneXYZ(xk, yk, zk)
                    continue

                # Switching Condition and Trust Region update
                if (((objk - obj) >= config.kappa_theta*
                     pow(thetak, config.gamma_s))
                     and
                     (thetak < config.theta_min)):
                    logger.iterlog.fStep = True

                    config.trust_radius = min(
                         max(config.gamma_e*stepNorm, config.trust_radius),
                         config.radius_max)
                    # config.trust_radius = min(config.gamma_e*config.trust_radius,
                    #     config.radius_max)
                    ROMAccuracy = True

                else:
                    logger.iterlog.thetaStep = True

                    fe = FilterElement(
                       obj - config.gamma_f*theta,
                       (1 - config.gamma_theta)*theta)
                    
                    filteR.addToFilter(fe)

                    #trust region update
                    if rhok < config.eta1:
                        config.trust_radius = max(config.gamma_c*stepNorm,
                                      config.delta_min)
                        # config.trust_radius = max(config.gamma_c*config.trust_radius,
                        #                   config.delta_min)
                        ROMAccuracy = False
                    elif rhok >= config.eta2:
                        config.trust_radius = min(
                            max(config.gamma_e*stepNorm, config.trust_radius),
                            config.radius_max)
                        # config.trust_radius = min(config.gamma_e*config.trust_radius,
                        #     config.radius_max)
                        ROMAccuracy = True
                    elif rhok >= config.eta1 and rhok < config.eta2:
                        ROMAccuracy = False
                    
            # Funnel method
            elif config.globalization_strategy == 1:
                status = funnel.classify_step(thetak, theta, objk, obj, config.trust_radius)

                if status == 'f':
                    funnel.accept_f(theta, obj)
                    logger.iterlog.fStep = True
                    config.trust_radius = min(
                         max(config.gamma_e*stepNorm, config.trust_radius),
                         config.radius_max)
                    # config.trust_radius = min(config.gamma_e*config.trust_radius,
                    #     config.radius_max)
                    ROMAccuracy = True
                elif status in ('theta', 'theta-relax'):
                    if status == 'theta':
                        funnel.accept_theta(theta)
                        logger.iterlog.thetaStep = True
                    else:
                        funnel.relax_theta(theta)
                        logger.iterlog.relaxthetaStep = True
                    
                    #trust region update
                    if rhok < config.eta1:
                        config.trust_radius = max(config.gamma_c*stepNorm,
                                      config.delta_min)
                        # config.trust_radius = max(config.gamma_c*config.trust_radius,
                        #                   config.delta_min)
                        ROMAccuracy = False
                    elif rhok >= config.eta2:
                        config.trust_radius = min(
                            max(config.gamma_e*stepNorm, config.trust_radius),
                            config.radius_max)
                        # config.trust_radius = min(config.gamma_e*config.trust_radius,
                        #     config.radius_max)
                        ROMAccuracy = True
                    elif rhok >= config.eta1 and rhok < config.eta2:
                        ROMAccuracy = False
                else:      # 'reject'
                    logger.iterlog.rejected = True
                    config.trust_radius = max(config.gamma_c*stepNorm,
                                  config.delta_min)
                    # config.trust_radius = max(config.gamma_c*config.trust_radius,
                    #                   config.delta_min)
                    rebuildROM = False
                    ROMAccuracy = False
                
                    x, y, z = cloneXYZ(xk, yk, zk)
                    continue

            

        # Accept step
        
        # print('xk')
        # print(xk)
        # print('yk')
        # print(yk)
        # print('zk')
        # print(zk)
        # print(theta)
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj
        end_time = time.time()
        IT = end_time - start_time


    logger.printVectors()
#    problem.reverseTransform()

    # ---------------------------------------------------------------------
    # Run summary: wall/CPU time, exact external (black-box) evaluations
    # per function + total, cache statistics, and the machine the run
    # executed on. Printed once at the end (captured in the results log too).
    # ---------------------------------------------------------------------
    wall_s = time.perf_counter() - _t_wall0
    cpu_s = time.process_time() - _t_cpu0
    print("\n========== RUN SUMMARY ==========")
    print(f"Status                : {exit_msg}")
    print(f"Iterations            : {iteration}")
    # x = black-box INPUT vars, y = black-box OUTPUT (surrogate/tear) vars,
    # z = remaining glass-box decision vars; total = x + y + z.
    _lx, _ly, _lz = problem.lx, problem.ly, problem.lz
    _neq, _nineq = problem.count_constraints()
    print(f"Problem size          : {_lx + _ly + _lz} variables  "
          f"(x|y|z = {_lx}|{_ly}|{_lz})")
    print(f"    x = black-box inputs | y = black-box outputs | z = remaining")
    print(f"Black-box functions   : {_ly}")
    print(f"Constraints           : {_neq + _nineq}  "
          f"(equality {_neq} | inequality {_nineq})")
    print(f"Wall time (s)         : {wall_s:.3f}")
    print(f"CPU time (s)          : {cpu_s:.3f}   (process_time)")
    print("External BB evaluations (true black-box callback calls):")
    for i in sorted(problem.bb_eval_counts):
        print(f"    bb[{i}]       : {problem.bb_eval_counts[i]}")
    print(f"    {'TOTAL':<10}: {problem.bb_eval_total}")
    print(f"    (evaluateDx sweeps: {problem.countDx})")
    if problem.bb_cache_enabled:
        hits = problem.bb_cache_hits
        misses = problem.bb_eval_total
        requests = hits + misses
        rate = (100.0 * hits / requests) if requests else 0.0
        print(f"BB cache              : {hits} hits / {requests} requests "
              f"({rate:.1f}% hit rate); {misses} true evaluations")
    else:
        print("BB cache              : disabled")

    # Optional: constraint dual values (multipliers) from the final subproblem
    # solve. Reports the count plus the largest-magnitude duals. Wrapped so a
    # reporting hiccup can never abort an otherwise-successful solve.
    duals_dict = dict(getattr(problem, 'last_duals', {}) or {})
    if getattr(problem, 'report_duals', False):
        print(f"Dual values           : {len(duals_dict)} captured "
              f"(last subproblem solve); largest |dual|:")
        for name, val in sorted(duals_dict.items(), key=lambda t: -abs(t[1]))[:15]:
            print(f"    {name:<32}: {val:.6g}")
    print("System:")
    print(system_report())
    print("=================================\n")

    return {
        'status':           exit_msg,
        'iterations':       int(iteration),
        'n_eq_constraints': int(_neq),
        'n_ineq_constraints': int(_nineq),
        'duals':            duals_dict,
        'final_theta':      float(thetak),
        'final_chi':        float(chik),
        'bb_evals':         int(problem.bb_eval_total),
        'bb_evals_detail':  {i: int(problem.bb_eval_counts[i])
                             for i in sorted(problem.bb_eval_counts)},
        'bb_cache_enabled': bool(problem.bb_cache_enabled),
        'bb_cache_hits':    int(problem.bb_cache_hits),
        'evaluateDx_sweeps': int(problem.countDx),
        'wall_s':           float(wall_s),
        'cpu_s':            float(cpu_s),
        'n_vars':           int(_lx + _ly + _lz),
        'n_vars_xyz':       (int(_lx), int(_ly), int(_lz)),
    }


class Tee:
    """Write to both the console and a file simultaneously (used by
    TrustRegionSolver when save_results_txt is enabled)."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout  # Save original stdout

    def write(self, message):
        self.stdout.write(message)  # Print to console
        self.file.write(message)    # Write to file
        # Return chars written, per the file-object protocol. Pyomo's
        # StreamIndenter does `written += ostream.write(...)`, so a None
        # return raises TypeError on model.display() under newer Pyomo.
        return len(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class TrustRegionSolver:
    """Trust-region solver wrapper. Holds all solver options (the CONFIG block
    below) and drives TRF(). Previously lived in RunFile.py; moved here so the
    run scripts only define problems."""

    CONFIG = ConfigBlock('Trust Region')

    CONFIG.declare('solver', ConfigValue(
        default='gams',
        description='solver to use, defaults to gams solvers, specify the solver at the end, you may choose independent solvers such as IPOPT as well',
    ))

    CONFIG.declare('solver_options', ConfigBlock(
        implicit=True,
        description='Options to pass to the subproblem solver',
    ))

    CONFIG.declare('max it', ConfigValue(
        default=100,
        domain=PositiveInt,
    ))

    # Initialize trust radius
    CONFIG.declare('trust radius', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
    ))

    # Initialize sample region and radius
    CONFIG.declare('sample region', ConfigValue(
        default=True,
        domain=bool,
    ))

    CONFIG.declare('sample radius', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    # Placeholder for 'radius max', value to be set in __init__
    CONFIG.declare('radius max', ConfigValue(
        default=None,
        domain=PositiveFloat,
    ))

    # Termination tolerances
    CONFIG.declare('ep i', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep s', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep delta', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep chi', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
    ))

    CONFIG.declare('delta min', ConfigValue(
        default=1e-6,
        domain=PositiveFloat,
        description='delta min <= ep delta',
    ))

    # Compatibility Check Parameters
    CONFIG.declare('kappa delta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
    ))

    CONFIG.declare('kappa mu', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
    ))

    CONFIG.declare('mu', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep compatibility', ConfigValue(
        default=None,  # Placeholder, to be set in __init__
        domain=PositiveFloat,
        description='Suggested value: ep compatibility == ep i',
    ))

    CONFIG.declare('compatibility penalty', ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
    ))

    # Criticality Check Parameters
    CONFIG.declare('criticality check', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('gamma e', ConfigValue(
        default=2.5,
        domain=PositiveFloat,
    ))

    # Switching Condition
    CONFIG.declare('gamma s', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
    ))

    CONFIG.declare('kappa theta', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    CONFIG.declare('kappa f', ConfigValue(
        default=0.25,
        domain=PositiveFloat,
        description='funnel‑shrink factor after f‑type',
    ))

    CONFIG.declare('kappa r', ConfigValue(
        default=1.1,
        domain=PositiveFloat,
        description='funnel expand factor for relax theta step',
    ))

    CONFIG.declare('theta min', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))

    CONFIG.declare('phi min', ConfigValue(
        default=1e-8,
        domain=PositiveFloat,
        description='hard floor on funnel width',
    ))

    # Filter
    CONFIG.declare('gamma f', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('gamma theta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('theta max', ConfigValue(
        default=50,
        domain=PositiveInt,
    ))

    CONFIG.declare('alpha', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='curvature exponent ( (theta)^alpha )',
    ))

    CONFIG.declare('beta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
        description='extra shrink required for theta‑type',
    ))

    CONFIG.declare('mu s', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='switching coefficient δ',
    ))

    CONFIG.declare('eta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='Armijo coefficient for f‑type',
    ))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default=0.05,
        domain=PositiveFloat,
    ))

    CONFIG.declare('eta2', ConfigValue(
        default=0.2,
        domain=PositiveFloat,
    ))

    # Output level (replace with real print levels)
    CONFIG.declare('print variables', ConfigValue(
        default=False,
        domain=bool,
    ))

    # Sample Radius reset parameter
    CONFIG.declare('sample radius adjust', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Black-box evaluation cache: memoise true black-box callback calls on an
    # exact (function, args) key. Never changes results, only the number of
    # true evaluations; hits/misses are reported in the run summary.
    CONFIG.declare('bb cache', ConfigValue(
        default=True,
        domain=bool,
        description='True = memoise true black-box evaluations (exact key); '
                    'False = always re-evaluate.',
    ))

    # Report constraint dual values (multipliers) from the final subproblem
    # solve in the run summary. Off by default (adds an IMPORT dual Suffix).
    CONFIG.declare('report duals', ConfigValue(
        default=False,
        domain=bool,
        description='True = capture and report constraint dual values from the '
                    'final subproblem solve.',
    ))

    # Default reduced model type (surrogate). Use names rather than codes.
    CONFIG.declare('reduced model type', ConfigValue(
        default='linear',
        domain=In(['linear', 'quadratic', 'quadratic_simp', 'ts']),
        description="'linear' = Linear, 'quadratic' = Quadratic, "
                    "'quadratic_simp' = Simplified Quadratic, "
                    "'ts' = Taylor Series Approximation",
    ))

    # Default globalization strategy
    CONFIG.declare('globalization strategy', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Filter, 1 = Funnel',
    ))

    # Default algorithm type
    CONFIG.declare('algorithm type', ConfigValue(
        default=0,
        domain=In([0, 1, 2, 3, 4]),
        description='0 = Normal Trust-region Subproblem, 1 = Trust-region Subproblem with Simple Diagonal Loading, 2 = Trust-region Subproblem with Projected Hessian_based Constraint (Clamped Version), 3 = Trust-region Subproblem with Projected Hessian_based Constraint (Absolute Version), 4 = Trust-region Subproblem with Adaptive Projected Hessian',
    ))

    # Results log: when True, tee console output AND model.display() detailed
    # results to a .txt file, so run scripts no longer need their own Tee
    # boilerplate. Default off -> behaviour unchanged (console only).
    CONFIG.declare('save results txt', ConfigValue(
        default=False,
        domain=bool,
        description='When True, write the full run log plus model.display() '
                    'to a .txt file (auto-named unless results_filename is set).',
    ))

    CONFIG.declare('results filename', ConfigValue(
        default=None,
        description='Explicit results .txt filename used when save_results_txt '
                    'is True. Default: auto-named '
                    '<problem>_A<alg>_S<rom>_GS<gs>.txt',
    ))

    def __init__(self, **kwargs):
        self.config = self.CONFIG(kwargs, preserve_implicit=True)

        # Set 'radius max' if not already provided
        if self.config['radius max'] is None:
            self.config['radius max'] = 1000.0 * self.config['trust radius']

        # Set 'ep compatibility' if not already provided
        if self.config['ep compatibility'] is None:
            self.config['ep compatibility'] = self.config['ep i']

    def solve(self, model, eflist, **kwds):
        """Solve `model`. Pass problem_name='...' to label the auto-generated
        results file when save_results_txt is enabled."""
        # Store all data needed to change in the original model
        model._tmp_trf_data = (list(model.component_data_objects(Var)), eflist, self.config)

        # Clone the model to work on it
        inst = model.clone()

        # Optional results log: tee both the solve output and the final
        # model.display() detailed results to a .txt file in the cwd.
        f = None
        saved_stdout = None
        if self.config.save_results_txt:
            fname = self.config.results_filename or (
                f"{kwds.get('problem_name', 'TRF_Result')}"
                f"_A{self.config['algorithm type']}"
                f"_S{self.config['reduced model type']}"
                f"_GS{self.config['globalization strategy']}.txt")
            f = open(fname, 'w', encoding='utf-8')
            saved_stdout = sys.stdout
            sys.stdout = Tee(f)

        result = None
        try:
            # Call the TRF function on the cloned model
            result = TRF(inst, inst._tmp_trf_data[1], inst._tmp_trf_data[2])

            # Copy potentially changed variable values back to the original model
            # (skip_validation: tolerate tiny solver domain violations, as the
            # subproblem solves do).
            for inst_var, orig_var in zip(inst._tmp_trf_data[0], model._tmp_trf_data[0]):
                set_value_novalidate(orig_var, value(inst_var))

            # Detailed results captured in the same log file
            if self.config.save_results_txt:
                model.display()
        finally:
            if f is not None:
                sys.stdout = saved_stdout
                f.close()
                print(f"[results log] written to {f.name}")

        # Augment the structured result with the objective in the user's own
        # sense (read off the original model after copy-back) so callers don't
        # have to scrape stdout.
        if result is not None:
            try:
                result['final_obj'] = value(model.obj)
            except Exception:
                result['final_obj'] = float('nan')
        return result

