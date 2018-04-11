# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:05:00 2015

Alexander Pitchford

Dynamically build the configuration for the Quantum Self Optim simulation
from a configuration file
"""

import sys
import os
import collections
import shutil
import argparse
import numpy as np
from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal)
#import timeit
#import datetime

from functools import reduce

#QuTiP
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
import qutip.logging_utils as logging
logger = logging.get_logger()
#QuTiP control modules
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.propcomp as propcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.errors as errors
import qutip.control.pulsegen as pulsegen
import qutip.control.loadparams as loadparams

from choi_closed_fidcomp import FidCompPureChoiLocal, FidCompPureChoiGlobal
import qso
import qsostats

log_level = logging.DEBUG
logger.setLevel(log_level)

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs."""
    if num != 0:
        return np.round(num,
                        -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

def gen_config(param_fname=None):
    """
    Generate the configuration for the quantum system and optimiser
    In general values from the parameter file overwrite those hardcoded
    and these in turn are overwritten by cmdline args

    Returns
    -------
    Optimizer
    """

    #TODO: Still working on this
    cfg = gen_optim_config(param_fname)
    optim = gen_optim_objects(cfg)
    config_dynamics(optim.dynamics)

    return optim


def gen_optim_config(param_fname=None):
    """
    Create the optimiser objects and load the configuration.

    Returns
    -------
    OptimConfig
    """

    parser = argparse.ArgumentParser(description="Command line argument parser")
    parser.add_argument('-p', '--param_file', type=str, default="",
                                    help="Parameters file name")
    parser.add_argument('-o', '--output_dir', type=str, default="",
                                    help="output sub directory")
    parser.add_argument('-j', '--job_id', type=int, default=0,
                                    help="Job id (from bsub)")
    parser.add_argument('-i', '--job_idx', type=int, default=0,
                                    help="Job index (from bsub batch array)")
    parser.add_argument('-I', '--idx_opt', type=str, default='',
                                    help="Job index option (what to do with it)")
    #parser.add_argument('-S', '--job_size', type=int, default=0,
    #                                help="Number of jobs in array")
    parser.add_argument('-N', '--num_qubits', type=int, default=0,
                                    help="Number of qubits")
    parser.add_argument('-u', '--init_amps', type=str, default=0,
                                    help="File name of initial amplitudes")
    parser.add_argument('-m', '--num_tslots', type=int, default=0,
                                    help="Number of timeslots")
    parser.add_argument('-T', '--evo_time', type=float, default=0,
                                    help="Total evolution time")
    parser.add_argument('--evo_time_npi', type=int, default=0,
                        help="Total evolution time in mulitples of pi")
    parser.add_argument('--evo_time_npi_g0s', type=int, default=0,
                        help="Total evolution time in mulitples of pi "
                            "scaled by first coupling constant")
    parser.add_argument('-a', '--numer_acc', type=float, default=-1.0,
                                    help="Numerical accuracy")
    parser.add_argument('-e', '--fid_err_targ', type=float, default=0.0,
                                    help="Fidelity error target")
    parser.add_argument('-n', '--num_cpus', type=int, default=0,
                                    help="Fidelity error target")
    parser.add_argument('-M', '--mp_opt', type=str, default='',
                                    help="Multiprocessing option")
    parser.add_argument('--mem_opt', type=int, default=0,
                                    help="Memory optimising level")
    parser.add_argument('--pulse_scaling', type=int, default=0.0,
                                    help="Initial pulse scaling")
    parser.add_argument('--max_wall_time', type=int, default=0.0,
                                    help="Maximum simulation wall time")

    cfg = optimconfig.OptimConfig()
    cfg.args = vars(parser.parse_args())
    cfg.log_level = log_level

    cfg.use_param_file = True

    if len(cfg.args['param_file']) > 0:
        param_fname = cfg.args['param_file']
        cfg.param_fpath = os.path.join(os.getcwd(), param_fname)
        if not os.path.isfile(cfg.param_fpath):
            raise ValueError("Commandline argument parameter "
                "file name '{}' does not exist.".format(param_fname))
        else:
            print("Parameters will be read from:\n{}".format(cfg.param_fpath))
    elif param_fname is None:
        print ("No parameter file. Using defaults in code.")
    elif os.path.abspath(param_fname):
        cfg.param_fname = os.path.basename(param_fname)
        cfg.param_fpath = param_fname
        param_fname = cfg.param_fname
    else:
        cfg.param_fpath = os.path.join(os.getcwd(), param_fname)
        if (not os.path.isfile(cfg.param_fpath)):
            print ("Default parameter file {} not present. "
                    "Using defaults in code.").format(param_fname)
            cfg.use_param_file = False
    cfg.param_fname = param_fname

    # Script operational parameters
    cfg.job_id = 0
    cfg.job_idx = 0
    cfg.verbosity = 0
    # Folder where output files will be stored
    # ~ can be used for home folder
    cfg.output_dir = None
    # Used as part of output file name.
    cfg.output_base_name = 'general'
    # True means print messages saved in output file as well as std_out
    cfg.double_print = True
    cfg.output_files = True
    cfg.plot_result = False
    cfg.plot_file_type = 'PNG'
    cfg.save_plots = False
    cfg.save_results = True
    # File extension for output file
    cfg.data_file_ext = "txt"
    # An extended file extension, which includes job_id and job_idx
    # when these are set by the job scheduler
    cfg.out_file_ext = "txt"

    # Optimizer config
    # Multi-var optimisation methdo
    cfg.optim_method = 'LBFGSB'
    # Pulse generator type
    cfg.p_type = 'RND'
    cfg.check_grad = False
    # If true then processing stats will be collected
    cfg.gen_stats = True
    # stats_type options: standard|local
    # local can be used with the pure_Choi_local FidelityComputer
    cfg.stats_type = 'standard'
    cfg.report_stats = True
    # Save pulse amplitdues to file (or not)
    cfg.save_initial_amps = False
    cfg.save_final_amps = False
    # fid_type options pure_Choi_local|pure_Choi_global|unit_global
    cfg.fid_type = 'pure_choi_local'
    # These will be used as bounds for the PulseGenerator and the Optimizer
    cfg.amp_lbound = -np.Inf
    cfg.amp_ubound = np.Inf
    # Number of repitions for each pulse optimisation
    cfg.num_reps = 2
    # Number of cpus to utilise
    cfg.num_cpus = 1
    # Multi-processing options:-
    #    <blank> - repetitions will be spread over available cpus
    #    num_tslots - scenarios with different num_tslots spread over cpus
    #    numer_acc_limit - automatic search numerical accuracy threshold
    cfg.mp_opt = ""
    # Maximum number of scenarios when mp_opt=numer_acc_limit
    cfg.max_mp_scens = np.inf
    # True implies that some external job scheduler is organising some
    # multi-processing using the job_idx to set parameter values
    # hence a fuller name will be in the output file.
    cfg.ext_mp = False
    # Number of threads given to each process
    # Effective only with large systems, > 6 qubits, and only really useful
    # on a large cluster
    cfg.num_threads = 1

    if cfg.use_param_file:
        # load the config parameters
        # note these will overide those above if present in the file
        print("Loading config parameters from {}".format(cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, config=cfg)

    # override with command line params (if any)
    if cfg.args['num_qubits'] > 0:
        cfg.num_qubits = cfg.args['num_qubits']
        print("Using num_qubits={} from command line".format(cfg.num_qubits))
    if len(cfg.args['output_dir']) > 0:
        cfg.output_dir = cfg.args['output_dir']
        print("Using output_dir '{}' from command line".format(
                                                        cfg.output_dir))
    if cfg.args['num_cpus'] > 0:
        cfg.num_cpus = cfg.args['num_cpus']
        print("Using num_cpus={} from command line".format(cfg.num_cpus))

    if len(cfg.args['mp_opt']) > 0:
        cfg.mp_opt = cfg.args['mp_opt']
        print("Using mp_opt={} from command line".format(cfg.mp_opt))

    cfg.plot_file_ext = cfg.plot_file_type.lower()
    if cfg.args['job_id'] > 0:
        cfg.job_id = cfg.args['job_id']

    if cfg.job_id:
        print("Processing job ID: {}".format(cfg.job_id))
        if cfg.args['job_idx'] > 0:
            cfg.job_idx = cfg.args['job_idx']
        if cfg.job_idx:
            print("Processing job array index: {}".format(cfg.job_idx))
            cfg.out_file_ext = "{}.{}.{}".format(cfg.job_id, cfg.job_idx,
                                                    cfg.data_file_ext)
            cfg.plot_file_ext = "{}.{}.{}".format(cfg.job_id, cfg.job_idx,
                                                    cfg.plot_file_type)
        else:
            cfg.out_file_ext = "{}.{}".format(cfg.job_id, cfg.data_file_ext)
            cfg.plot_file_ext = "{}.{}".format(cfg.job_id, cfg.plot_file_type)

    logger.setLevel(cfg.log_level)
    if not cfg.output_dir:
        cfg.output_dir = cfg.target_id_text

    return cfg

def gen_optim_objects(cfg):
    """
    Generate the optimiser objects based on the configuration

    Returns
    -------
    Optimizer
    """

    dyn = dynamics.DynamicsUnitary(cfg)
    dyn.dense_oper = True
    dyn.memory_optimization = 1

    # Number of timeslots for the pulse discretisation
    dyn.num_tslots = 12
    # For a range of evo_time based on job_idx (cfg.idx_opt='num_tslots')
    # For a specific list of num_tslots values based on job_idx
    dyn.num_tslots_list = []
    # Or generate a range
    dyn.st_num_tslots = 10
    dyn.d_num_tslots = 5
    dyn.num_num_tslots = 10

    # Time allowed for the gate to evolve
    dyn.evo_time = 10.0
    # For a range of evo_time based on job_idx (cfg.idx_opt='evo_time'):
    # For a specific list of evo_time values based on job_idx
    dyn.evo_time_list = [7, 10, 12, 20]
    # Or generate a range
    dyn.st_evo_time = 0.2
    dyn.d_evo_time = 0.2
    dyn.num_evo_time = 50

    # Any number of qubits >= 3 can choosen.
    # However larger numbers take a lot of processing
    dyn.num_qubits = 3

    # Only CNOT implemented for target_type
    dyn.target_type = 'CNOT'
    # interact options: Ising|Heisenberg| or combinations of xyz
    dyn.interact = 'Ising'
    # topology options: chain|star|full|ring
    dyn.topology = 'chain'
    # ctrls_type options: combinations of xyz
    dyn.ctrls_type = 'XY'
    # If True then same coupling constant will be used for
    # all interactions between qubit pairs
    dyn.iso_coup = True
    # Coupling strength for each interaction
    # If a single float list, then this will be used for all interactions
    # Otherwise
    # If iso_coup=True then should be one list item per iteracting pair
    # If iso_coup=False then should be blocks of list items per iteracting pair
    # e.g. for 4 qubit Heisenberg chain:
    # [g_0x, g_1x, g_2x, g_0y, g_1y, g_2y, g_0z, g_1z, g_2z]
    # Note that the number of interacting pairs is n-1 for chain and star,
    # n for ring, and n*(n-1)/2 for fully connected

    dyn.coup_const = [1.0]
    # Used to change the order of the qubits in the Hilbert space
    # The two qubit gate will always be between 0 and 1,
    # but the couplings will be permuted based on hspace_order
    dyn.hspace_order = []
    # When True the hspace_order will be automatically generated
    # based on the other attributes
    dyn.auto_hspace = False
    # Separation of the first and second qubits in the coupling Hilbert space
    # 0 implies adjacent, -1 implies random
    dyn.hspace_01_sep = 0
    # Index of the first qubit in the coupling Hilbert space
    # -1 implies random
    dyn.hspace_0_idx = 0


    # dynamical decoup attribs
    # These are the decoup amplitudes in the three axes
    dyn.decoup_x = 0.0
    dyn.decoup_y = 0.0
    dyn.decoup_z = 0.0
    # num_decoup_tslots = None means use the specific mask
    # num_decoup_tslots = 0 use all
    # num_decoup_tslots +ve is the number of tslots from start
    # num_decoup_tslots -ve is the number of zeros at the end
    dyn.num_decoup_tslots = 0
    dyn.decoup_tslots = []
    #dyn.decoup_tslot_mask = []

    if cfg.use_param_file:
        # load the dynamics parameters
        # note these will overide those above if present in the file
        print("Loading dynamics parameters from {}".format(cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, dynamics=dyn)

    if cfg.args['num_qubits'] > 0:
        dyn.num_qubits = cfg.args['num_qubits']
        print("num_qubits = {} from command line".format(dyn.num_qubits))

    # If job_idx is given, then use it to calculate some other parameter
    # (which supersedes all other settings)
    if cfg.job_idx and len(cfg.args['idx_opt']) > 0:
        cfg.ext_mp = True
        opt_str = cfg.args['idx_opt'].lower()
        if opt_str == 'evo_time':
            print("basing evo_time on job_idx... ")
            num_evo_time = len(dyn.evo_time_list)
            if num_evo_time > 0:
                print("...using evo_time_list")
                # get the evo time from the list
                T_idx = (cfg.job_idx - 1) % num_evo_time
                dyn.evo_time = dyn.evo_time_list[T_idx]
            else:
                print("...using start and increment")
                # calculate the evo_time from job_idx
                T_idx = (cfg.job_idx - 1) % dyn.num_evo_time
                dyn.evo_time = dyn.st_evo_time + dyn.d_evo_time*float(T_idx)
            print("evo_time={} for job idx {}".format(dyn.evo_time,
                                                      cfg.job_idx))
        elif opt_str == 'num_tslots':
            print("basing num_tslots on job_idx... ")
            num_nts = len(dyn.num_tslots_list)
            if num_nts > 0:
                print("...using num_tslots_list")
                nts_idx = (cfg.job_idx - 1) % num_nts
                dyn.num_tslots = dyn.num_tslots_list[nts_idx]
            else:
                print("...using start and increment")
                # calculate the num_tslots from job_idx
                nts_idx = (cfg.job_idx - 1) % dyn.num_num_tslots
                dyn.num_tslots = dyn.st_num_tslots + dyn.d_num_tslots*nts_idx
            print("num_tslots={} for job idx {}".format(dyn.num_tslots,
                                                      cfg.job_idx))
        else:
            raise ValueError("No option for idx_opt '{}' "
                            "in command line".format(opt_str))

    if cfg.args['evo_time'] > 0:
        dyn.evo_time = cfg.args['evo_time']
        print("Using evo_time={} from command line".format(dyn.evo_time))

    if cfg.args['evo_time_npi'] > 0:
        dyn.evo_time = np.pi*cfg.args['evo_time_npi']
        print("Using evo_time={} from command line evo_time_npi".format(
                                                        dyn.evo_time))
    if cfg.args['evo_time_npi_g0s'] > 0:
        dyn.evo_time = np.pi*cfg.args['evo_time_npi_g0s']/dyn.coup_const[0]
        print("Using evo_time={} from command line evo_time_npi_g0s".format(
                                                        dyn.evo_time))

    if cfg.args['num_tslots'] > 0:
        dyn.num_tslots = cfg.args['num_tslots']
        print("Using num_tslots={} from command line".format(dyn.num_tslots))

    if cfg.args['mem_opt'] > 0:
        dyn.memory_optimization = cfg.args['mem_opt']
        print("Using mem_opt={} from command line".format(dyn.memory_optimization))

    print("evo_time={}".format(dyn.evo_time))
    print("num_tslots={}".format(dyn.num_tslots))

    if dyn.num_tslots == 0:
        dyn.num_tslots = dyn.num_qubits*4 + 8
        print("num_tslots calculated and set to be {}".format(dyn.num_tslots))

    if len(dyn.coup_const) == 1:
        dyn.coup_const = dyn.coup_const[0]

    dyn.prop_computer = propcomp.PropCompFrechet(dyn)

    if dyn.dense_oper:
        dyn.oper_dtype = np.ndarray
    else:
        dyn.oper_dtype = Qobj

    # Pulse optimisation termination conditions
    # Create the TerminationConditions instance
    tc = termcond.TerminationConditions()
    # Target for the infidelity (1 - gate fidelity)
    tc.fid_err_targ = 1e-3
    # Sum of the gradients wrt optimisation parameters (timeslot amplitudes)
    tc.min_gradient_norm = 1e-30
    # Number of iterations of the algorithm
    tc.max_iter = 2400
    # Computation time (in seconds)
    tc.max_wall_time = 120*60.0
    # Relative change in fid_err between iterations
    tc.accuracy_factor = 1e5

    if cfg.use_param_file:
        # load the termination condition parameters
        # note these will overide those above if present in the file
        print("Loading termination condition parameters from {}".format(
                cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, term_conds=tc)
    if cfg.args['fid_err_targ'] > 0.0:
        tc.fid_err_targ = cfg.args['fid_err_targ']
        print("fid_err_targ = {} from command line".format(tc.fid_err_targ))
    if cfg.args['max_wall_time'] > 0.0:
        tc.max_wall_time = cfg.args['max_wall_time']
        print("max_wall_time = {} from command line".format(tc.max_wall_time))

    # Fidelity oomputer
    ft = cfg.fid_type.lower()
    if ft == 'pure_choi_local':
        dyn.fid_computer = FidCompPureChoiLocal(dyn)
    elif ft == 'pure_choi_global':
        dyn.fid_computer = FidCompPureChoiGlobal(dyn)
    elif ft == 'unit_global':
        dyn.fid_computer = fidcomp.FidCompUnitary(dyn)
        dyn.fid_computer.local = False
    else:
        raise errors.UsageError("Unknown fid type {}".format(cfg.fid_type))
    fid_comp = dyn.fid_computer
    fid_comp.numer_acc = 0.0
    fid_comp.numer_acc_exact = False
    fid_comp.st_numer_acc = 0.01
    fid_comp.end_numer_acc = 0.2
    fid_comp.success_prop_uthresh = 0.95
    fid_comp.success_prop_lthresh = 0.01

    if cfg.use_param_file:
        # load the pulse generator parameters
        # note these will overide those above if present in the file
        print("Loading fidcomp parameters from {}".format(cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, obj=dyn.fid_computer,
                                    section='fidcomp')

    if cfg.args['numer_acc'] >= 0.0:
        fid_comp.numer_acc = cfg.args['numer_acc']
        print("numer_acc = {} from command line".format(fid_comp.numer_acc))

    if not fid_comp.numer_acc_exact:
        fid_comp.numer_acc = round_sigfigs(
                                fid_comp.numer_acc*tc.fid_err_targ, 6)
        fid_comp.st_numer_acc = round_sigfigs(
                                fid_comp.st_numer_acc*tc.fid_err_targ, 6)
        fid_comp.end_numer_acc = round_sigfigs(
                                fid_comp.end_numer_acc*tc.fid_err_targ, 6)

    # Pulse generator
    p_gen = pulsegen.create_pulse_gen(pulse_type=cfg.p_type, dyn=dyn)
    p_gen.all_ctrls_in_one = False
    p_gen.lbound = cfg.amp_lbound
    p_gen.ubound = cfg.amp_ubound
    if cfg.use_param_file:
        # load the pulse generator parameters
        # note these will overide those above if present in the file
        print("Loading pulsegen parameters from {}".format(cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, pulsegen=p_gen)

    if not isinstance(p_gen, pulsegen.PulseGenCrab):
        if not (np.isinf(cfg.amp_lbound) or np.isinf(cfg.amp_ubound)):
            p_gen.scaling = cfg.amp_ubound - cfg.amp_lbound
            p_gen.offset = (cfg.amp_ubound + cfg.amp_lbound) / 2.0

    if cfg.args['pulse_scaling'] > 0.0:
        p_gen.scaling = cfg.args['pulse_scaling']
        print("p_gen.scaling = {} from command line".format(p_gen.scaling))

    # Create the Optimiser instance
    if cfg.optim_method is None:
        raise errors.UsageError("Optimisation method must be specified "
                                "via 'optim_method' parameter")

    om = cfg.optim_method.lower()
    if om == 'bfgs':
        optim = optimizer.OptimizerBFGS(cfg, dyn)
    elif om == 'lbfgsb':
        optim = optimizer.OptimizerLBFGSB(cfg, dyn)
    else:
        # Assume that the optim_method is valid
        optim = optimizer.Optimizer(cfg, dyn)
        optim.method = cfg.optim_method
    if cfg.verbosity > 1:
        print("Created optimiser of type {}".format(type(optim)))

    optim.config = cfg
    optim.dynamics = dyn
    optim.termination_conditions = tc
    optim.pulse_generator = p_gen
    optim.method_approach = 'DEF'
    optim.dumping = 'SUMMARY'

    if cfg.use_param_file:
        # load the optimiser parameters
        # note these will overide those above if present in the file
        print("Loading optimiser parameters from {}".format(cfg.param_fpath))
        loadparams.load_parameters(cfg.param_fpath, optim=optim)
    optim.termination_conditions = tc

    if cfg.gen_stats:
        # Create a stats object
        # Note that stats object is optional
        # if the Dynamics and Optimizer stats attribute is not set
        # then no stats will be collected, which could improve performance
        stats_type = 'standard'
        try:
            stats_type = cfg.stats_type.lower()
        except:
            pass
        if stats_type == 'local':
            sts = qsostats.StatsFidCompLocal()
        else:
            sts = stats.Stats()
        dyn.stats = sts
        optim.stats = sts

    return optim

def config_dynamics(dyn):
    """
    Configure the dynamics generators and targets etc
    """

    # ****************************************************************
    # Define the physics of the problem
    fid_comp = dyn.fid_computer

    print("Configuring drift...")

    nq = dyn.num_qubits
    # ***** Drift *****
    print("... for {} qubits".format(nq))

    H_d = qso.get_drift(dyn)
    print("Drift dims {}".format(H_d.dims))

    # Normalise based on ising chain
    H_d_ising_chain = qso.get_drift(dyn, topology='chain', interact='ising',
                                        coup_const=1.0)
    norm_fact = H_d_ising_chain.norm() / H_d.norm()
    print("Normalising drift with factor {}".format(norm_fact))
    H_d = H_d*norm_fact

    # **** Controls ****
    H_c, Sx_cidx, Sy_cidx, Sz_cidx = qso.get_ctrls(dyn)
    n_ctrls = len(H_c)

    #t0 evo
    U_0 = qso.get_initial_op(dyn)

    #*** Target ****
    U_targ, U_local_targs = qso.get_target(dyn)

    sub_dims = []
    for k in range(len(U_local_targs)):
        sub_dims.append(U_local_targs[k].dims[0][0])

    #Enforcing all dimensions are correct for the qBranch code
    #Not all of these are needed, but this is safer
    for k in range(len(H_c)):
        H_c[k].dims = [sub_dims, sub_dims]
    H_d.dims = [sub_dims, sub_dims]
    U_0.dims = [sub_dims, sub_dims]
    U_targ.dims = [sub_dims, sub_dims]

    dyn.drift_dyn_gen = H_d
    dyn.ctrl_dyn_gen = H_c
    dyn.initial = U_0
    dyn.target = U_targ

    # These are the indexes to the controls on the three axes
    dyn.Sx_cidx = Sx_cidx
    dyn.Sy_cidx = Sy_cidx
    dyn.Sz_cidx = Sz_cidx

    fid_comp.U_local_targs = U_local_targs
    fid_comp.sub_dims = sub_dims
    fid_comp.num_sub_sys = len(U_local_targs)

    # print("Num acc: {}".format(fid_comp.numer_acc))

