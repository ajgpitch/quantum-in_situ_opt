# -*- coding: utf-8 -*-
"""
Generic functions for the quantum self optimisation.
In particular there are the functions that build the dynamics generators
and targets.
"""

# started 2016 Aug 1 by Alexander Pitchford
# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

import sys
import numpy as np
import random
import collections
from functools import reduce
from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor

def get_cfg_str(optim, full=False, num_tslots=None, evo_time=None,
                     fid_err_targ=None, numer_acc=None):
    """Build a string to be included in file names"""
    cfg = optim.config
    dyn = optim.dynamics
    tc = optim.termination_conditions
    fid_comp = dyn.fid_computer
    if full:
        num_tslots = dyn.num_tslots
        evo_time = dyn.evo_time
        fid_err_targ = tc.fid_err_targ
        numer_acc = fid_comp.numer_acc

    cfg_str = "{}-{}-{}-{}-{}-{}-nq{}".format(
                    cfg.output_base_name, cfg.fid_type,
                    dyn.topology, dyn.interact,
                    dyn.ctrls_type, dyn.target_type,
                    dyn.num_qubits)

    if evo_time is not None:
        cfg_str = "{}-T{:0.2f}".format(cfg_str, evo_time)
    if num_tslots is not None:
        cfg_str = "{}-nts{:d}".format(cfg_str, num_tslots)
    if fid_err_targ is not None:
        cfg_str = "{}-fet{:0.2e}".format(cfg_str, fid_err_targ)
    if numer_acc is not None:
        cfg_str = "{}-na{:0.2e}".format(cfg_str, numer_acc)

    return cfg_str

def get_out_file_ext(data_file_ext, job_id=None, scen_idx=None, reps_idx=None):
    """
    Build a multi part file extension for use with output files.
    This is particular useful when using an external job scheduler,
    to ensure that file names are unique and can be linked to processing
    records.
    """
    out_file_ext = data_file_ext
    if reps_idx is not None:
        out_file_ext = "r{}.{}".format(reps_idx, out_file_ext)
    if scen_idx is not None:
        out_file_ext = "s{}.{}".format(scen_idx, out_file_ext)
    if job_id is not None:
        out_file_ext = "j{}.{}".format(job_id, out_file_ext)

    return out_file_ext

def get_coupling_hspace(num_qubits, idx0, sep):
    """
    Generate the coupling Hilbert space order

    Parameters
    ----------
    num_qubits : int
        number of qubits in the system
    idx0 : int
        Index of the first qubit in the coupling Hilbert space
        -1 implies random
    sep : int
        Separation of the first and second qubits in the coupling Hilbert space
       0 implies adjacent, -1 implies random
    """

    # Not sure this is fully robust, but

    if num_qubits < 3:
        raise ValueError("Cannot generate hspace order for "
                         "{} qubits".format(num_qubits))
    if sep > num_qubits - 2:
        raise ValueError("Cannot separate {} qubits by "
                         "{}".format(num_qubits, sep))

    if idx0 >= num_qubits - 1:
        raise ValueError("First qubit index must be less than the number of "
                         "qubits less 1".format(num_qubits, idx0))

    #idx1 = -1
    if sep < 0:
        sep = random.randint(0, num_qubits - 2)
#        if idx0 < 0:
#
#            idx1 = random.randint(0, num_qubits - 1)
#        else:
#            sep = random.randint(0, num_qubits - idx0 - 2)

    if idx0 < 0:
        idx0 = random.randint(0, num_qubits - 2)

    idx1 = idx0 + sep + 1
    if idx1 >= num_qubits:
        idx1 -= num_qubits
#        raise ValueError("Cannot separate by {} for idx0 {} and "
#                         "{} qubits".format(sep, idx0, num_qubits))

    hso = list(range(2, num_qubits))
    hso.insert(idx0, 0)
    hso.insert(idx1, 1)

    return hso

def get_drift(dyn, num_qubits=None, topology=None, interact=None,
              coup_const=None, hspace_order=None, verbosity=None):
    """Build the drift operator based on the topology and interaction type"""
    if verbosity is None:
        verbosity = dyn.config.verbosity
    def printv(msg, verb_tresh=1):
        if verbosity >= verb_tresh:
            print(msg)

    if num_qubits is None:
        nq = dyn.num_qubits
    else:
        nq = num_qubits
    if topology is None:
        topology = dyn.topology
    if interact is None:
        interact = dyn.interact
    if coup_const is None:
        coup_const = dyn.coup_const
    if hspace_order is None:
        hspace_order = dyn.hspace_order

    if len(hspace_order) == 0:
        reorder = False
    else:
        if len(hspace_order) != nq:
            raise ValueError("hspace_order must have len equal to "
                             "num_qubits ({})".format(nq))
        if sorted(hspace_order) != list(range(nq)):
            raise ValueError("Invalid hspace_order: {}".format(hspace_order))
        reorder = True

    Sx = sigmax()
    Sy = sigmay()
    Sz = sigmaz()
    Si = identity(2)
    nq = dyn.num_qubits

    # Set coupling parameter
    inter = interact.lower()
    if inter == 'ising':
        printv("using Ising interactions")
        S_interacts = [Sz]
    elif inter == 'heisenberg':
        printv("using Heisenberg interactions")
        S_interacts = [Sx, Sy, Sz]
    elif len(inter) <= 3:
        printv("using custom interactions...")
        S_interacts = []
        for i in inter:
            if i == 'x':
                printv("...adding Sx interactions")
                S_interacts.append(Sx)
            elif i == 'y':
                printv("...adding Sy interactions")
                S_interacts.append(Sy)
            elif i == 'z':
                printv("...adding Sz interactions")
                S_interacts.append(Sz)
        if len(S_interacts) == 0:
            raise ValueError(
                "No interactions in interact '{}'".format(interact))
    else:
        raise ValueError("No option for interact '{}'".format(interact))

    dim0 = [2 for i in range(nq)]
    n_Hs = 2**nq
    H_d = Qobj(inpt=np.zeros([n_Hs, n_Hs]), dims=[dim0, dim0])
    j = 0
    if topology.lower() == 'chain':

        if nq < 2:
            raise ValueError(
                "Cannot build chain with {} qubits.".format(nq))

        ng = nq-1
        if not dyn.iso_coup:
            ng *= len(S_interacts)
        g = get_coup_const_list(coup_const, ng)
        for Sc in S_interacts:
            if dyn.iso_coup:
                j = 0
            spins = collections.deque([Sc, Sc])
            for q in range(nq-2):
                spins.append(Si)

            for q in range(nq-1):
                if reorder:
                    opers = [spins[i] for i in hspace_order]
                else:
                    opers = spins
                H_d += g[j]*tensor(*opers)
                spins.rotate()
                j += 1

    elif topology.lower() == 'star':
        if nq < 3:
            raise ValueError(
                "Cannot build star with {} qubits.".format(nq))
        ng = nq-1
        if not dyn.iso_coup:
            ng *= len(S_interacts)
        g = get_coup_const_list(coup_const, ng)
        for Sc in S_interacts:
            if dyn.iso_coup:
                j = 0
            sats = collections.deque([Sc])
            for q in range(nq-2):
                sats.append(Si)

            for q in range(nq-1):
                spins = [Sc]
                spins.extend(sats)
                H_d += g[j]*tensor(*spins)
                sats.rotate()
                j += 1

    elif topology.lower() == 'full':
        ng = nq*(nq-1) // 2
        if not dyn.iso_coup:
            ng *= len(S_interacts)
        g = get_coup_const_list(coup_const, ng)

        for Sc in S_interacts:
            if dyn.iso_coup:
                j = 0
            for i in range(nq-1):
                for k in range(i+1, nq):
                    spins = [Si for q in range(nq)]
                    spins[i] = Sc
                    spins[k] = Sc
                    H_d += g[j]*tensor(*spins)
                    j += 1

    elif topology.lower() == 'ring':

        if nq < 3:
            raise ValueError(
                "Cannot build ring with {} qubits.".format(nq))

        ng = nq
        if not dyn.iso_coup:
            ng *= len(S_interacts)
        g = get_coup_const_list(coup_const, ng)
        for Sc in S_interacts:
            if dyn.iso_coup:
                j = 0
            spins = collections.deque([Sc, Sc])
            for q in range(nq-2):
                spins.append(Si)

            for q in range(nq):
                if reorder:
                    opers = [spins[i] for i in hspace_order]
                else:
                    opers = spins
                H_d += g[j]*tensor(*opers)
                spins.rotate()
                j += 1

    else:
        raise ValueError("No option for topology '{}'".format(topology))

    return H_d

def get_ctrls(dyn):
    """
    Build the controls list
    Add local control on all qubits
    Combinations of Sx, Sy, Sz are added based on the ctrls_type
    attribute of the dynamics parameter.

    Returns
    -------
    Full list of control operators
    Lists of indexes to the Sx, Sy, Sz controls respectively
    """

    Sx = sigmax()
    Sy = sigmay()
    Sz = sigmaz()
    Si = identity(2)
    nq = dyn.num_qubits

    def append_ctrl(ctrl, ctrl_list, idx_list):
        ctrl_list.append(ctrl)
        idx_list.append(len(ctrl_list) - 1)

    ct = dyn.ctrls_type.lower()
    H_c = list()
    xc = list()
    yc = list()
    zc = list()

    if 'x' in ct:
        spx = collections.deque([Sx])
        for q in range(nq-1):
            spx.append(Si)
        for q in range(nq):
            append_ctrl(reduce(tensor, spx), H_c, xc)
            spx.rotate()
    if 'y' in ct:
        spy = collections.deque([Sy])
        for q in range(nq-1):
            spy.append(Si)
        for q in range(nq):
            append_ctrl(reduce(tensor, spy), H_c, yc)
            spy.rotate()
    if 'z' in ct:
        spz = collections.deque([Sz])
        for q in range(nq-1):
            spz.append(Si)
        for q in range(nq):
            append_ctrl(reduce(tensor, spz), H_c, zc)
            spz.rotate()

    return H_c, xc, yc, zc

def get_initial_op(dyn):
    """Build the starting operator for the evo"""
    Si = identity(2)
    nq = dyn.num_qubits
    U_0 = reduce(tensor, [Si for q in range(nq)])
    return U_0

def get_target(dyn):
    """Build the target - cNOT on first 2 qubits"""
    Si = identity(2)
    nq = dyn.num_qubits

    cNOT = Qobj([[1,0,0,0],\
                [0,1,0,0],\
                [0,0,0,1],\
                [0,0,1,0]])

    U_local_targs = [cNOT]
    for q in range(nq-2):
        U_local_targs.append(Si)

    #Turn local information above into objects needed
    #No need to ever edit any of this
    U_targ = reduce(tensor, U_local_targs)

    return U_targ, U_local_targs

def gen_rnd_coup_consts(n, lb, ub):
    """Generate a list of random numbers (lb, ub]"""
    w = ub - lb
    coup_consts = np.random.random([n])*w + lb
    return coup_consts.tolist()

def gen_rnd_aniso_coup_const(num_coup, side_coup=0.1):
    """
    Generate coupling constants for the anisotropic couplings
    for use with the Heisenberg topology.
    For each coupling
    Z couplings of 1.0
    either X or Y of side_coup
    """
    z = [1.0 for i in range(num_coup)]
    x = [random.choice([0, 1]) for i in range(num_coup)]
    y = [(not x[i])*side_coup for i in range(num_coup)]
    return [x[i]*side_coup for i in range(num_coup)] + y + z

def get_coup_const_list(coup_const, n):
    """return list of n coupling consts for the interactions"""
    if isinstance(coup_const, list):
        try:
            g = coup_const[:n]
        except IndexError:
            raise IndexError("Insufficient coupling constants in list")
    else:
        g = [coup_const for j in range(n)]

    return g

def print_phys_params(optim, f=sys.stdout):
    """ Print physical params """

    tc = optim.termination_conditions
    dyn = optim.dynamics
    tc = optim.termination_conditions
    fid_comp = dyn.fid_computer

    f.write("\nPhysical Parameters:\n")
    coup = "rnd"
    if isinstance(dyn.coup_const, float):
        coup = "const"
    f.write("{} qubit {} {} with {} couplings\n".format(dyn.num_qubits,
                                             dyn.interact, dyn.topology, coup))
    f.write("with a Hilbert space order of: {}\n".format(dyn.hspace_order))
    f.write("ctrls = {}, target = {}\n".format(dyn.ctrls_type, dyn.target_type))

    """ f.write search parameters """
    f.write("\nEvolution Parameters:\n")
    f.write("n_ts = {}, evo_time = {}\n".format(dyn.num_tslots, dyn.evo_time))
    f.write("\nOptim Parameters:\n")
    f.write("type: {}, method: {}, amp_lbound = {}, amp_ubound = {}\n".format(
                            optim.__class__.__name__, optim.method,
                            optim.amp_lbound, optim.amp_ubound))
    f.write("\nSearch Parameters:\n")
    f.write("max_iter = {}, max_wall_time = {}, min_grad = {}, "
            "acc_factor = {}, numer_acc = {}\n".format(
                    tc.max_iterations, tc.max_wall_time, tc.min_gradient_norm,
                    tc.accuracy_factor, fid_comp.numer_acc))
    f.write("\nTarget information:\n")
    f.write("fid_err_targ = {},\nlocal_targs =\n{}\n".format(tc.fid_err_targ,
                                              fid_comp.U_local_targs))

