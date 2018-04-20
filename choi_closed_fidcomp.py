"""
Custom FidelityComputer class for computing global and local Choi fidelities
"""

# started 2015 Nov 3 by Ben Dive
# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

# Python stardard libraries
import numpy as np
import timeit

from functools import reduce
# QuTiP
import qutip.logging_utils as logging
from qutip import Qobj, identity, tensor
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.fidcomp as fidcomp
# local imports
from ptrace import partial_trace, calc_perm
from qsostats import StatsFidCompLocal

def my_round(val, numer_acc):
    """
    Round value to specified numerical accuracy.

    Built in functions only round to nearest decimal place
    want a function that can round to a given accuracy.
    """
    if numer_acc == 0:
        return val
    else:
        return np.round(val / numer_acc) * numer_acc


class FidCompPureChoiGlobal(fidcomp.FidCompUnitary):
    """
    Uses the choi fidelity for a unitary target and unitary evolution.
    Equivalent to the modsquare of the usual gate fidelity.

    Attributes
    ----------
    numer_acc  : float
        Numerical accuracy. Fidelity will be rounded to this level
        The default 0 means no rounding
    """
    def reset(self):
        self.conversion_time = 0
        self.numer_acc = 0
        fidcomp.FidCompUnitary.reset(self)

    def set_phase_option(self, phase_option=None):
        """
        Set normalisation functions
        Phase options are not longer important
        Keeping the self.dimensional_norm as it is, just use the
        square of it for normalisation.
        """
        self.fid_norm_func = self.choi_norm
        self.grad_norm_func = self.choi_norm_grad

    def choi_norm(self, A):
        """
        Normalise the fidelity based on target's matrix dimensions
        Also apply the numerical accuracy rounding
        """
        if isinstance(A, Qobj):
            norm = A.tr()
        elif isinstance(A, np.ndarray):
            norm = np.trace(A)
        else:
            norm = A
        return my_round(np.real(norm) / self.dimensional_norm**2, self.numer_acc)

    def choi_norm_grad(self, A):
        return my_round(np.real(A) / self.dimensional_norm**2, self.numer_acc)


    def get_fidelity_prenorm(self):
        """
        Gets the current fidelity value prior to normalisation
        Note the gradient function uses this value
        The value is cached, because it is used in the gradient calculation
        """
        if not self.fidelity_prenorm_current:
            dyn = self.parent
            if self.log_level <= logging.DEBUG:
                logger.debug("**** Computing Choi fidelity ****")
            k = dyn.tslot_computer._get_timeslot_for_fidelity_calc()
            dyn.compute_evolution()
            # **** CUSTOMISATION starts here *****
            if dyn.oper_dtype == Qobj:
                f_half = (dyn._onto_evo[k]*dyn._fwd_evo[k]).tr()
            else:
                f_half = np.trace(dyn._onto_evo[k].dot(dyn._fwd_evo[k]))

            #f_half = (dyn._fwd_evo[k] * dyn._onto_evo[k]).tr()
            #Writem this way to make it easier to generalise
            #to local estimator later
            f = f_half * np.conjugate(f_half)
            # **** END OF CUSTOMISATION ****
            self.fidelity_prenorm = f
            self.fidelity_prenorm_current = True
            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (pre normalisation): {}".format(
                    self.fidelity_prenorm))

        return self.fidelity_prenorm

    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot
        control amplitudes. Note these gradients are not normalised
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots

        if self.log_level <= logging.DEBUG:
            logger.debug("**** Computing Choi fidelity gradient ****")

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()
        for j in range(n_ctrls):
            for k in range(n_ts):
                owd_evo = dyn._onto_evo[k+1]
                fwd_evo = dyn._fwd_evo[k]
                # **** CUSTOMISE starts here *****
                # Writen this way to make it easier to generalise
                # to local estimator later
                if dyn.oper_dtype == Qobj:
                    #to local estimator later
                    g_half = (owd_evo * dyn.prop_grad[k, j] * fwd_evo).tr()
                    #The same half fidelity as above
                    f_half = (fwd_evo * dyn._onto_evo[k]).tr()
                else:
                    f_half = np.trace(dyn._onto_evo[k].dot(dyn._fwd_evo[k]))
                    g_half = np.trace(
                                owd_evo.dot(dyn.prop_grad[k, j]).dot(fwd_evo))

                g = g_half*np.conjugate(f_half) + np.conjugate(g_half)*f_half

                grad[k, j] = np.real(g)
                # **** END OF CUSTOMISATION ****

        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st

        return grad


class FidCompPureChoiLocal(fidcomp.FidCompUnitary):
    """
    Calculates the estimator of the fidelity (and gradient) based on the
    sums of the local fidelities. This requires more inputs than just the
    global targets - it also requires the local targets.

    Attributes
    ----------
    U_local_targs : list of Qobj
        The local targets. This could be a gate on one of more qubits
        or the identity on single qubits.

    sub_dims : List of int
        dimensions of the U_local_targs
        Assumes all local targets are unitary ops (hence square matrices)
    num_sub_sys : int
        number of sub systems
    numer_acc  : float
        Numerical accuracy. Fidelity will be rounded to this level
        The default 0 means no rounding
    """

    def reset(self):
        self.numer_acc = 0 #0 = no rounding (see definition at top)
        #Do in this order so apply_params over rules default
        self.local = True
        fidcomp.FidCompUnitary.reset(self)

        #Change what part of the backwards evolution it stores
        self.uses_onwd_evo = True
        self.uses_onto_evo = False

        self.fid_norm_func = self.already_normalised
        self.grad_norm_func = self.already_normalised

    def apply_params(self, params=None):
        #pre existing code...#
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    def init_comp(self):
        """
        Initialise the object after the configuration has been made,
        before the optimisation starts.
        """
        # Convenient to have extra attributes calculated once,
        # this is best place to put them

        dyn = self.parent
        #and now the new code
        self.full_dim = np.product(self.sub_dims)
        self.dimensional_norm = self.full_dim
        self.oper_dims = [self.sub_dims, self.sub_dims]

        #Simply the local targets with the relevent number of identities tensored
        # before and after it.
        # We only ever need the .dag() of this, so only storing that.
        self.large_local_targs_dag = []
        for sub_sys in range(self.num_sub_sys):
            large_local_targ = []
            for k in range(sub_sys):
                large_local_targ.append(identity(self.sub_dims[k]))
            large_local_targ.append(self.U_local_targs[sub_sys])
            for k in range(sub_sys+1,self.num_sub_sys):
                large_local_targ.append(identity(self.sub_dims[k]))
            large_local_targ = reduce(tensor, large_local_targ)

            if dyn.oper_dtype == Qobj:
                self.large_local_targs_dag.append(large_local_targ.dag())
            else:
                self.large_local_targs_dag.append(
                    large_local_targ.dag().full())

        # Calculate the permutation matrices for the partial trace
        self.ptrace_perms = {}
        for sub_sys in range(self.num_sub_sys):
            self.ptrace_perms[sub_sys] = calc_perm(self.oper_dims, sub_sys)


    def already_normalised(self, arg):
        # Normalisation dealt with elsewhere, cannot be postponed till end
        return np.real(arg)

    def get_fidelity_prenorm(self):
        """
        Get the current fidelity value prior to normalisation
        Note the gradient function uses this value
        The value is cached, because it is used in the gradient calculation
        """
        if not self.fidelity_prenorm_current:
            dyn = self.parent
            #k = dyn.tslot_computer._get_timeslot_for_fidelity_calc()
            dyn.compute_evolution()
            """Also save the local pseudo fidelities to save time"""
            f = self.compute_total_fid()
            self.fidelity_prenorm = f
            self.fidelity_prenorm_current = True
            if dyn.stats is not None:
                    dyn.stats.num_fidelity_computes += 1
            if self.log_level <= logging.DEBUG:
                logger.debug("Fidelity (pre normalisation): {}".format(
                    self.fidelity_prenorm))
        return self.fidelity_prenorm

    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot
        control amplitudes. Note these gradients **** are ****
        These are returned as a (nTimeslots x n_ctrls) array
        """
        dyn = self.parent
        n_ctrls = dyn.num_ctrls
        n_ts = dyn.num_tslots

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        dyn.tslot_computer.flag_all_calc_now()
        dyn.compute_evolution()

        # loop through all ctrl timeslots calculating gradients
        time_st = timeit.default_timer()
        #Calculate the pseudo_fid's here, so it only has to be done once
        pseudo_fids_dag = self.compute_all_pseudo_fids(dag=True)
        #Change order of looping to save some time
        for k in range(n_ts):

            #Slight modification due to using _onwd_evo
            #rather than evot2targ
            if k + 1 < n_ts:
                target_less_owd_evo = dyn._onwd_evo[k+1]
            else:
                if dyn.oper_dtype == Qobj:
                    target_less_owd_evo = identity(self.full_dim)
                    target_less_owd_evo.dims = [self.sub_dims, self.sub_dims]
                else:
                    target_less_owd_evo = np.eye(self.full_dim)
                    #target_less_owd_evo_dims = [self.sub_dims, self.sub_dims]

            fwd_evo = dyn._fwd_evo[k]
            owd_local_targs = self.compute_all_owd_local_target(target_less_owd_evo)
            for j in range(n_ctrls):
                prop_grad = dyn._get_prop_grad(k, j)
                total_fid_grad = 0
                if dyn.oper_dtype == Qobj:
                    prop_grad_fwd_evo = prop_grad * fwd_evo
                else:
                    prop_grad_fwd_evo = prop_grad.dot(fwd_evo)
                for sub_sys in range(self.num_sub_sys):
                    total_fid_grad += self.compute_local_fid_grad(sub_sys,
                        owd_local_targs[sub_sys], prop_grad_fwd_evo, pseudo_fids_dag[sub_sys])
                grad[k, j] = total_fid_grad
        if dyn.stats is not None:
            dyn.stats.wall_time_gradient_compute += \
                timeit.default_timer() - time_st
        return grad


    def _ptrace(self, sys, sel):
        if isinstance(sel, int):
            perm = self.ptrace_perms[sel]
        elif len(sel) == 1:
            perm = self.ptrace_perms[sel[0]]
        else:
            perm=self.ptrace_perms.get(str(sel), None)
            if perm is None:
                perm = calc_perm(self.oper_dims, sel)
                self.ptrace_perms[str(sel)] = perm

        return partial_trace(sys, self.oper_dims, sel, perm=perm)[0]

    # New fidelity functions pass large arrays to each other,
    # but there are fewer Qobj conversations and fewer repetitions.
    # The whole thing should hopefully be faster. Option to calculate
    # .dag() directly, saves time for gradient
    def compute_all_pseudo_fids(self, dag=False):
        dyn = self.parent
        time_st = timeit.default_timer()
        fwd_evo = dyn._fwd_evo[dyn.num_tslots]
        pseudo_fids = []
        for sub_sys in range(self.num_sub_sys):
            if dyn.oper_dtype == Qobj:
                overlap = self.large_local_targs_dag[sub_sys] * fwd_evo
            else:
                overlap = self.large_local_targs_dag[sub_sys].dot(fwd_evo)

            surviving_systems = list(range(self.num_sub_sys))
            surviving_systems.remove(sub_sys)
            #Checks if there is only one sub-system
            if dyn.oper_dtype == Qobj:
                if not surviving_systems:
                    pseudo_fids.append(overlap.tr())
                elif not dag:
                    pseudo_fids.append(overlap.ptrace(surviving_systems))
                else:
                    pseudo_fids.append(
                            (overlap.ptrace(surviving_systems)).dag())
            else:
                if not surviving_systems:
                    pseudo_fids.append(np.trace(overlap))
                elif not dag:
                    pseudo_fids.append(
                            self._ptrace(overlap, surviving_systems))
                else:
                    pseudo_fids.append(
                            self._ptrace(overlap, surviving_systems).T.conj())

        if isinstance(dyn.stats, StatsFidCompLocal):
            dyn.stats.wall_time_pseudo_fids_compute += \
                timeit.default_timer() - time_st
        return pseudo_fids

    def compute_total_fid(self):
        dyn = self.parent
        time_st = timeit.default_timer()
        pseudo_fids = self.compute_all_pseudo_fids()
        total_fid = 1
        for sub_sys in range(self.num_sub_sys):
            pseudo_fid = pseudo_fids[sub_sys]
            # replace .dag() and trace with absolute value squared
            # in case of only one sub system
            if isinstance(pseudo_fid, complex):
                local_fid_unnorm = np.real( pseudo_fid * np.ma.conjugate(pseudo_fid) )
            elif dyn.oper_dtype == Qobj:
                local_fid_unnorm = np.real((pseudo_fid * pseudo_fid.dag()).tr())
            else:
                local_fid_unnorm = np.real(np.trace(pseudo_fid.dot(pseudo_fid.T.conj())))
            local_fid_norm = (local_fid_unnorm / (self.full_dim * self.sub_dims[sub_sys]))
            local_fid_norm = my_round(local_fid_norm, self.numer_acc)
            total_fid += (local_fid_norm - 1)

        if isinstance(dyn.stats, StatsFidCompLocal):
            dyn.stats.wall_time_total_fid_compute += \
                timeit.default_timer() - time_st
        return total_fid

    def compute_all_owd_local_target(self, target_less_owd_evo):
        dyn = self.parent
        time_st = timeit.default_timer()
        all_owd_local_target = []
        for sub_sys in range(self.num_sub_sys):
            if dyn.oper_dtype == Qobj:
                all_owd_local_target.append(self.large_local_targs_dag[sub_sys]*target_less_owd_evo)
            else:
                all_owd_local_target.append(self.large_local_targs_dag[sub_sys].dot(target_less_owd_evo))
        if isinstance(dyn.stats, StatsFidCompLocal):
            dyn.stats.wall_time_local_target_compute += \
                timeit.default_timer() - time_st
        return all_owd_local_target

    def compute_local_fid_grad(self, sub_sys, owd_local_targ, prop_grad_fwd_evo, pseudo_fid_dag):
        dyn = self.parent
        time_st = timeit.default_timer()
        surviving_systems = list(range(self.num_sub_sys))
        surviving_systems.remove(sub_sys)
        norm = self.full_dim * self.sub_dims[sub_sys]
        #Checks if there is only one sub-system
        if not surviving_systems:
            if dyn.oper_dtype == Qobj:
                pseudo_grad = (owd_local_targ * prop_grad_fwd_evo).tr()
                overlap = pseudo_grad * np.ma.conjugate(pseudo_fid_dag)
            else:
                pseudo_grad = np.trace(owd_local_targ.dot(prop_grad_fwd_evo))
                overlap = pseudo_grad.dot(np.ma.conjugate(pseudo_fid_dag))
        else:
            if dyn.oper_dtype == Qobj:
                print("compute_local_fid_grad: subsys {}, dims{}".format(sub_sys, (owd_local_targ * prop_grad_fwd_evo).dims))
                pseudo_grad = (owd_local_targ * prop_grad_fwd_evo).ptrace(surviving_systems)
                overlap = (pseudo_grad * pseudo_fid_dag).tr()
            else:
                pseudo_grad = self._ptrace(owd_local_targ.dot(prop_grad_fwd_evo), surviving_systems)
                overlap = np.trace(pseudo_grad.dot(pseudo_fid_dag))
        local_fid_grad = np.real(overlap/norm)*2
        local_fid_grad = my_round(local_fid_grad, self.numer_acc)
        if isinstance(dyn.stats, StatsFidCompLocal):
            dyn.stats.wall_time_local_fid_grad_compute += \
                timeit.default_timer() - time_st
        return local_fid_grad

    def compute_global_choi_fid(self):
        # Not used in optimisation, but useful to have at the end to check answer
        # Exact copy of what FidCompPureChoiGlobal uses
        # Except from no rounding, which is more useful
        dyn = self.parent
        if self.log_level <= logging.DEBUG:
            logger.debug("**** Computing Global Choi fidelity ****")
        dyn.compute_evolution()

        if dyn.oper_dtype == Qobj:
            f_half = (dyn._fwd_evo[-1] * dyn.onto_evo_target).tr()
        else:
            f_half = np.trace(dyn._fwd_evo[-1].dot(dyn.onto_evo_target.full()))

        f = np.real((f_half * np.conjugate(f_half))/(self.full_dim**2))

        return f
