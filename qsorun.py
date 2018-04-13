# -*- coding: utf-8 -*-
"""
Created: 2016-12-5

Alexander Pitchford

functions to run optimisations
"""
import sys
import os
import numpy as np
import datetime
from multiprocessing import Pool
from copy import copy
import glob
import qso
import qsoresult

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs."""
    if num != 0:
        return np.round(num,
                        -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

def run_qso_sims(optim, num_reps=None, verbosity=None,
                 report_phys_params=None, save_results=None,
                 scen_idx=None, reps_idx=None, num_threads=None,
                 num_tslots=None, evo_time=None,
                 fid_err_targ=None, numer_acc=None):
    """
    Runs the qso optimistion for the specified number of reps
    (taken from optim.cfg if 0). The results are analysed and returned
    in a result object
    """

    #print("run_qso_sims\nnum_reps {}, job_idx {}".format(num_reps, job_idx))

    cfg = optim.config
    dyn = optim.dynamics
    tc = optim.termination_conditions
    fid_comp = dyn.fid_computer
    pgen = optim.pulse_generator

    cfg_str = qso.get_cfg_str(optim, num_tslots=num_tslots, evo_time=evo_time,
                     fid_err_targ=fid_err_targ, numer_acc=numer_acc)
    out_file_ext = qso.get_out_file_ext(cfg.data_file_ext, job_id=cfg.job_id,
                            scen_idx=scen_idx, reps_idx=reps_idx)

    if num_reps is None: num_reps = cfg.num_reps
    if verbosity is None: verbosity = cfg.verbosity
    if report_phys_params is None: report_phys_params = cfg.report_phys_params
    if save_results is None: save_results = cfg.save_results
    if num_threads is None: num_threads = cfg.num_threads
    if num_tslots is not None:
        dyn.num_tslots = num_tslots
        pgen.num_tslots = num_tslots
        pgen.tau = None
    if evo_time is not None:
        dyn.evo_time = evo_time
        pgen.pulse_time = evo_time
        pgen.tau = None
    if fid_err_targ is not None: tc.fid_err_targ = fid_err_targ
    if numer_acc is not None: fid_comp.numer_acc = numer_acc

    # Only use stdout for logging messages when first process
    # (which is true when the idx vars are both None or 0)
    base_log = True
    if scen_idx is not None or reps_idx is not None:
        datetimestamp = datetime.datetime.now().strftime('%d%b_%H-%M')
        script_name = "{}-{}.{}".format(cfg_str, datetimestamp,
                                        out_file_ext)
        script_path = os.path.join(cfg.output_dir, script_name)
        lfh = open(script_path, 'a')
        base_log = False
    else:
        lfh = sys.stdout

    if verbosity > 0:
        lfh.write("want {} threads per rep\n".format(num_threads))

    try:
        import mkl
        use_mkl = True
    except:
        use_mkl = False

    if use_mkl:
        mkl.set_num_threads(num_threads)
        if verbosity > 0:
            lfh.write("Number of threads set as {}\n".format(
                        mkl.get_max_threads()))
    else:
        if verbosity > 0:
            lfh.write("mkl unavailable\n")

    if verbosity > 0:
        lfh.write("Running {} reps under scen_idx {}, reps_idx {}\n".format(
                        num_reps, scen_idx, reps_idx))

    multires = qsoresult.MultiRepResult(tc.fid_err_targ, fid_comp.local,
                num_tslots=num_tslots, evo_time=evo_time, numer_acc=numer_acc)

    if verbosity > 2:
        lfh.write("multires optional attribs: num_tslots={}, evo_time={}, "
                     "fid_err_targ={}, numer_acc={}\n".format(
                        multires.num_tslots, multires.evo_time,
                        multires.fid_err_targ, multires.numer_acc))

    # Repetition paramaters and results arrays

    # force the random number generator to reseed, as would cause issue
    # when using multiprocessing
    np.random.seed()

    # set up the decoupling slots
    # dyn.num_decoup_tslots implies that a specific decoup tslot has been given
    if dyn.num_decoup_tslots is not None:
        if dyn.num_decoup_tslots == 0:
            # assume all timeslots
            dyn.decoup_tslots = np.ones([dyn.num_tslots])
        else:
            dyn.decoup_tslots = np.zeros([dyn.num_tslots])
            dyn.decoup_tslots[:dyn.num_decoup_tslots] = 1

    if verbosity > 2:
        lfh.write("Decoup timeslots: {}\n".format(dyn.decoup_tslots))

    if len(dyn.decoup_tslots) != dyn.num_tslots:
        raise RuntimeError("Number of decoupling tslots {} not equal to "
                           "number of timeslots {}".format(
                                   len(dyn.decoup_tslots, num_tslots)))

    try:
        for k in range(num_reps):
            if dyn.auto_hspace and (dyn.hspace_0_idx < 0
                                    or dyn.hspace_01_sep < 0):
                dyn.hspace_order = qso.get_coupling_hspace(dyn.num_qubits,
                                                           dyn.hspace_0_idx,
                                                           dyn.hspace_01_sep)
                if verbosity > 0:
                    lfh.write("reconfiguring drift with hspace_order "
                              "= {}\n".format(dyn.hspace_order))
                dyn.drift_dyn_gen = qso.get_drift(dyn)

            # Generate# pulses for each control
            init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])
            pgen = optim.pulse_generator
            for j in range(dyn.num_ctrls):
                init_amps[:, j] = pgen.gen_pulse()
            if dyn.decoup_x > 0:
                for i in dyn.Sx_cidx:
                    init_amps[:, i] += dyn.decoup_tslots*dyn.decoup_x
            if dyn.decoup_y > 0:
                for i in dyn.Sy_cidx:
                    init_amps[:, i] += dyn.decoup_tslots*dyn.decoup_y
            if dyn.decoup_z > 0:
                for i in dyn.Sz_cidx:
                    init_amps[:, i] += dyn.decoup_tslots*dyn.decoup_z
            dyn.initialize_controls(init_amps)

            if cfg.save_initial_amps:
                pulsefile = "init_amps_{}_rep{}.{}".format(cfg_str, k+1,
                                                           out_file_ext)
                pfpath = os.path.join(cfg.output_dir, pulsefile)
                dyn.save_amps(pfpath, times="exclude")
                if verbosity > 1: lfh.write("Initial amps saved\n")

            if optim.dump:
                optim.dump.clear()
                optim.dump.dump_file_ext = out_file_ext
                optim.dump.fname_base = "optim_dump_rep{}_{}".format(k+1, cfg_str)
            if dyn.dump:
                dyn.dump.clear()
                dyn.dump.dump_file_ext = out_file_ext
                dyn.dump.fname_base = "dyn_dump_rep{}_{}".format(k+1, cfg_str)

            if verbosity > 0:
                lfh.write("\nStarting pulse optimisation {} of {}\n".format(
                                                k+1, num_reps))
            if verbosity > 1:
                lfh.write("Max wall time {}\n".format(
                            optim.termination_conditions.max_wall_time))
            optres = optim.run_optimization()

            optres.optim_dump = optim.dump
            optres.dyn_dump = dyn.dump

            repres = multires.add_optim_result(optres)


            if cfg.save_final_amps:
                pulsefile = "final_amps_{}_rep{}.{}".format(cfg_str, k+1,
                                                           out_file_ext)
                pfpath = os.path.join(cfg.output_dir, pulsefile)
                dyn.save_amps(pfpath, times="exclude")
                if verbosity > 1: lfh.write("Final amps saved\n")

            if verbosity > 0 and cfg.report_stats:
                lfh.write("Optimising complete. Stats follow:\n")
                optres.stats.report()

            if verbosity > 0:
                lfh.write("********* Summary *****************\n")
                lfh.write("Initial fidelity error {}\n".format(optres.initial_fid_err))
                lfh.write("Final fidelity error {}\n".format(optres.fid_err))
                if fid_comp.local:
                    lfh.write("Final TRUE choi fidelity error {}\n".format(
                                    1-dyn.fid_computer.compute_global_choi_fid()))
                lfh.write("Terminated due to {}\n".format(optres.termination_reason))
                lfh.write("Number of iterations {}\n".format(optres.num_iter))
                lfh.write("Completed in {} HH:MM:SS.US\n".format(
                        datetime.timedelta(seconds=optres.wall_time)))
                lfh.write("Final gradient normal {}\n".format(optres.grad_norm_final))
                lfh.write("***********************************\n")

            if optres.optim_dump:
                if verbosity > 0: lfh.write("Optim dump saved\n")
                optres.optim_dump.writeout()

            if optres.dyn_dump:
                if verbosity > 0: lfh.write("Dynamics dump saved\n")
                optres.dyn_dump.writeout()

            if cfg.keep_optim_result:
                repres.optim_result = optres
            else:
                del(optres)

    except KeyboardInterrupt as e:
        lfh.write("\nProcessing interrupted\n")
        if not base_log:
            lfh.close()
        raise e

    if verbosity > 0:
        lfh.write("\n***** ALL SEARCHING FINISHED *****\n\n")

    multires.analyse_results()

    if save_results:
        fname = "results_{}.{}".format(cfg_str, out_file_ext)
        fpath = os.path.join(cfg.output_dir, fname)
        with open(fpath, 'w') as fh:
            multires.write_results(fh)
            if verbosity > 0:
                lfh.write("Results saved to:\n{}\n".format(fpath))

    if verbosity > 0:
        lfh.write("\nFull results\n")
        multires.write_results(lfh)
        # Print very short summary
        multires.report_analysis(f=lfh)
        if report_phys_params:
            qso.print_phys_params(optim, f=lfh)

    if not base_log:
        lfh.close()

    return multires


def get_mp_params(num_reps, num_cpus, lfh=sys.stdout, verbosity=0):
    """
    Determine the multiprocessing parameters
    This will either specify a number of reps per process
    or a number of threads per rep
    """

    threads_per_proc = 1
    if num_reps > num_cpus:
        num_procs = num_cpus
        reps_per_proc = num_reps // num_cpus
        reps_per_proc_rem = num_reps % num_cpus
    else:
        num_procs = num_reps
        reps_per_proc = 1
        reps_per_proc_rem = 0
        if num_cpus > num_reps:
            threads_per_proc = num_cpus // num_reps
            if num_cpus % num_reps > 0:
                threads_per_proc += 1

    if verbosity > 0:
        if reps_per_proc_rem == 0:
            lfh.write("Will run {} reps on {} processes\n".format(
                        reps_per_proc, num_procs))
        else:
            lfh.write("Will run {} reps on process 1 "
                        "and {} on another {} processes\n".format(
                        reps_per_proc + reps_per_proc_rem, reps_per_proc,
                        num_procs-1))
        lfh.write("Will use {} threads per process\n".format(
                                                threads_per_proc))

    return num_procs, reps_per_proc, reps_per_proc_rem, threads_per_proc

def get_rep_task_kwargs_list(def_kwargs, num_procs, reps_per_proc,
                             reps_per_proc_rem, threads_per_proc,
                             lfh=sys.stdout, verbosity=0):

    """
    Get list of kwargs for the multiprocessing.
    Set the number of threads for mkl
    """

    task_kwargs_list = []
    reps_this_proc = reps_per_proc + reps_per_proc_rem
    for reps_idx in range(num_procs):
        task_kwargs = def_kwargs.copy()
        task_kwargs['reps_idx'] = reps_idx
        task_kwargs['num_reps'] = reps_this_proc
        task_kwargs['num_threads'] = threads_per_proc
        if verbosity > 0:
            lfh.write("Will start reps_idx {} for {} reps\n".format(reps_idx,
                                          reps_this_proc))
        task_kwargs_list.append(task_kwargs)
        reps_this_proc = reps_per_proc

    return task_kwargs_list

def run_qso_sims_mp(optim, lfh=sys.stdout, num_reps=None, verbosity=None,
                    report_phys_params=None, save_results=None, num_cpus=None,
                    write_anal=True):
    """
    Run the QSO simulations using multiprocessing
    """
    cfg = optim.config

    if num_cpus is None: num_cpus = cfg.num_cpus
    if num_reps is None: num_reps = cfg.num_reps
    if verbosity is None: verbosity = cfg.verbosity
    if report_phys_params is None: report_phys_params = cfg.report_phys_params
    if save_results is None: save_results = cfg.save_results

    def_task_kwargs = {'verbosity':verbosity, 'save_results':save_results}

    all_res = None

    num_procs, reps_per_proc, reps_per_proc_rem, threads_per_proc = \
            get_mp_params(num_reps, num_cpus, lfh=lfh, verbosity=verbosity)

    if num_procs <= 1:
        if verbosity > 0:
            lfh.write("Single process only\n")

        all_res = run_qso_sims(optim, num_reps=num_reps, verbosity=verbosity,
                               save_results=save_results,
                               num_threads=threads_per_proc)
    else:
        tkl = get_rep_task_kwargs_list(def_task_kwargs,
                                       num_procs, reps_per_proc,
                                       reps_per_proc_rem, threads_per_proc,
                                       lfh=lfh, verbosity=verbosity)

        try:
            pool = Pool(processes=len(tkl))

            async_res = []
            for task_kwargs in tkl:
                async_res.append(pool.apply_async(run_qso_sims,
                                               (optim, ), task_kwargs))

            while not all([ar.ready() for ar in async_res]):
                for ar in async_res:
                    ar.wait(timeout=0.1)

            pool.terminate()
            pool.join()

        except KeyboardInterrupt as e:
            pool.terminate()
            pool.join()
            raise e

        for ar in async_res:
            multires = ar.get()
            if all_res is None:
                all_res = multires
            else:
                all_res.combine(multires, sort=False)

        if not all_res.sorted: all_res.sort_results()

    lfh.write("All MP reps complete\n")

    cfg_str = qso.get_cfg_str(optim)
#    print("cfg.data_file_ext {}, cfg.job_id {}".format(cfg.data_file_ext,
#                                                      cfg.job_id))
    out_file_ext = qso.get_out_file_ext(cfg.data_file_ext, job_id=cfg.job_id)
    fname = "comb_{}.{}".format(cfg_str, out_file_ext)
    fpath = os.path.join(cfg.output_dir, fname)
    with open(fpath, 'w') as fh:
        all_res.write_results(fh)
        if verbosity > 0:
            lfh.write("Combined results saved to:\n{}".format(fpath))

    if verbosity > 0:
        lfh.write("\nFull results\n")
        all_res.write_results(lfh)
        # Print very short summary
        all_res.report_analysis(f=lfh)
        if report_phys_params:
            qso.print_phys_params(optim, f=lfh)

    if write_anal:
        fname = "collate_{}.{}".format(cfg_str, out_file_ext)
        fpath = os.path.join(cfg.output_dir, fname)
        with open(fpath, 'w') as fh:
            all_res.write_file_header(fh)
            all_res.write_analysis(fh)
            if cfg.verbosity > 0:
                lfh.write("Analysis saved to:\n{}\n".format(fpath))

    return all_res

def run_qso_sims_tslot_range(optim, lfh=sys.stdout, verbosity=None,
                             report_phys_params=None,
                             save_results=None, num_cpus=None):
    """
    Run the QSO simulations using multiprocessing
    for a range of num_tslots
    """
    cfg = optim.config
    dyn = optim.dynamics

    if num_cpus is None: num_cpus = cfg.num_cpus
    if verbosity is None: verbosity = cfg.verbosity
    if report_phys_params is None: report_phys_params = cfg.report_phys_params
    if save_results is None: save_results = cfg.save_results

    if len(dyn.num_tslots_list) > 0:
        nts_list = dyn.num_tslots_list
    else:
        nts_list = range(dyn.st_num_tslots, dyn.st_num_tslots +
                                dyn.d_num_tslots*dyn.num_num_tslots,
                                dyn.d_num_tslots)

    num_scen = len(nts_list)
    def_task_kwargs = {'verbosity':verbosity, 'save_results':save_results}
    if num_cpus > num_scen:
        cpus_per_scen = num_cpus // num_scen
        cpus_per_scen_rem = num_cpus % num_scen
    else:
        cpus_per_scen = 1
        cpus_per_scen_rem = 0

    if cpus_per_scen_rem == 0:
        lfh.write("Running {} scenarios with {} cpus per scenario\n".format(
                    num_scen, cpus_per_scen))
    else:
        lfh.write("Running 1 scenario {} on cpus "
                "and {} scenarios on {} cpus\n".format(
                    cpus_per_scen + cpus_per_scen_rem, num_scen-1,
                    cpus_per_scen))
    try:
        pool = Pool(processes=num_cpus)
        async_res = []
        cpus_this_scen = cpus_per_scen + cpus_per_scen_rem
        scen_idx = 0
        for nts in nts_list:
            scen_task_kwargs = def_task_kwargs.copy()
            scen_task_kwargs['scen_idx'] = scen_idx
            scen_task_kwargs['num_tslots'] = nts
            lfh.write("Run scenario nts={} with {} cpus\n".format(nts,
                                                          cpus_this_scen))
            tkl = get_rep_task_kwargs_list(scen_task_kwargs, cfg.num_reps,
                                         cpus_this_scen, verbosity=verbosity)

            for task_kwargs in tkl:
                async_res.append(pool.apply_async(run_qso_sims,
                                               (optim, ), task_kwargs))

            scen_idx += 1
            cpus_this_scen = cpus_per_scen

        while not all([ar.ready() for ar in async_res]):
            for ar in async_res:
                ar.wait(timeout=0.1)

        pool.terminate()
        pool.join()

    except KeyboardInterrupt as e:
        pool.terminate()
        pool.join()
        raise e

    # collate all the multiple results by num_tslots
    # as reps may have been completed in multiple processes.
    coll_nts = {}
    for ar in async_res:
        multires = ar.get()
        nts_res = coll_nts.get(multires.num_tslots, None)
        if nts_res:
            nts_res.combine(multires)
        else:
            coll_nts[multires.num_tslots] = multires

    # Now loop through them in order writing out the analysis
    # and all the results
    cfg_str = qso.get_cfg_str(optim)
    out_file_ext = qso.get_out_file_ext(cfg.data_file_ext, job_id=cfg.job_id)
    coll_fname = "nts_collate_{}.{}".format(cfg_str, out_file_ext)
    coll_fpath = os.path.join(cfg.output_dir, coll_fname)
    clf = open(coll_fpath, 'w')
    inc_header_clf = True

    comb_fname = "nts_comb_{}.{}".format(cfg_str, out_file_ext)
    comb_fpath = os.path.join(cfg.output_dir, comb_fname)
    cbf = open(comb_fpath, 'w')
    inc_header_cbf = True

    all_res = None
    for nts in nts_list:
        nts_res = coll_nts.get(nts, None)
        if not nts_res:
            lfh.write("<<< WARNING: No results for num_tslots {} >>>".format(
                        nts))
            continue
        nts_res.add_opt_file_attribs()
        nts_res.write_analysis(clf, inc_header=inc_header_clf)

        if not nts_res.sorted: nts_res.sort_results()
        nts_res.write_results(cbf, inc_opt_attribs=True,
                              inc_header=inc_header_cbf)

        inc_header_clf = False
        inc_header_cbf = False

        if all_res:
            all_res.combine(nts_res)
        else:
            all_res = copy(nts_res)

    clf.close()
    cbf.close()
    if verbosity > 0:
        lfh.write("Analysis saved to:\n{}\n".format(coll_fpath))
        lfh.write("Results saved to:\n{}\n".format(comb_fpath))
        lfh.write("All num_tslot scenarios complete\n")
        lfh.write("\nFull results\n")
        all_res.write_results(inc_opt_attribs=True, f=lfh)
        # Print very short summary
        all_res.report_analysis(f=lfh)
        if report_phys_params:
            qso.print_phys_params(optim, f=lfh)

def run_qso_sims_numer_acc_limit(optim, lfh=sys.stdout, verbosity=None,
                                 report_phys_params=None,
                                 save_results=None, num_cpus=None):
    """
    Run the QSO simulations using multiprocessing
    Looking to get a spread of results in between the no affect
    from numer_acc and numer_acc too large for any pulse optim
    """
    cfg = optim.config
    tc = optim.termination_conditions
    dyn = optim.dynamics
    fid_comp = dyn.fid_computer

    if num_cpus is None: num_cpus = cfg.num_cpus
    if verbosity is None: verbosity = cfg.verbosity
    if report_phys_params is None: report_phys_params = cfg.report_phys_params
    if save_results is None: save_results = cfg.save_results

    cfg_str = qso.get_cfg_str(optim)
    out_file_ext = qso.get_out_file_ext(cfg.data_file_ext, job_id=cfg.job_id)
    coll_fname = "nal_collate_{}.{}".format(cfg_str, out_file_ext)
    coll_fpath = os.path.join(cfg.output_dir, coll_fname)
    comb_fname = "nal_comb_{}.{}".format(cfg_str, out_file_ext)
    comb_fpath = os.path.join(cfg.output_dir, comb_fname)

    def fill_na_list_gaps(active_na, test_na=[]):
        """Add numer_acc to the list where the biggest gaps are through
        bisection.
        """
        if num_scen <= len(test_na):
            return sorted(test_na)

        sel = np.diff(np.asarray(active_na)).argsort()[::-1]

        if num_scen - len(test_na) > len(sel):
            # More possible available scenarios than gaps
            scen_per_gap = num_scen - len(test_na) - len(sel) + 1
            num_gaps = len(sel)
        else:
            scen_per_gap = 1
            num_gaps = 1

        if verbosity > 2:
            lfh.write("{} gaps with {} scen per gap (initially)\n".format(
                                   num_gaps, scen_per_gap))
        for i in range(num_gaps):
            ld_idx = sel[i]
            new_na = np.linspace(active_na[ld_idx], active_na[ld_idx+1],
                                 scen_per_gap + 1, endpoint=False)
            for na in new_na[1:]:
                #print("Adding {}".format(round_sigfigs(na, 8)))
                test_na.append(round_sigfigs(na, 8))
            num_gaps = 1
        return sorted(test_na)

    def writeout_results(coll_na, all_na):
        """Loop through them in order writing out the analysis
        and all the results
        """
        clf = open(coll_fpath, 'w')
        inc_header_clf = True
        cbf = open(comb_fpath, 'w')
        inc_header_cbf = True
        for numer_acc in all_na:
            na_res = coll_na.get(numer_acc, None)
            if not na_res:
                lfh.write("<<< WARNING: No results for "
                          "numer_acc {} >>>\n".format(numer_acc))
                continue
            na_res.add_opt_file_attribs()
            na_res.write_analysis(clf, inc_header=inc_header_clf)

            # collation won't have any actual results
            # if it was loaded from file
            if len(na_res.results) > 0:
                if verbosity > 1:
                    lfh.write("Writing out {} results for "
                              "numer_acc={}\n".format(len(na_res.results),
                                                                  numer_acc))
                if not na_res.sorted: na_res.sort_results()
                na_res.write_results(cbf, inc_header=inc_header_cbf)
            inc_header_clf = False
            inc_header_cbf = False
        clf.close()
        cbf.close()

    def get_na_lists(coll_na, all_na):
        """ Determine lists of numer_acc that must be tested
        and those that are in the active area
        """
        st_idx = -1
        end_idx = len(all_na)
        succ_props = []
        i = 0
        for numer_acc in all_na:
            na_res = coll_na.get(numer_acc, None)
            if not na_res:
                lfh.write("<<< WARNING: No results for numer_acc {} >>>".format(
                            numer_acc))
                continue

            if verbosity > 1:
                lfh.write("Analysising result of combined result:\n"
                          "na: {}, num_res: {}, num_succ: {}\n".format(
                          na_res.numer_acc, na_res.num_res,
                          na_res.num_primary_success))

            na_res.succ_prop = (float(na_res.num_primary_success) /
                                     na_res.num_res)
            succ_props.append(na_res.succ_prop)
            if na_res.succ_prop > fid_comp.success_prop_uthresh:
                st_idx = i
            if (na_res.succ_prop < fid_comp.success_prop_lthresh
                        and i < end_idx):
                end_idx = i

            i += 1

        if verbosity > 0:
            lfh.write("succ props: {}\n"
                      "Upper threshold last met at idx {}\n"
                      "Lower threshold first met at idx {}\n".format(
                              succ_props, st_idx, end_idx))

        test_na = []
        # If upper success threshold not met then reduce lowest numer_acc
        if st_idx < 0:
            st_idx = 0
            if verbosity > 0:
                lfh.write("Upper threshold not met, reducing min numer_acc\n")
            test_na.append(all_na[0] / 2)
        # If lower success threshold not met then increase highest numer_acc
        if end_idx >= len(all_na):
            end_idx = len(all_na)
            if verbosity > 0:
                lfh.write("Lower threshold not met, increasing max numer_acc\n")
            test_na.append(all_na[-1] * 2)
        else:
            end_idx += 1

        active_na = all_na[st_idx:end_idx]
        if verbosity > 0:
            lfh.write("forcing na vals: {}\n"
                      "active na vals (for gaps): {}\n".format(
                                                  test_na, active_na))

        return test_na, active_na


    # The number of cpus must be some multiple of the number of reps
    # (or less than). Remaining cpus will be ignored
    num_scen = num_cpus // cfg.num_reps
    if num_scen == 1:
        num_cpus = cfg.num_reps
    elif num_scen == 0:
        num_scen = 1
    cpus_per_scen = num_cpus // num_scen
    lfh.write("Running {} concurrent scenarios "
              "with {} cpus per scenario\n".format(
                num_scen, cpus_per_scen))


    coll_na = {}
    scen_idx = 0
    na_list = []
    # look for files to read existing results
    collf_pat = os.path.join(cfg.output_dir, "nal_collate*.{}".format(
                             cfg.data_file_ext))
    if verbosity > 0:
        lfh.write("Looking for collation file matching {}\n".format(collf_pat))
    files = glob.glob(collf_pat)
    if len(files) > 0:
        # Take the most recent file, assume last in list
        collf = files[-1]
        if verbosity > 0:
            lfh.write("Loading collation from file:\n{}\n".format(collf))
        reslist = qsoresult.MultiRepResult.load_from_txt(collf)
        for na_res in reslist:
            na_list.append(na_res.numer_acc)
            coll_na[na_res.numer_acc] = na_res
            scen_idx += 1

    # otherwise take the initial numer acc values from the settings
    if len(na_list) < 2:
        test_na = [fid_comp.st_numer_acc, fid_comp.end_numer_acc]
        if verbosity > 0:
            lfh.write("Start with fixed scenarios {}:\n".format(test_na))
        test_na = fill_na_list_gaps(test_na + na_list, test_na)
    else:
        test_na = fill_na_list_gaps(na_list)
    if verbosity > 0:
        lfh.write("All initial scenarios {}:\n".format(test_na))

    def_task_kwargs = {'verbosity':verbosity, 'save_results':save_results}

    # all_res just used for report at the end
    # note the fid_err_targ and numer_acc are set just to give the
    # file attributes
    all_res = qsoresult.MultiRepResult(tc.fid_err_targ, True,
                                       numer_acc=fid_comp.st_numer_acc)
    all_res.add_opt_file_attribs()
    while scen_idx < cfg.max_mp_scens:
        try:
            pool = Pool(processes=num_cpus)
            async_res = []
            for numer_acc in test_na:
                scen_task_kwargs = def_task_kwargs.copy()
                scen_task_kwargs['scen_idx'] = scen_idx
                scen_task_kwargs['numer_acc'] = numer_acc
                lfh.write("Run scenario numer_acc={} "
                          "with {} cpus\n".format(numer_acc, cpus_per_scen))
                num_procs, reps_per_proc, reps_per_proc_rem, threads_per_proc = \
                        get_mp_params(cfg.num_reps, cpus_per_scen, lfh=lfh,
                                          verbosity=verbosity)
                tkl = get_rep_task_kwargs_list(scen_task_kwargs,
                                               num_procs, reps_per_proc,
                                               reps_per_proc_rem,
                                               threads_per_proc,
                                               lfh=lfh, verbosity=verbosity)

                for task_kwargs in tkl:
                    async_res.append(pool.apply_async(run_qso_sims,
                                                   (optim, ), task_kwargs))

                scen_idx += 1

            while not all([ar.ready() for ar in async_res]):
                for ar in async_res:
                    ar.wait(timeout=0.1)

            pool.terminate()
            pool.join()

        except KeyboardInterrupt as e:
            pool.terminate()
            pool.join()
            raise e

        # collate all the multiple results by numer_acc
        # as reps may have been completed in multiple processes.
        for ar in async_res:
            multires = ar.get()
            na_res = coll_na.get(multires.numer_acc, None)
            if na_res:
                if verbosity > 2:
                    lfh.write("Result for na {}, num_reps {} being combined"
                          " with existing result for na {}\n".format(
                          multires.numer_acc, multires.num_res,
                          na_res.numer_acc))
                na_res.combine(multires)
            else:
                if verbosity > 2:
                    lfh.write("New result for na {}, num_reps {}\n".format(
                              multires.numer_acc, multires.num_res))
                coll_na[multires.numer_acc] = multires

        for numer_acc in test_na:
            na_res = coll_na.get(numer_acc, None)
            if not na_res:
                lfh.write("<<< WARNING: No results for "
                          "numer_acc {} >>>\n".format(numer_acc))
                continue
            all_res.combine(na_res)

        all_na = sorted(coll_na.keys())
        if verbosity > 0:
            lfh.write("All na now: {}\n".format(all_na))
        writeout_results(coll_na, all_na)
        test_na, active_na = get_na_lists(coll_na, all_na)
        fill_na_list_gaps(active_na, test_na)
        if verbosity > 0:
            lfh.write("All chosen na vals: {}\n".format(test_na))

    lfh.write("All numer_acc scenarios complete\n")
    if cfg.verbosity > 0:
        if all_res:
            lfh.write("Analysis saved to:\n{}\n".format(coll_fpath))
            lfh.write("Results saved to:\n{}\n".format(comb_fpath))
            lfh.write("\nFull results\n")
            all_res.write_results(f=lfh)
            # Print very short summary
            all_res.report_analysis(f=lfh)
        if report_phys_params:
            qso.print_phys_params(optim, f=lfh)
