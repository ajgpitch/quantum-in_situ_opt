# -*- coding: utf-8 -*-
"""
Interpolate the numerical accuracy search data to determine the
threshold at which it starts to reduce the pulse optimisation success
below some specified value (typically 50%)

It will look for data files in a directory specified by:
    data_dir/result_subdir
This can be an absolute path or relative to the current working directory
or relative to the user home directory by starting data_dir with '~/'

A collated results file will be looked for that must match the pattern
 coll_file_pat = "nal_collate*.txt".

Some parameters used for the optimisations are also needed. These will first
be looked for in a file interp_fname = "interp.dat". If this is not present
then it will look for a parameter file matching  param_file_pat = "params*.ini"

The data are interpolated using numpy.polyfit. This gives a line of best fit
with gradient and an intercept, with variance on both. Hence these can be used
to estimate the threshold and it's uncertianty. Although the data would seem
to look more Gaussian than linear, linear would seem to be a good approximation
around the 50% threshold. This method would probably not be reliable for
estimating high (e.g. 90%) or low (e.g. 10%) thesholds. However, the objective
is to look for a trend as system size grows, so the choice of success
proportion threshold is somehow arbitary, so long as it is reliable and
consistent. A similar interpolation is used to estimate the number of
iterations required on average to optimise pulses if the numerical accuracy
was at the threshold level.

The results are shown in two plots. These show the data points, trend lines,
and the threshold points are marked with error bars, which illustrate the
uncertainty.

As the data at the extremes of success proportion are not expected to fit
the linear model, then these are excluded from the interpolation. For the
first attempt these exclusion boundaries are estimated based on the fidelity
error target. It is possible to change them interactively using console
input.

The parameters used, iterpolation results, and exclusion boundaries are
saved in the interp_fname = "interp.dat" file. So if the script is re-run,
then the exclusion boundaries are rememebered. Also the plots are saved
to file.
"""

# this version 2018 April 6
# Author: Alexander Pitchford

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import qsoconfig
import qsoresult

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs."""
    if num != 0:
        return np.round(num,
                        -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

DEF_SUCC_THRESH = 0.5
data_dir = "example_output"
result_subdir = "3qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq3fet1e-3-try4"
#result_subdir = "5qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq5fet3e-2-try2"
#result_subdir = "7qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq7fet1e-2-try2"
#data_dir = "output/optim_CNOT"
#result_subdir = "nal""nal_collate*.txt"
#result_subdir = "5qubit-chain-ising_equal-xy_ctrl-cNOT1-fet1e-1-na2e-2"
param_file_pat = "params*.ini"
coll_file_pat = "nal_collate*.txt"
interp_fname = "interp.dat"
plot_fname = "interplot.png"

verbosity = 1

if '~' in data_dir:
    data_dir = os.path.expanduser(data_dir)
elif not os.path.abspath(data_dir):
    # Assume relative path from cwd given
    data_dir = os.path.join(os.getcwd(), data_dir)

res_dir = os.path.join(data_dir, result_subdir)

print("Results dir:\n{}".format(res_dir))
if not os.path.isdir(res_dir):
    raise RuntimeError("Results dir not valid")

# look for files to read existing results
coll_na = {}
na_list = []

if verbosity > 0:
    print("Searching in:\n{}".format(res_dir))
    print("Looking for collation file matching {}\n".format(coll_file_pat))
files = glob.glob(os.path.join(res_dir, coll_file_pat))

if len(files) == 0:
    print("No collation file to process")

plot_fpath = os.path.join(res_dir, plot_fname)
# Take the most recent file, assume last in list
collf = sorted(files)[-1]
if verbosity > 0:
    print("Loading collation from file:\n{}\n".format(collf))
reslist = qsoresult.MultiRepResult.load_from_txt(collf)
for na_res in reslist:
    na_list.append(na_res.numer_acc)
    coll_na[na_res.numer_acc] = na_res

na_arr = np.array(sorted(na_list))
succ_props = []
num_iters = []
for na in na_arr:
    na_res = coll_na[na]
    succ_props.append(float(na_res.num_primary_success) / na_res.num_res)
    num_iters.append(na_res.mean_iter_primary_success)
    fid_err_targ = na_res.fid_err_targ

succ_arr = np.array(succ_props)
iter_arr = np.array(num_iters)
na_lim = na_arr[-1]*1.2

# Look for previous interp results file
interp_fpath = os.path.join(res_dir, interp_fname)
if os.path.isfile(interp_fpath):
    if verbosity > 0:
        print("Loading interp params from:\n{}\n".format(interp_fpath))
    data = np.loadtxt(interp_fpath)
    # Note num_qubits only added to the file 2018-04-13
    # interp files from before that will need number of qubits inserting
    nq = int(data[0])
    na_lb = data[1]
    na_ub = data[2]
    try:
        succ_thresh = data[3]
    except:
        succ_thresh = DEF_SUCC_THRESH
else:
    # No interp results file, look for a parameter file
    param_pat = os.path.join(res_dir, param_file_pat)
    if verbosity > 0:
        print("Looking for parameter file matching:\n{}".format(param_pat))
    files = glob.glob(param_pat)
    n_files = len(files)
    if n_files == 0:
        print("NO PARAMETER FILES FOUND!")
        raise RuntimeError("No parameter file")
    print("Loading configuration from:\n{}".format(files[0]))
    optim = qsoconfig.gen_config(files[0])
    dyn = optim.dynamics
    tc = optim.termination_conditions
    fid_comp = optim.dynamics.fid_computer
    nq = dyn.num_qubits
    if not fid_comp.numer_acc_exact:
        fid_comp.st_numer_acc = round_sigfigs(
                fid_comp.st_numer_acc*fid_err_targ/tc.fid_err_targ, 6)
        fid_comp.end_numer_acc = round_sigfigs(
                fid_comp.end_numer_acc*fid_err_targ/tc.fid_err_targ, 6)

    na_lb = fid_comp.st_numer_acc
    na_ub = fid_comp.end_numer_acc
    succ_thresh = 0.5

fig = plt.figure(figsize=(16,6))
# This turns on interactive plotting
plt.ion()

exit_all = False
while not exit_all:
    sel = (na_arr > na_lb) & (na_arr < na_ub)
    sel_na = na_arr[sel]
    other_na = na_arr[~sel]
    sel_succ = succ_arr[sel]
    other_succ = succ_arr[~sel]
    sel_iter = iter_arr[sel]
    other_iter = iter_arr[~sel]

    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.plot(sel_na, sel_succ, 'x', label='included {}'.format(len(sel_na)))
    ax1.plot(other_na, other_succ, 'x', label='excluded {}'.format(len(other_na)))
    ax1.vlines([na_lb, na_ub], 0, 1)
    ax1.set_xlim(0, na_lim)
    ax1.set_title("Numerical accuracy limit for {} "
                 "qubits fid_err_targ {}".format(nq, fid_err_targ))
    ax1.set_xlabel("numerical accuracy")
    ax1.set_ylabel("pulseoptim success proportion")

    ax2 = fig.add_subplot(122)
    ax2.plot(sel_na, sel_iter, 'x', label='included {}'.format(len(sel_na)))
    ax2.plot(other_na, other_iter, 'x', label='excluded {}'.format(len(other_na)))
    ax2.vlines([na_lb, na_ub], 0, max(iter_arr))

    ax2.set_xlim(0, na_lim)
    ax2.set_title("Iterations for {} "
                 "qubits fid_err_targ {}".format(nq, fid_err_targ))
    ax2.set_xlabel("numerical accuracy")
    ax2.set_ylabel("num iter")

    num_sel = len(sel_na)
    interp_data = [nq, na_lb, na_ub, succ_thresh, fid_err_targ, num_sel]

    # Note we are doing this inverted,
    # as we are interested in the error in na to succ_prop
    # m, c, r, p, e = linregress(sel_succ, sel_na)
    try:
        p, V = np.polyfit(sel_succ, sel_na, 1, cov=True)
        m = p[0]
        c = p[1]
        m_var = V[0][0]
        c_var = V[1][1]
        interpsp = True
    except:
        print("success proportion polyfit failed: {}".format(sys.exc_info()[0]))
        print("Try changing the result bounds")
        interpsp = False

    if interpsp:
        # plot the fit
        # Not this is simple only because range is succ=0 to succ=1
        fitlinex = [c, m + c]
        fitliney = [0, 1]
        na_thresh = succ_thresh*m + c
        na_thresh_err = np.sqrt(m_var*succ_thresh**2 + c_var)
        na_thresh_msg = '{}% thresh={:0.3e} +/- {:0.2e}'.format(
                            succ_thresh*100, na_thresh, na_thresh_err)
        print("Numerical accuracy threshold calculated:\n"
              + na_thresh_msg + "\n")

        # Save interp data
        interp_data += [na_thresh, na_thresh_err]
        ax1.plot(fitlinex, fitliney, label='lin fit')
        ax1.errorbar(na_thresh, succ_thresh, xerr=na_thresh_err,
                    label=na_thresh_msg)
        ax1.legend()

        try:
            p, V = np.polyfit(sel_na, sel_iter, 1, cov=True)
            m = p[0]
            c = p[1]
            m_var = V[0][0]
            c_var = V[1][1]
            if verbosity > 1:
                print("interpni: m {}, c{}".format(m, c))
                print("interpni variance: m {}, c{}".format(m_var, c_var))
            interpni = True
        except:
            print("num iter polyfit failed: {}".format(sys.exc_info()[0]))
            interpni = False

        if interpni:
            # plot the fit
            miny = min(sel_iter)
            maxy = max(sel_iter)
            minx = min(sel_na)
            maxx = max(sel_na)
            fitlinex = [minx, maxx]
            fitliney = [m*minx + c, m*maxx + c]
            na_thresh_iter = na_thresh*m + c
            iter_err = np.sqrt(m_var*na_thresh**2 + c_var)
            comb_err = np.sqrt((m*na_thresh_err)**2 + iter_err**2)
            na_thresh_iter_msg ='na_thresh_iter={:0.3e} +/- {:0.2e}'.format(
                                    na_thresh_iter, comb_err)
            print("Mean interations and combined error for numerical accuracy "
                  "threshold calculated:\n" + na_thresh_iter_msg + "\n")
            # Save interp data
            interp_data += [na_thresh_iter, iter_err, comb_err]
            ax2.plot(fitlinex, fitliney, label='lin fit')
            ax2.errorbar(na_thresh, na_thresh_iter, yerr=iter_err,
                        label='na_thresh_iter err')
            ax2.errorbar(na_thresh, na_thresh_iter, yerr=comb_err,
                        label=na_thresh_iter_msg)
            ax2.legend(loc=0)

        plt.draw()

    print("Numerical accuracy result bounds: lower {},"
          " upper {}".format(na_lb, na_ub))
    np.savetxt(interp_fpath, interp_data, fmt='%.5e')


    #plt.show()
    #plt.show(block=False)
    plt.pause(0.1)
    fig.savefig(plot_fpath, bbox_inches='tight')

    get_new_vals = True
    while get_new_vals:
        new_vals = input("Enter new values [na_lb[,na_ub[,succ_thresh]]]:\n")
        # print("new_vals: '{}'".format(new_vals))
        if len(new_vals) == 0:
            exit_all = True
            break
        if new_vals == 'keep':
            get_new_vals = False
            break
        vals = new_vals.split(',')
        try:
            na_lb = float(vals[0].strip())
        except:
            print("na_lb not a float")
            continue
        if len(vals) > 1:
            try:
                na_ub = float(vals[1].strip())
            except:
                print("na_ub not a float")
                continue
        if len(vals) > 2:
            try:
                succ_thresh = float(vals[2].strip())
            except:
                print("succ_thresh not a float")
                continue
        get_new_vals = False

    na_lim = na_ub*1.2

plt.ioff()
plt.show()
