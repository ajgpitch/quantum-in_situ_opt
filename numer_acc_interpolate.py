# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:29:39 2015

Alexander Pitchford

Combine the results of multiple optimisation runs and summarise
Results must be in the same folder
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
#import re
#import datetime
#import shutil
import qsoconfig, qso
import qsoresult
from scipy.stats import linregress

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs."""
    if num != 0:
        return np.round(num,
                        -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

#
# /home/alex/quant_self_optim/HPCW_output/na_limit/3qubit/3qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq3fet1e-1-try3
# /home/alex/quant_self_optim/HPCW_output/na_limit/4qubit/4qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq4fet1e-1-try4
#/home/alex/quant_self_optim/HPCW_output/na_limit/5qubit/5qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq5fet1e-1-try2
# /home/alex/quant_self_optim/HPCW_output/na_limit/6qubit/6qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq6fet1e-1-try2
# /home/agp1/quant_self_optim/HPCW_output/na_limit-CNOT_sep/3qubit/3qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-3q3e-3-try1
data_dir = "/home/agp1/quant_self_optim/HPCW_output/na_limit-CNOT_sep/7qubit"
all_subdir = True
result_subdir = "7qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-7q1e-3-try1"
#set comb_coll_fname=None to collate separately
comb_coll_fname = None
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

# First look for a parameter file
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
fid_comp = dyn.fid_computer

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
    na_lb = data[0]
    na_ub = data[1]
    succ_thresh = data[2]
else:
    if not fid_comp.numer_acc_exact:
        fid_comp.st_numer_acc = round_sigfigs(
                fid_comp.st_numer_acc*fid_err_targ/tc.fid_err_targ, 6)
        fid_comp.end_numer_acc = round_sigfigs(
                fid_comp.end_numer_acc*fid_err_targ/tc.fid_err_targ, 6)

    na_lb = fid_comp.st_numer_acc
    na_ub = fid_comp.end_numer_acc
    succ_thresh = 0.5


fig = plt.figure(figsize=(16,6))
plt.ion()
#plt.show()

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
                 "qubits fid_err_targ {}".format(dyn.num_qubits,
                                             fid_err_targ))
    ax1.set_xlabel("numerical accuracy")
    ax1.set_ylabel("pulseoptim success proportion")

    ax2 = fig.add_subplot(122)
    ax2.plot(sel_na, sel_iter, 'x', label='included {}'.format(len(sel_na)))
    ax2.plot(other_na, other_iter, 'x', label='excluded {}'.format(len(other_na)))
    ax2.vlines([na_lb, na_ub], 0, max(iter_arr))

    ax2.set_xlim(0, na_lim)
    ax2.set_title("Iterations for {} "
                 "qubits fid_err_targ {}".format(dyn.num_qubits,
                                             fid_err_targ))
    ax2.set_xlabel("numerical accuracy")
    ax2.set_ylabel("num iter")

    num_sel = len(sel_na)
    interp_data = [na_lb, na_ub, succ_thresh, fid_err_targ, num_sel]

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
        print("sp polyfit failed: {}".format(sys.exc_info()[0]))
        interpsp = False

    if interpsp:
        # plot the fit
        # Not this is simple only because range is succ=0 to succ=1
        fitlinex = [c, m + c]
        fitliney = [0, 1]
        na_thresh = succ_thresh*m + c
        na_thresh_err = np.sqrt(m_var*succ_thresh**2 + c_var)
#        # get the first numer_acc after the threshold
#        first_thresh_na = na_arr[na_arr > na_thresh][0]
#        print("First numer_acc after thresh: {}".format(first_thresh_na))
#        na_res = coll_na[first_thresh_na]
#        mean_iter = na_res.mean_iter_primary_success
#        std_iter = na_res.std_iter_primary_success
#        print("Num iter for numer_acc after thresh {} +/- {}".format(
#                  mean_iter, std_iter))

        # Save interp data
        interp_data += [na_thresh, na_thresh_err]
        ax1.plot(fitlinex, fitliney, label='lin fit')
        ax1.errorbar(na_thresh, succ_thresh, xerr=na_thresh_err,
                    label='{}% thresh={:0.3e} +/- {:0.2e}'.format(
                            succ_thresh*100, na_thresh, na_thresh_err))
        ax1.legend()

        try:
            p, V = np.polyfit(sel_na, sel_iter, 1, cov=True)
            m = p[0]
            c = p[1]
            m_var = V[0][0]
            c_var = V[1][1]
            print("interpni: m {}, c{}".format(m, c))
            print("interpni variance: m {}, c{}".format(m_var, c_var))
            interpni = True
        except:
            print("ni polyfit failed: {}".format(sys.exc_info()[0]))
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

            # Save interp data
            interp_data += [na_thresh_iter, iter_err, comb_err]
            ax2.plot(fitlinex, fitliney, label='lin fit')
            ax2.errorbar(na_thresh, na_thresh_iter, yerr=iter_err,
                        label='na_thresh_iter err')
            ax2.errorbar(na_thresh, na_thresh_iter, yerr=comb_err,
                        label='na_thresh_iter={:0.3e} +/- {:0.2e}'.format(
                                na_thresh_iter, comb_err))
            ax2.legend(loc=0)

        plt.draw()

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
