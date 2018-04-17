"""
Script for running simulations of in-situ quantum gate optimisation.
Specifically the optimisation of a CNOT gate on one pair of qubits, which are
typically part of a larger system, which can be configured in a variety of
topologies and with different interaction types.

QuTiP (qutip.org), along with its prequistes, is required to run this script.

The script can be used to generate the results from
https://arxiv.org/abs/1701.01723

The script loads its configuration from the local file:
    params-quant_self_opt.ini
These can also be overridden in some cases with cmdline args
The majority of the code is located in modules.
See qsoconfig.py for descriptions of the config parameters
A different parameter file can be specified using the -p cmdline option

There are three options for the objective of the script.
The option can be chosen by setting the optimconfig.mp_opt parameter,
so called because Python multiprocessing is used to spread the optimisation
repeats and or scenarios over the specified available resources.
1. mp_opt = <blank>
    This simply repeats the control pulse optimisation the number of times
    specified by num_reps. The number of successful repeats and averaged
    statisics are reported.
2. mp_opt = num_tslots
    A number of different num_tslots scenarios will be tried. This is either
    based on a list or range parameters. The specified num_reps will be
    completed for each scenario.
3. mp_opt = numer_acc_limit


For all options a results file will be produced by each process that
performs one or more repetitions of the pulse optimisation. All the repetition
results will also be combined in to one file. Another file will contain
collated results, grouped appropriately, with averaged statistics.
"""

# started 2015 Oct 6
# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

#Python Core
import inspect, os, sys
import shutil
import datetime
import signal
#QuTiP
import qutip.logging_utils as logging
logger = logging.get_logger()
#Local
import qsoconfig, qso, qsorun
from doubleprint import DoublePrint

def sigterm_handler(_signo, _stack_frame):
    print("Terminated")
    raise KeyboardInterrupt("Terminated")
    #sys.exit(0)

def sigkill_handler(_signo, _stack_frame):
    print("Killed")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

log_level=logging.WARN
logger.setLevel(log_level)
base_name = ''

# Load the configuration (from file)
log_level = logging.DEBUG
logger.setLevel(log_level)

param_fname = 'params-quant_self_opt.ini'

optim = qsoconfig.gen_config(param_fname)
cfg = optim.config
dyn = optim.dynamics
tc = optim.termination_conditions
fid_comp = dyn.fid_computer
#p_gen = optim.pulse_generator


dir_ok, cfg.output_dir, msg = cfg.check_create_output_dir(cfg.output_dir)
if not dir_ok:
    raise RuntimeError(msg)

if optim.dump:
    optim.dump.dump_dir = cfg.output_dir
if dyn.dump:
    dyn.dump.dump_dir = cfg.output_dir

print("Output files will be saved to:\n{}".format(cfg.output_dir))
if cfg.use_param_file:
    # Copy the parameters file to the output directory for reference
    param_ref_fpath = os.path.join(cfg.output_dir, cfg.param_fname)
    shutil.copyfile(cfg.param_fpath, param_ref_fpath)
    print("Config parameters copied to:\n{}".format(param_ref_fpath))

datetimestamp = datetime.datetime.now().strftime('%d%b_%H-%M')
cfg_str = qso.get_cfg_str(optim, full=cfg.ext_mp)
script_name = "{}-{}.{}".format(cfg_str, datetimestamp, cfg.out_file_ext)
script_path = os.path.join(cfg.output_dir, script_name)
if cfg.double_print:
    sys.stdout = DoublePrint(script_path)

#Useful check when doing stuff via ssh and the like
filename = inspect.getfile(inspect.currentframe())
mod_time = datetime.datetime.fromtimestamp(os.stat(filename).st_mtime).strftime('%d%b-%H:%M:%S')
print("\nRunning file {} (last edited at {})".format(filename, mod_time))
print("Terminal output saved to {}".format(script_path))

try:
    if cfg.mp_opt is None or len(cfg.mp_opt) == 0:
        combres = qsorun.run_qso_sims_mp(optim, lfh=sys.stdout)
    elif cfg.mp_opt.lower() == 'num_tslots':
        qsorun.run_qso_sims_tslot_range(optim, lfh=sys.stdout)
    elif cfg.mp_opt.lower() == 'numer_acc_limit':
        qsorun.run_qso_sims_numer_acc_limit(optim, lfh=sys.stdout)
    else:
        raise ValueError("No option for mp_opt={}".format(cfg.mp_opt))

except KeyboardInterrupt as e:
    if isinstance(sys.stdout, DoublePrint):
        print("\nProcessing interrupted\n")
        try:
            sys.stdout.close()
        except:
            pass
    raise e

if isinstance(sys.stdout, DoublePrint):
    print("Closing log")
    try:
        sys.stdout.close()
    except:
        pass
