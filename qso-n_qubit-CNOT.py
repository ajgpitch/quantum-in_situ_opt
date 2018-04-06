"""
Script for running simulations of in-situ quantum gate optimisation
Specifically the optimisation of

The script loads its configuration from the local file:
    params-quant_self_opt.ini
These can also be overridden in some cases with cmdline args
The majority of the code is located in modules.
See qsoconfig.py for descriptions of the config parameters

The script can be used to generate the results from
https://arxiv.org/abs/1701.01723
"""

# started 2015 Oct 6
# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

#Python Core
import inspect, os, sys
import shutil
import datetime
#QuTiP
import qutip.logging_utils as logging
logger = logging.get_logger()
#Local
import qsoconfig, qso, qsorun
from doubleprint import DoublePrint

log_level=logging.WARN
logger.setLevel(log_level)
base_name = ''

# Load the configuration (from file)
log_level = logging.DEBUG
logger.setLevel(log_level)

param_fname = 'params-quant_self_opt-3q_ising_chain.ini'

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
