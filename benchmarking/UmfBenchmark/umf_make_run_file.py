import argparse

# Internal imports
import multiphaseflowsr.benchmark.UmfDataset.UmfCorrelation as Umf

# Local imports
from benchmarking import utils as bu
import umf_config as uconfig

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Creates a jobfile to run all Umf correlations at specified noise level.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level.")
config = vars(parser.parse_args())

NOISE_LEVEL = float(config["noise"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

# Expected performances on unistra HPC
# With N_SAMPLES = 1e5 on 1 CPU core -> 40min/10k evaluations
# With 1M expressions -> each run .log -> 400 Mo

N_TRIALS = uconfig.N_TRIALS
ORIGINAL_VAR_NAMES = uconfig.ORIGINAL_VAR_NAMES

# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

# List of correlations to generate
CORR_LIST = [0, 1, 7, 36, 45, 46, 47, 56, 59, 69, 88, 89]

commands = []
# Iterating through Umf correlations
for i_corr in CORR_LIST:
    print("\nCorrelation #%i"%(i_corr))
    # Loading a correlation
    pb = Umf.UmfCorrelation(i_corr, original_var_names=ORIGINAL_VAR_NAMES)
    print(pb)
    # Iterating through trials
    for i_trial in range (N_TRIALS):
        # File name
        command = "python umf_run.py -i %i -t %i -n %f"%(i_corr, i_trial, NOISE_LEVEL)
        commands.append(command)
bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





