import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import argparse

# Internal imports
import multiphaseflowsr.benchmark.UmfDataset.UmfCorrelation as Umf
import multiphaseflowsr

# Local imports
import umf_config as uconfig

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT        = 1

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Runs a Umf correlation job.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--correlation", default = 0,
                    help = "Correlation number in the set (e.g. 1 to 45 for type1 corrs, 1 to 43 for type2 corrs, 1 to 6 for type3 corrs and 1 to 18 for type4 corrs).")
parser.add_argument("-t", "--trial", default = 0,
                    help = "Trial number (sets seed accordingly).")
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level fraction.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Umf correlation number
I_UMF  = int(config["correlation"])
# Trial number
N_TRIAL = int(config["trial"])
# Noise level
NOISE_LEVEL = float(config["noise"])
# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

if __name__ == '__main__':

    # ----- HYPERPARAMS -----
    FIXED_CONSTS       = uconfig.FIXED_CONSTS
    FIXED_CONSTS_UNITS = uconfig.FIXED_CONSTS_UNITS
    FREE_CONSTS_NAMES  = uconfig.FREE_CONSTS_NAMES
    FREE_CONSTS_UNITS  = uconfig.FREE_CONSTS_UNITS
    OP_NAMES           = uconfig.OP_NAMES
    N_SAMPLES          = uconfig.N_SAMPLES
    CONFIG             = uconfig.CONFIG
    MAX_N_EVALUATIONS  = uconfig.MAX_N_EVALUATIONS
    N_EPOCHS           = uconfig.N_EPOCHS
    ORIGINAL_VAR_NAMES = uconfig.ORIGINAL_VAR_NAMES

    # Fixing seed accordingly with attempt number
    seed = N_TRIAL
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Paths
    RUN_NAME       = "UMF_%i_%i_%f"%(I_UMF, N_TRIAL, NOISE_LEVEL)
    PATH_DATA      = "%s_data.csv"%(RUN_NAME)
    PATH_DATA_PLOT = "%s_data.png"%(RUN_NAME)

    # Making a directory for this run and run in it
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME,))

    # Copying .py this script to the directory
    # shutil.copy2(src = __file__, dst = os.path.join(os.path.dirname(__file__), RUN_NAME))

    # MONITORING CONFIG TO USE
    get_run_logger     = lambda : multiphaseflowsr.learn.monitoring.RunLogger(
                                          save_path = 'SR.log',
                                          do_save   = True)
    get_run_visualiser = lambda : multiphaseflowsr.learn.monitoring.RunVisualiser (
                                               epoch_refresh_rate = 1,
                                               save_path = 'SR_curves.png',
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )

    # Loading Umf correlation
    pb = Umf.UmfCorrelation(I_UMF, original_var_names=ORIGINAL_VAR_NAMES)

    # Generate data
    X, y = pb.generate_data_points (n_samples = N_SAMPLES)

    # Noise
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon

    # Save data
    df = pd.DataFrame(data=np.concatenate((y[np.newaxis, :], X), axis=0).transpose(),
                  columns=[pb.y_name] + pb.X_names)
    df.to_csv(PATH_DATA, sep=";")

    # Plot data
    # Set font and style for publication quality
    n_dim = X.shape[0]
    mpl.rcParams.update({
        'font.size': 16,  # Adjust font size for better readability
        'font.family': 'serif',  # Use a serif font for a professional look
        'font.serif': 'Times New Roman',  # Set font to Times New Roman
        'axes.labelsize': 20,  # Size of the x and y labels
        'axes.titlesize': 22,  # Size of the plot title
        'legend.fontsize': 18,  # Size of the legend text
        'xtick.labelsize': 18,  # Size of the x-axis tick labels
        'ytick.labelsize': 18,  # Size of the y-axis tick labels
        'figure.figsize': [10, n_dim * 4],  # Size of the figure
        'savefig.dpi': 300,  # DPI for saving figures
        'savefig.format': 'png',  # Format for saving figures
        'lines.markersize': 3,  # Marker size
        'text.usetex': False,  # Disable LaTeX rendering
    })

    # Create the plot
    fig, ax = plt.subplots(n_dim, 1, figsize=(10, n_dim * 4))
    fig.suptitle(f"Correlation: {pb.formula_original}", fontsize=22, wrap=True, y=0.97, bbox={'facecolor': 'white', 'pad': 10})

    colors = plt.colormaps['tab10']

    pb = Umf.UmfCorrelation(I_UMF, original_var_names=ORIGINAL_VAR_NAMES)

    print(f"Loaded correlation {pb.i_corr}: {pb.corr_name}")
    print(f"Output variable: {pb.y_name}")
    print(f"Formula: {pb.formula_original}")

    X, y = pb.generate_data_points(n_samples=N_SAMPLES)


    for i in range(n_dim):
        curr_ax = ax if n_dim == 1 else ax[i]
        color = colors(i / n_dim)
        curr_ax.scatter(X[i], y, color=color, s=3)
        curr_ax.set_xlabel(f"{pb.X_names[i]} : {pb.X_units[i]}")
        curr_ax.set_ylabel(f"{pb.y_name} : {pb.y_units}")
        curr_ax.grid(True)  # Add grid for better readability
    # Adjust layout to make room for the title and improve spacing between subplots
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    fig.subplots_adjust(hspace=0.3)  # Increase hspace to improve spacing between subplots

    # Save the plot with high resolution
    fig.savefig(PATH_DATA_PLOT, dpi=300, format='png')

    # Printing start
    print("%s : Starting SR task"%(RUN_NAME))

    # Running SR task
    expression, logs = multiphaseflowsr.SR(X, y,
                # Giving names of variables (for display purposes)
                X_names = pb.X_names,
                # Giving units of input variables
                X_units = pb.X_units,
                # Giving name of root variable (for display purposes)
                y_name  = pb.y_name,
                # Giving units of the root variable
                y_units = pb.y_units,
                # Fixed constants
                fixed_consts       = FIXED_CONSTS,
                # Units of fixed constants
                fixed_consts_units = FIXED_CONSTS_UNITS,
                # Free constants names (for display purposes)
                free_consts_names = FREE_CONSTS_NAMES,
                # Operations allowed
                op_names = OP_NAMES,
                # Units of free constants
                free_consts_units = FREE_CONSTS_UNITS,
                # Run config
                run_config = CONFIG,
                # Run monitoring
                get_run_logger     = get_run_logger,
                get_run_visualiser = get_run_visualiser,
                # Stopping condition
                stop_reward = 1.1,  # not stopping even if perfect 1.0 reward is reached
                max_n_evaluations = MAX_N_EVALUATIONS,
                epochs            = N_EPOCHS,
                # Parallel mode
                parallel_mode = PARALLEL_MODE,
                n_cpus        = N_CPUS,
        )

    # Printing end
    print("%s : SR task finished"%(RUN_NAME))

