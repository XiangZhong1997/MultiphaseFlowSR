# **MultiphaseFlowSR: Discovering Physical Laws in Multiphase Flow Systems**

MultiphaseFlowSR is a symbolic regression package developed to model and discover physical laws in multiphase flow systems. Based on the Physical Symbolic Optimization ($\Phi$-SO) framework, it integrates advanced physics-aware methodologies to address the unique challenges of gas-solid flow modeling.

[![GitHub Repo stars](https://img.shields.io/github/stars/XiangZhong1997/MultiphaseFlowSR?style=social)](https://github.com/XiangZhong1997/MultiphaseFlowSR)  
[![Documentation Status](https://readthedocs.org/projects/multiphaseflowsr/badge/?version=latest)](https://multiphaseflowsr.readthedocs.io/en/latest/?badge=latest)

---

## **Acknowledgments**
This project is inspired by the pioneering work of Wassim Tenachi et al. on the Physical Symbolic Optimization ($\Phi$-SO) framework. Their contributions to integrating dimensional analysis with symbolic regression have laid the foundation for this package. For details, see the original repository: [WassimTenachi/PhySO](https://github.com/WassimTenachi/PhySO).

---

## **Highlights**

Key features of MultiphaseFlowSR include:  
- **Physics-aware symbolic regression**: Incorporates dimensional consistency and unit constraints to improve accuracy.  
- **Noise robustness**: Handles noisy and sparse data effectively for experimental reliability.  
- **Specialized tools**: Tailored for analyzing gas-solid flow behaviors in multiphase systems.  

---

## **Installation**

### Prerequisites
MultiphaseFlowSR supports:
- Linux
- macOS (ARM & Intel)
- Windows

### Setup

#### Virtual Environment and Installation
1. Create a virtual environment: `conda create -n MultiphaseFlowSR python=3.8`  
2. Activate the environment: `conda activate MultiphaseFlowSR`  
3. Clone the repository: `git clone https://github.com/XiangZhong1997/MultiphaseFlowSR`  
4. Navigate to the repository directory: `cd MultiphaseFlowSR`  
5. Install dependencies: `conda install --file requirements.txt`  
6. Install the package: `python -m pip install -e .`

---

## **Running a Benchmark**

### How to Run `umf_run.py`

The `umf_run.py` script runs a symbolic regression task on Umf correlations. It includes data generation, noise handling, visualization, and symbolic regression modeling. 

#### **Command**
```
python umf_run.py -i <correlation_id> -t <trial_id> -n <noise_level> -p <parallel_mode> -ncpus <number_of_cpus>
```

**Parameters**
- -i or --correlation: Select the Umf correlation ID (e.g., for Type 1: 1–45).
- -t or --trial: Specify the trial number (sets the random seed).
- -n or --noise: Set the noise level (default 0.0, no noise).
- -p or --parallel_mode: Enable parallel mode (default False).
- -ncpus or --ncpus: Number of CPUs to use (default 1).

**Examples**
**Basic run:**
```
python umf_run.py -i 1 -t 0 -n 0.1
```
Runs correlation ID 1 with a trial ID of 0, 10% noise level, and no parallel mode.

**Parallel run:**
```
python umf_run.py -i 2 -t 1 -n 0.2 -p True -ncpus 4
```
Runs correlation ID 2, trial ID 1, 20% noise level, parallel mode enabled, and 4 CPUs.

**Outputs**
- Data file: The generated dataset is saved as <RUN_NAME>_data.csv.
- Visualization: Scatter plot of the data is saved as <RUN_NAME>_data.png.
- Symbolic regression results:
- SR.log: Symbolic regression logs.
- SR_curves.png: Convergence curves.

## **Notes**
For large datasets, non-parallel mode is recommended to avoid performance issues.
If configurations need to be customized, modify the umf_config.py file.

## **License**
This project is open-source under the MIT License. See the LICENSE file for details.