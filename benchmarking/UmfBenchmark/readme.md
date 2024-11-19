## Umf benchmarking

### Running a single Umf correlation

E.g. Running correlation 5 with a trial seed 1, at a noise level of 0.0, parallel mode activated and 4 CPUs:
```
python umf_run.py --correlation 5 --trial 1 --noise 0.0 --parallel_mode 1 --ncpus 4
```

### Making an HPC jobfile

Making a jobfile to run all Umf correlations at a noise level 0.0:
```
python umf_make_run_file.py --noise 0.0
```

### Analyzing results

Analyzing a results folder:
```
python umf_results_analysis.py --path [results folder]
```

### Results

![logo](https://raw.githubusercontent.com/ZhongXiang/MultiphaseFlowmultiphaseflowsr/main/benchmarking/UmfBenchmark/results/umf_results.png)