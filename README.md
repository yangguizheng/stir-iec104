# Skew-Tolerant Power-Flow Invariants for IEC 104 Intrusion Detection

This repository implements the skew-tolerant interval residual (STIR) for power-flow invariants and a passive protocol detector for IEC 60870-5-104. Version 1.1.0 includes the Sherlock v3 detector, extended evaluations, and archived outcomes for reproducing the reported comparisons.

## Installation

Use Python 3.11. Create a virtual environment.

```sh
python -m venv .venv
```

On Linux and macOS, activate with `source .venv/bin/activate`. In Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`. Then install the dependencies.

```sh
python -m pip install -r requirements.txt
```

## Data

Get [Sherlock v3](https://doi.org/10.5281/zenodo.18467070). Extract `01-Basic.zip`, `02-Semiurban.zip`, and `03-Rural.zip` into a common directory, such as `data/sherlock`. Each ZIP contains its own scenario directory. The three downloads total about 8.9 GB. The experiments read the packaged IPAL state files, event labels, test configurations, and PCAPs.

## Reproduce the experiments

Start with the complete detector on 01-Basic.

```sh
python reproduce.py main --scenario 01 --data-dir data/sherlock --out-dir results
```

Run the main experiments, ablations, parameter and noise sweeps, figures, and localization for all three scenarios with:

```sh
python reproduce.py all --data-dir data/sherlock --out-dir results
```

The same data and output options work for these individual stages:

| Stage | Result |
| --- | --- |
| `main` | Physical and combined metrics, score arrays, alarm streams, and per-event CSVs in `results/main/` |
| `ablation` | Plain residual, moving-average and linear-interpolation baselines in `results/ablation/` |
| `sensitivity` | Window, persistence, and validation-tolerance sweeps in `results/sensitivity/` |
| `noise` | Three noise realizations in `results/noise/`; use `--seed 0`, `1`, or `2` to run one realization |
| `figures` | Six vector PDFs in `results/figures/`, using results from the main, ablation, sensitivity, and noise stages |
| `localization` | First-alarm checks for drift-off attacks, printed to standard output |
| `dataset-stats` | Scenario statistics in `results/dataset_stats.json` |

Run `main` before ablation, sensitivity, noise, or localization. If you run 03-Rural by itself, use the same output directory as a completed 01-Basic run. The setup for 03-Rural uses the 01-Basic measurement conventions and thresholds, a 10.5 s reporting-gap bound, and a 600 s capture prefix to identify the control center.

The main metrics are `peralert` and `combined` in each `net_*.json`. These correspond to detection by individual alert onset and by the onset of the merged alert stream, respectively. The main detector's per-alert counts are 17/18, 27/29, and 27/28 with zero false alarm onsets. For 02-Semiurban, the merged count is 20/29.

## Extended experiments

After installing the requirements, rebuild the comparison tables and figures from the distributed results:

```sh
python extended.py reference-report --out-dir results
```

This command needs no Sherlock download. It creates five LaTeX tables, three PDF figures and the sufficient-condition counts under `results/reference-report/`. The file `reference_results.json.gz` contains baseline counts, operating curves, held-out fold results, noise results for seeds 0–19, state-estimation statistics and one outcome per recorded numerical perturbation trial. There are 28,500 such outcomes across three grids. The perturbation tables and plots are regenerated from those records. The release does not generate the perturbed time series.

To recompute the detector and extended analyses on Sherlock v3, run:

```sh
python extended.py all --data-dir data/sherlock --out-dir results
```

This command runs the main detector and all three alignment baselines, then prepares measurement and protocol caches, computes operating curves, evaluates six held-out folds in each training scenario, runs 20 noise realizations at each noise level, fits the state-estimation baseline, evaluates ramp rate and rural transfer, and writes the comparisons to `results/run-report/`. State estimation fits the models over the entire recordings and is the most time-consuming stage. The report uses the distributed perturbation outcomes for that comparison.

Individual stages use the same arguments. Run `cache` before `heldout`, `noise`, `operating-curves` or `state-estimation`. Use `reproduce.py main` before `noise`, `ramp` or `transfer`, and run both `reproduce.py main` and `reproduce.py ablation` before `operating-curves`. Use one output directory per dataset version. The reference report uses temporary inputs and can be built independently of any raw-data runs.

## Optional SIMPLE baseline

Install the additional dependencies and check out the baseline framework at the specified revision.

```sh
python -m pip install -r requirements-simple.txt
git clone https://github.com/fkie-cad/ipal_ids_framework.git third_party/ipal_ids_framework
git -C third_party/ipal_ids_framework checkout 0d71866b777deb13bd2ee038d5058c2425ec4568
python reproduce.py simple --data-dir data/sherlock --out-dir results --ipal-dir third_party/ipal_ids_framework
```

This runs the SIMPLE baseline on all state features of 01-Basic and 02-Semiurban, using the Sherlock event labels for scoring.

## References

- [Sherlock dataset v3](https://zenodo.org/records/18467070) and [dataset documentation](https://sherlock.wattson.it/)
- [IPAL IDS Framework](https://github.com/fkie-cad/ipal_ids_framework)
