# Skew-Tolerant Power-Flow Invariants for IEC 104 Intrusion Detection

This repository implements the skew-tolerant interval residual (STIR) for power-flow invariants and a protocol detector for IEC 60870-5-104. The reproduction driver evaluates both on Sherlock v3.

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
| `ablation` | Plain residual and moving-average baselines in `results/ablation/` |
| `sensitivity` | Window, persistence, and validation-tolerance sweeps in `results/sensitivity/` |
| `noise` | Three noise realizations in `results/noise/`; use `--seed 0`, `1`, or `2` to run one realization |
| `figures` | Six vector PDFs in `results/figures/`, using results from the main, ablation, sensitivity, and noise stages |
| `localization` | First-alarm checks for drift-off attacks, printed to standard output |
| `dataset-stats` | Scenario statistics in `results/dataset_stats.json` |

Run `main` before ablation, sensitivity, noise, or localization. If you run 03-Rural by itself, use the same output directory as a completed 01-Basic run. The setup for 03-Rural uses the 01-Basic measurement conventions and thresholds, a 10.5 s reporting-gap bound, and a 600 s capture prefix to identify the control center.

The main metrics are `peralert` and `combined` in each `net_*.json`. These correspond to detection by individual alert onset and by the onset of the merged alert stream, respectively. The main detector's per-alert counts are 17/18, 27/29, and 27/28 with zero false alarm onsets. For 02-Semiurban, the merged count is 20/29.

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
