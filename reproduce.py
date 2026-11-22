#!/usr/bin/env python3
"""Reproduce IEC 104 intrusion-detection experiments on Sherlock v3."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


SCRIPTS = Path(__file__).resolve().parent / "scripts"


@dataclass(frozen=True)
class Scenario:
    tag: str
    name: str
    vantage: str
    grid: str
    has_training: bool

    def state(self, data, split):
        return data / self.name / f"{split}.{self.vantage}.state.gz"

    def pcaps(self, data, split):
        return data / self.name / "raw" / split / "pcap"

    def events(self, data):
        return data / self.name / "ipal" / "test" / "events.json"


SCENARIOS = (
    Scenario("01", "01-Basic", "n302", "cigre-mv", True),
    Scenario("02", "02-Semiurban", "n406", "1-MV-semiurb--2-sw", True),
    Scenario("03", "03-Rural", "n402", "1-MV-rural--2-sw", False),
)


def run(script, *arguments):
    subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPTS / script), *(str(arg) for arg in arguments)],
        check=True,
    )


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_dataset(data, scenarios, *, captures=True):
    for scenario in scenarios:
        require_file(scenario.events(data))
        splits = ("train", "test") if scenario.has_training else ("test",)
        for split in splits:
            require_file(scenario.state(data, split))
            if captures and not any(scenario.pcaps(data, split).glob("*.pcap")):
                raise FileNotFoundError(f"No .pcap files found in {scenario.pcaps(data, split)}")


def physics(data, scenario, output, *, skew=10, persist=30, align="interval",
            tolerance=0.01, convention=None, thresholds=None, signs=None,
            noise=0.0, seed=0, scores=None):
    arguments = [
        "--grid", scenario.grid, "--test", scenario.state(data, "test"),
        "--events", scenario.events(data), "--out", output,
        "--skew", skew, "--persist", persist, "--align", align,
        "--valid-rel", tolerance, "--noise", noise, "--noise-seed", seed,
    ]
    if scenario.has_training:
        arguments.extend(("--train", scenario.state(data, "train")))
        if signs is not None:
            arguments.extend(("--signs-from", require_file(signs)))
    else:
        arguments.extend(("--convention-from", require_file(convention),
                          "--thr-from", require_file(thresholds)))
    if scores is not None:
        arguments.extend(("--dump-scores", scores))
    run("m5_physics.py", *arguments)


def protocol(data, scenario, scores, output, *, alarms=None):
    arguments = [
        "--test-pcaps", scenario.pcaps(data, "test"),
        "--state", scenario.state(data, "test"), "--events", scenario.events(data),
        "--phys-scores", require_file(scores), "--out", output,
    ]
    if scenario.has_training:
        arguments.extend(("--train-pcaps", scenario.pcaps(data, "train")))
    else:
        arguments.extend(("--gap-default", "10.5", "--init-seconds", "600"))
    if alarms is not None:
        arguments.extend(("--dump-alarms", alarms))
    run("m5_netchannels.py", *arguments)


def main_experiment(data, output, scenarios):
    destination = output / "main"
    destination.mkdir(parents=True, exist_ok=True)
    settings = destination / "phys_01.json"
    for scenario in scenarios:
        tag = scenario.tag
        print(f"Main experiment: {scenario.name}", flush=True)
        scores = destination / f"scores_{tag}.npz"
        physics(data, scenario, destination / f"phys_{tag}.json", convention=settings,
                thresholds=settings, scores=scores)
        protocol(data, scenario, scores, destination / f"net_{tag}.json",
                 alarms=destination / f"alarms_{tag}.npz")
        run("export_events.py", destination, tag, scenario.events(data), destination / f"events_{tag}.csv")


def ablation_experiment(data, output):
    destination = output / "ablation"
    convention = require_file(output / "main" / "phys_01.json")
    destination.mkdir(parents=True, exist_ok=True)
    for name, skew, align in (("plain", 0, "interval"), ("mean", 10, "mean"), ("interp", 10, "interp")):
        thresholds = destination / f"phys_01_{name}.json"
        for scenario in SCENARIOS:
            tag = f"{scenario.tag}_{name}"
            print(f"Ablation ({name}): {scenario.name}", flush=True)
            scores = destination / f"scores_{tag}.npz"
            physics(data, scenario, destination / f"phys_{tag}.json", convention=convention,
                    thresholds=thresholds, scores=scores, skew=skew, align=align)
            protocol(data, scenario, scores, destination / f"net_{tag}.json")


def sensitivity_experiment(data, output):
    destination = output / "sensitivity"
    convention = require_file(output / "main" / "phys_01.json")
    destination.mkdir(parents=True, exist_ok=True)
    for skew, persist in ((0, 30), (5, 30), (8, 30), (9, 30), (10, 30), (11, 30), (12, 30), (20, 30), (10, 15), (10, 60)):
        suffix = f"skew{skew}_p{persist}"
        thresholds = destination / f"01_{suffix}.json"
        for scenario in SCENARIOS:
            print(f"Sensitivity ({suffix}): {scenario.name}", flush=True)
            physics(data, scenario, destination / f"{scenario.tag}_{suffix}.json", skew=skew,
                    persist=persist, convention=convention, thresholds=thresholds)
    for tolerance in (0.005, 0.02):
        physics(data, SCENARIOS[1], destination / f"02_valrel_{tolerance:g}.json", tolerance=tolerance)


def noise_experiment(data, output, seeds):
    reference = output / "main"
    convention = require_file(reference / "phys_01.json")
    require_file(reference / "phys_02.json")
    for seed in seeds:
        destination = output / "noise" / f"seed-{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for sigma in (0.005, 0.01):
            suffix = f"s{sigma:g}"
            thresholds = destination / f"phys_01_{suffix}.json"
            for scenario in SCENARIOS:
                tag = f"{scenario.tag}_{suffix}"
                print(f"Noise (seed {seed}, sigma {sigma:g}): {scenario.name}", flush=True)
                scores = destination / f"scores_{tag}.npz"
                physics(data, scenario, destination / f"phys_{tag}.json",
                        tolerance=max(0.01, 4 * sigma), convention=convention,
                        thresholds=thresholds, signs=reference / f"phys_{scenario.tag}.json",
                        noise=sigma, seed=seed, scores=scores)
                protocol(data, scenario, scores, destination / f"net_{tag}.json")


def figures(data, output):
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    run(
        "fig_timeline.py", "--alarms",
        *(require_file(output / "main" / f"alarms_{s.tag}.npz") for s in SCENARIOS),
        "--events", *(require_file(s.events(data)) for s in SCENARIOS),
        "--out", destination / "fig_timeline.pdf",
    )
    run("fig_skew.py", "--state", require_file(SCENARIOS[1].state(data, "test")),
        "--out", destination / "fig_skew.pdf")
    run("fig_results.py", "--data-dir", data, "--results-dir", output, "--out-dir", destination)


def localization(data, output):
    for scenario in SCENARIOS:
        print(f"Localization: {scenario.name}", flush=True)
        config = data / scenario.name / "raw" / "test" / "sherlock-config.yml"
        run("locate_first.py", require_file(config), require_file(scenario.events(data)),
            require_file(output / "main" / f"phys_{scenario.tag}.json"))


def simple_baseline(data, output, framework, scenarios):
    require_file(framework / "ipal_iids" / "iids.py")
    for scenario in scenarios:
        print(f"SIMPLE baseline: {scenario.name}", flush=True)
        destination = output / "simple" / scenario.name
        destination.mkdir(parents=True, exist_ok=True)
        labeled = destination / "labeled.state.gz"
        train, test = scenario.state(data, "train"), scenario.state(data, "test")
        with TemporaryDirectory(prefix="stir-simple-", dir=destination) as temp:
            config = Path(temp) / "config.json"
            run("ipal_baselines.py", "make-config", "--state", train,
                "--model-dir", Path(temp) / "model", "--out", config)
            run("run_ipal.py", framework, "--config", config,
                "--train.state", train, "--live.state", test, "--output", labeled)
        run("ipal_baselines.py", "score", "--output", labeled, "--events", scenario.events(data),
            "--out", destination / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("main", "ablation", "sensitivity", "noise", "figures",
                                          "localization", "dataset-stats", "simple", "all"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the three Sherlock scenario directories")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--scenario", choices=[s.tag for s in SCENARIOS], help="Select a single scenario for main or simple")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), help="Select one realization for noise")
    parser.add_argument("--ipal-dir", type=Path, help="Path to the IPAL IDS Framework checkout for simple")
    args = parser.parse_args()
    if args.scenario and args.stage not in ("main", "simple"):
        parser.error("--scenario is supported by main and simple")
    if args.seed is not None and args.stage != "noise":
        parser.error("--seed is supported by noise")
    if args.stage == "simple" and (args.ipal_dir is None or args.scenario == "03"):
        parser.error("simple requires --ipal-dir and a scenario with a training recording")
    data = args.data_dir.resolve()
    output = args.out_dir.resolve()
    scenarios = [s for s in SCENARIOS if not args.scenario or s.tag == args.scenario]

    try:
        if args.stage in ("main", "ablation", "sensitivity", "noise", "simple", "dataset-stats", "all"):
            if args.stage == "simple":
                scenarios = [s for s in scenarios if s.has_training]
            check_dataset(data, scenarios, captures=args.stage not in ("sensitivity", "simple", "dataset-stats"))
        if args.stage in ("main", "all"):
            main_experiment(data, output, scenarios)
        if args.stage in ("ablation", "all"):
            ablation_experiment(data, output)
        if args.stage in ("sensitivity", "all"):
            sensitivity_experiment(data, output)
        if args.stage in ("noise", "all"):
            noise_experiment(data, output, (args.seed,) if args.seed is not None else (0, 1, 2))
        if args.stage in ("figures", "all"):
            figures(data, output)
        if args.stage in ("localization", "all"):
            localization(data, output)
        if args.stage == "dataset-stats":
            output.mkdir(parents=True, exist_ok=True)
            run("dataset_stats.py", data, output / "dataset_stats.json")
        if args.stage == "simple":
            simple_baseline(data, output, args.ipal_dir.resolve(), scenarios)
    except FileNotFoundError as exc:
        parser.exit(1, f"Missing input: {exc}\n")
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, f"Failed while running {Path(exc.cmd[3]).name} (exit {exc.returncode}).\n")


if __name__ == "__main__":
    main()

