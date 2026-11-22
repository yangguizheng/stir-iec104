"""Run extended detector evaluations or rebuild published tables from archived outcomes."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
CASES = ("01", "02", "03")


def run(script, *args, env):
    subprocess.run([sys.executable, "-I", "-B", str(ROOT / "experiments" / script),
                    *map(str, args)], env=env, check=True)


def reference():
    with gzip.open(ROOT / "reference_results.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def materialize(data, path):
    for name, values in data.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(values), encoding="utf-8")


def reports(env, destination):
    destination.mkdir(parents=True, exist_ok=True)
    run("tables.py", destination / "tables", env=env)
    run("curves.py", destination, env=env)
    run("noise_figure.py", destination, env=env)
    run("noise_summary.py", destination, env=env)


def evaluate(stage, env):
    if stage == "cache":
        for case in CASES:
            for split in (("train", "test") if case != "03" else ("test",)):
                run("build_cache.py", case, split, env=env)
            run("make_proto.py", case, env=env)
    elif stage == "heldout":
        for case in CASES[:2]:
            for mode in ("strict", "fixed"):
                for fold in range(6):
                    run("heldout.py", case, fold, 6, mode, env=env)
    elif stage == "state-estimation":
        for case, split, chunks in (("01", "train", 8), ("01", "test", 8), ("03", "test", 12)):
            for topology in ("static", "flows"):
                for chunk in range(chunks):
                    run("se_baseline.py", case, split, topology, chunk, chunks, env=env)
        run("se_eval.py", env=env)
    elif stage == "noise":
        run("noise_seeds.py", 0, 19, env=env)
    elif stage == "operating-curves":
        run("opcurve.py", env=env)
    elif stage == "ramp":
        run("ramp_analysis.py", env=env)
    elif stage == "transfer":
        run("transfer.py", env=env)


def main():
    stages = ("cache", "operating-curves", "heldout", "noise", "state-estimation", "ramp", "transfer")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference-report", "run-report", "all", *stages))
    parser.add_argument("--data-dir", type=Path, default=Path("data/sherlock"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.out_dir.resolve()
    env = dict(os.environ, STIR_DATA=str(args.data_dir.resolve()), STIR_OUTPUT=str(output))
    if args.stage == "reference-report":
        with TemporaryDirectory(prefix="stir-reference-") as tmp:
            materialize(reference(), Path(tmp))
            reports(dict(env, STIR_OUTPUT=tmp), output / "reference-report")
        return
    if args.stage == "all":
        for stage in ("main", "ablation"):
            subprocess.run([sys.executable, "-B", str(ROOT / "reproduce.py"), stage,
                            "--data-dir", env["STIR_DATA"], "--out-dir", str(output)], env=env, check=True)
    for stage in stages if args.stage == "all" else (args.stage,):
        evaluate(stage, env)
    if args.stage in ("run-report", "all"):
        reports(env, output / "run-report")


if __name__ == "__main__":
    main()

