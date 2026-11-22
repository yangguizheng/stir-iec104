"""Summarize detector counts and alarm durations across the 20 noise realizations."""
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import OUTPUT


def main(destination):
    runs = [json.loads((OUTPUT / "noise20" / f"seed{n}.json").read_text()) for n in range(20)]
    summary = {}
    for sigma in ("0.005", "0.01"):
        counts = []
        starts = []
        durations = []
        benign_starts = 0
        for run in runs:
            scenarios = run[sigma].values()
            counts.append(sum(row["detected"] for row in scenarios))
            starts.append(sum(row["fa"] for row in scenarios))
            for row in scenarios:
                durations.extend(row["physical_false_alarm_seconds"])
                benign_starts += row["physical_false_alarms_benign"]
        summary[sigma] = {
            "detections_per_seed_range": [min(counts), max(counts)],
            "false_alarm_onsets": sum(starts),
            "recording_hours": len(runs) * 3 * 12,
            "false_alarm_onsets_per_hour": sum(starts) / (len(runs) * 3 * 12),
            "physical_false_alarm_durations_seconds": sorted(durations),
            "physical_false_alarms_in_benign_events": benign_starts,
        }
    (destination / "noise_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
