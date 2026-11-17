"""Measure actual end-to-end turn compute including audio decode and response policy."""
import argparse
import json
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_interaction.data import ROOT, load_rows, media_index
from robot_interaction.engine import Engine
from robot_interaction.train import parameter_count




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=100)
    args = parser.parse_args()
    if args.turns < 1:
        parser.error("turns must be positive")
    process = psutil.Process()
    peak = [0]
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                processes = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
                peak[0] = max(peak[0], rss)
            except psutil.Error:
                pass
            stop.wait(0.01)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    started = time.perf_counter()
    engine = Engine()
    load_ms = (time.perf_counter()-started)*1000
    rows = load_rows("test")
    index = media_index()["test"]
    rows = rows[rows.clip_id.isin(index)].head(args.turns)
    if rows.empty:
        raise RuntimeError("No test clips found")
    first = rows.iloc[0]
    cold_result = engine.predict(first.Utterance, index[first.clip_id], "fusion", first.clip_id)
    by_mode = {}
    for mode in ("text", "audio", "fusion"):
        latency, actual = [], []
        for row in rows.itertuples():
            result = engine.predict(row.Utterance, index[row.clip_id], mode, row.clip_id)
            latency.append(result["latency_ms"])
            actual.append(row.clip_id)
        by_mode[mode] = {"turns": len(latency), "p50_ms": float(np.percentile(latency,50)),
                         "p95_ms": float(np.percentile(latency,95)), "max_ms": max(latency),
                         "clip_ids": actual}
    stop.set()
    watcher.join()
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu_model)
    quota = Path("/sys/fs/cgroup/cpu.max")
    report = {"hardware": {"cpu": cpu_model, "logical_cpus": psutil.cpu_count(),
              "affinity_cpus": len(process.cpu_affinity()) if hasattr(process,"cpu_affinity") else None,
              "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
              "system_memory_gib": psutil.virtual_memory().total/2**30, "gpu_used": False,
              "platform": platform.platform(), "python": platform.python_version()},
              "model_load_ms": load_ms, "first_fusion_turn_ms": cold_result["latency_ms"],
              "observed_peak_parent_and_children_rss_mib": peak[0]/2**20,
              "memory_sampling_interval_ms":10, "model_bundle_mib": (ROOT/"artifacts/model.joblib").stat().st_size/2**20,
              "inference":by_mode,
              "definition":"Presegmented turn available on disk to structured state plus full template reply. Includes FFmpeg decode, acoustic features, text transform, classifier and policy. Excludes clip duration, endpoint detection, supplied transcription, model startup, WebSocket/network/browser overhead and queuing. Local files may be OS cached. Single sequential client, one BLAS thread.",
              "interaction_target_ms":500, "parameters": {"total": parameter_count(engine.bundle)["total_required_for_demo"], "components": parameter_count(engine.bundle)["components"], "limit": 6_000_000_000, "notes": parameter_count(engine.bundle)["note"]}}
    (ROOT/"reports/benchmark.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"latency":{m:{k:v for k,v in r.items() if k!='clip_ids'} for m,r in by_mode.items()},
                      "peak_mib":report["observed_peak_parent_and_children_rss_mib"],"parameters":report["parameters"]},indent=2))


if __name__ == "__main__":
    main()
