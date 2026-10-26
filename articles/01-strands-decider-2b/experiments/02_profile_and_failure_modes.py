"""
Step 2: Latency Profiling & Failure Mode Analysis
-------------------------------------------------
This script benchmarks the Strands Decider 2B model on our local NVIDIA GeForce
RTX 4090 GPU (power-capped at 270W TDP for workstation safety and efficiency).

Shared evaluation primitives are imported directly from `bench_utils.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel

# Ensure local directory is on sys.path for direct imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bench_utils import (
    BenchmarkTask,
    CUDATimer,
    TaskEvaluationResult,
    build_benchmark_suite,
    calculate_percentiles,
    evaluate_single_task,
    warmup_engine,
)


def print_benchmark_report(
    results: list[TaskEvaluationResult],
    device: torch.device,
    power_cap_watts: float = 270.0,
) -> None:
    """
    Renders an engineering report detailing hardware parameters,
    latency percentiles, per-task accuracy, and failure mode analysis.
    """
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    latencies = [r.latency_ms for r in results]
    stats = calculate_percentiles(latencies)

    print("\n" + "=" * 70)
    print("  STRANDS DECIDER 2B: HARDWARE PROFILING & FAILURE MODE REPORT")
    print("=" * 70)
    print(f"🖥️  Compute Device     : {gpu_name}")
    print(f"⚡ Hardware Envelope  : Power Capped at {power_cap_watts:.1f}W TDP (for safety & efficiency)")
    print(f"📊 Total Tasks Scored : {len(results)}")
    print("-" * 70)
    print("  LATENCY DISTRIBUTION (Steady-State CUDA Events)")
    print("-" * 70)
    print(f"  Min   : {stats.get('min', 0.0):6.1f} ms  |  p50 (Median) : {stats.get('p50', 0.0):6.1f} ms")
    print(f"  p90   : {stats.get('p90', 0.0):6.1f} ms  |  p95          : {stats.get('p95', 0.0):6.1f} ms")
    print(f"  Max   : {stats.get('max', 0.0):6.1f} ms  |  Mean         : {stats.get('mean', 0.0):6.1f} ms")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("  PER-TASK ACCURACY & DECISION AUDIT")
    print("-" * 70)

    for r in results:
        status_badges = []

        if r.choice_correct is not None:
            badge = "✅ PASS" if r.choice_correct else "❌ FAIL"
            status_badges.append(f"Choice: {badge} (pred: '{r.choice_predicted}', exp: '{r.choice_expected}')")

        if r.noul_correct is not None:
            badge = "✅ PASS" if r.noul_correct else "❌ FAIL"
            status_badges.append(f"Noul: {badge} (pred: {r.noul_predicted}, exp: {r.noul_expected})")

        if r.score_abs_error is not None:
            badge = "✅ PASS" if r.score_abs_error <= 0.5 else "⚠️ DRIFT"
            status_badges.append(f"Score: {badge} (pred: {r.score_predicted:.2f}, exp: {r.score_expected:.2f}, MAE: {r.score_abs_error:.2f})")

        verdict_str = " | ".join(status_badges)
        print(f"\n▶ [{r.task_id}] (Family: {r.family} | Difficulty: {r.difficulty})")
        print(f"  Latency: {r.latency_ms:.1f} ms | Tokens: {r.input_tokens}")
        print(f"  Verdict: {verdict_str}")
        conf_details = []
        if r.choice_confidence is not None:
            conf_details.append(f"Choice: {r.choice_confidence:.2%}")
        if r.noul_confidence is not None:
            conf_details.append(f"Noul: {r.noul_confidence:.2%}")
        if r.score_confidence is not None:
            conf_details.append(f"Score: {r.score_confidence:.2%}")
        if conf_details:
            print(f"  Confidence: {' | '.join(conf_details)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("  Initializing Step 2: Latency Profiling & Failure Mode Analysis")
    print("=" * 70)

    # 1. Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timer = CUDATimer(device)

    # 2. Load model
    MODEL_ID = "StrandsAgents/strands-decider-2B-hobson-v21"
    print(f"\n⏳ Loading checkpoint: '{MODEL_ID}'...")
    model = StrandsDeciderModel.load(MODEL_ID, device_map=str(device))
    model.eval()

    # 3. Instantiate Engine
    cfg = EngineConfig(device=str(device), use_prefix_cache=True)
    engine = SystemOneEngine(model=model, config=cfg)

    # 4. Warm up GPU to eliminate cold-start compile noise
    print("\n🔥 Warming up GPU with dummy forward passes...")
    warmup_engine(engine, timer, n_warmup=2)
    print("✅ GPU warmed up and ready for steady-state profiling.")

    # 5. Build and execute test suite
    suite = build_benchmark_suite()
    print(f"\n🚀 Running {len(suite)} benchmark tasks through RTX 4090...")

    evaluation_results: list[TaskEvaluationResult] = []
    for task in suite:
        res = evaluate_single_task(engine, task, timer)
        evaluation_results.append(res)

    # 6. Generate and print the comprehensive report
    print_benchmark_report(evaluation_results, device=device, power_cap_watts=270.0)
