"""
Step 3: 2B vs 4B Scaling Experiment (Part 1 - Comparative Schemas & VRAM Tracking)
---------------------------------------------------------------------------------
This script benchmarks and compares the baseline 2B model against the 4B model
on our local NVIDIA GeForce RTX 4090 GPU (power-capped at 270W TDP).

Hypothesis being tested:
1. Accuracy & Robustness: Does scaling to 4B resolve the adversarial prompt injection
   failure mode observed in Task 2 on the 2B model?
2. Latency Penalty: How much does median (p50) and tail (p95) latency increase when
   moving from 2B to 4B on a power-capped workstation card?
3. Memory Footprint: What is the exact VRAM overhead of 4B vs 2B in bfloat16 precision?

Part 1 establishes:
- VRAM profiling utilities using PyTorch CUDA memory management.
- Data structures for side-by-side model comparison across tasks.
- A clean model loading and initialization wrapper.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel

# Ensure current directory is on sys.path for direct imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Clean, idiomatic imports from our shared benchmarking module
from bench_utils import (
    BenchmarkTask,
    CUDATimer,
    TaskEvaluationResult,
    build_benchmark_suite,
    calculate_percentiles,
    evaluate_single_task,
    warmup_engine,
)

# =====================================================================
# 1. Hardware VRAM Profiling Helper
# =====================================================================


def get_vram_usage_gb(device: torch.device) -> dict[str, float]:
    """
    Measures physical on-device VRAM allocation on the RTX 4090.

    Returns:
    - allocated_gb: Tensor memory actively in use by model weights and activations.
    - reserved_gb: Total memory claimed by the PyTorch caching allocator from the OS.
    """
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}

    # Synchronize to ensure all asynchronous GPU allocations have completed
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    return {"allocated_gb": allocated, "reserved_gb": reserved}


# =====================================================================
# 2. Comparative Data Structures
# =====================================================================


@dataclass
class ModelComparisonRecord:
    """
    Encapsulates the side-by-side performance of a specific model architecture.
    """

    model_name: str
    model_id: str
    parameter_tier: Literal["2B", "4B"]
    vram_allocated_gb: float
    load_time_seconds: float
    results: list[TaskEvaluationResult] = field(default_factory=list)
    latency_stats: dict[str, float] = field(default_factory=dict)


def load_model_and_measure_vram(
    model_id: str,
    parameter_tier: Literal["2B", "4B"],
    model_name: str,
    device: torch.device,
) -> tuple[SystemOneEngine, ModelComparisonRecord]:
    """
    Instantiates a model checkpoint, measures loading time and VRAM footprint,
    and returns an active SystemOneEngine.
    """
    print("\n" + "-" * 60)
    print(f"  Loading [{parameter_tier}] Model: '{model_id}'")
    print("-" * 60)

    # Clean PyTorch GPU cache before loading to measure pure delta
    if device.type == "cuda":
        torch.cuda.empty_cache()

    start_vram = get_vram_usage_gb(device)["allocated_gb"]
    t0 = time.time()

    model = StrandsDeciderModel.load(model_id, device_map=str(device))
    model.eval()

    elapsed = time.time() - t0
    end_vram = get_vram_usage_gb(device)["allocated_gb"]
    vram_delta = end_vram - start_vram

    print(f"✅ Model weights loaded in {elapsed:.2f}s")
    print(
        f"📊 VRAM Footprint: {vram_delta:.2f} GB (Total Allocated: {end_vram:.2f} GB)"
    )

    cfg = EngineConfig(device=str(device), use_prefix_cache=True)
    engine = SystemOneEngine(model=model, config=cfg)

    record = ModelComparisonRecord(
        model_name=model_name,
        model_id=model_id,
        parameter_tier=parameter_tier,
        vram_allocated_gb=vram_delta,
        load_time_seconds=elapsed,
    )

    return engine, record


# =====================================================================
# 3. Model Benchmark & Execution Runner
# =====================================================================


def benchmark_model_tier(
    engine: SystemOneEngine,
    record: ModelComparisonRecord,
    tasks: list[BenchmarkTask],
    timer: CUDATimer,
) -> ModelComparisonRecord:
    """
    Runs the benchmark suite on a model tier, measures warm steady-state latency,
    evaluates decisions against ground truth, and cleans up memory.
    """
    print(
        f"\n🚀 Running benchmark suite on {record.model_name} ({record.parameter_tier})..."
    )

    # 1. Warm up the GPU to purge driver compile overhead
    warmup_engine(engine, timer, n_warmup=2)

    # 2. Execute each task in the suite
    latencies: list[float] = []
    for task in tasks:
        res = evaluate_single_task(engine, task, timer)
        record.results.append(res)
        latencies.append(res.latency_ms)

    # 3. Compute non-parametric latency statistics (p50, p90, p95)
    record.latency_stats = calculate_percentiles(latencies)

    p50 = record.latency_stats.get("p50", 0.0)
    p95 = record.latency_stats.get("p95", 0.0)
    print(f"⚡ Steady-State Latency: p50 (Median) = {p50:.1f} ms | p95 = {p95:.1f} ms")

    # 4. Explicitly free GPU memory so the next model has a clean slate
    print(f"🧹 Unloading {record.parameter_tier} model from GPU memory...")
    del engine.model
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return record


# =====================================================================
# 4. Comparative Scaling Report & Showdown Table
# =====================================================================


def print_scaling_showdown(
    rec_2b: ModelComparisonRecord,
    rec_4b: ModelComparisonRecord,
    device: torch.device,
    power_cap_watts: float = 270.0,
) -> None:
    """
    Renders an engineering showdown table comparing 2B vs 4B across:
    1. VRAM footprint and load times.
    2. p50 and p95 latency percentiles with scaling factor ratios.
    3. Per-task decision accuracy, focusing on the adversarial failure mode.
    """
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    p50_2b = rec_2b.latency_stats.get("p50", 1.0)
    p50_4b = rec_4b.latency_stats.get("p50", 1.0)
    p95_2b = rec_2b.latency_stats.get("p95", 1.0)
    p95_4b = rec_4b.latency_stats.get("p95", 1.0)

    latency_ratio = p50_4b / p50_2b if p50_2b > 0 else 0.0
    vram_ratio = (
        rec_4b.vram_allocated_gb / rec_2b.vram_allocated_gb
        if rec_2b.vram_allocated_gb > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("  STRANDS DECIDER: 2B vs 4B EMPIRICAL SCALING SHOWDOWN")
    print("=" * 80)
    print(f"🖥️  Compute Device     : {gpu_name}")
    print(
        f"⚡ Hardware Envelope  : Power Capped at {power_cap_watts:.1f}W TDP (for safety & efficiency)"
    )
    print("-" * 80)
    print("  HARDWARE & LATENCY COMPARISON")
    print("-" * 80)
    print(
        f"  {'Metric':<25} | {'2B Baseline':<18} | {'4B Scaled':<18} | {'Delta / Ratio'}"
    )
    print(f"  {'-' * 25}-|-{'-' * 18}-|-{'-' * 18}-|-{'-' * 15}")
    print(
        f"  {'VRAM Allocation':<25} | {rec_2b.vram_allocated_gb:>6.2f} GB          | {rec_4b.vram_allocated_gb:>6.2f} GB          | {vram_ratio:.2f}x memory"
    )
    print(
        f"  {'Model Load Time':<25} | {rec_2b.load_time_seconds:>6.2f} s           | {rec_4b.load_time_seconds:>6.2f} s           | +{rec_4b.load_time_seconds - rec_2b.load_time_seconds:.1f} s"
    )
    print(
        f"  {'Median Latency (p50)':<25} | {p50_2b:>6.1f} ms          | {p50_4b:>6.1f} ms          | {latency_ratio:.2f}x latency"
    )
    print(
        f"  {'Tail Latency (p95)':<25} | {p95_2b:>6.1f} ms          | {p95_4b:>6.1f} ms          | {p95_4b / p95_2b:.2f}x tail"
    )
    print("=" * 80)

    print("\n" + "-" * 80)
    print("  PER-TASK ACCURACY & ROBUSTNESS SHOWDOWN")
    print("-" * 80)

    for r2, r4 in zip(rec_2b.results, rec_4b.results):
        print(f"\n▶ [{r2.task_id}] (Family: {r2.family} | Difficulty: {r2.difficulty})")

        # Choice comparison
        if r2.choice_predicted is not None and r4.choice_predicted is not None:
            badge_2b = "✅ PASS" if r2.choice_correct else "❌ FAIL"
            badge_4b = "✅ PASS" if r4.choice_correct else "❌ FAIL"
            print(
                f"  • 2B Choice : {badge_2b} -> '{r2.choice_predicted}' (Conf: {r2.choice_confidence:.2%})"
            )
            print(
                f"  • 4B Choice : {badge_4b} -> '{r4.choice_predicted}' (Conf: {r4.choice_confidence:.2%})"
            )

        # Noul comparison
        if r2.noul_predicted is not None and r4.noul_predicted is not None:
            badge_2b = "✅ PASS" if r2.noul_correct else "❌ FAIL"
            badge_4b = "✅ PASS" if r4.noul_correct else "❌ FAIL"
            print(
                f"  • 2B Noul   : {badge_2b} -> {r2.noul_predicted} (Conf: {r2.noul_confidence:.2%})"
            )
            print(
                f"  • 4B Noul   : {badge_4b} -> {r4.noul_predicted} (Conf: {r4.noul_confidence:.2%})"
            )

        # Score comparison
        if r2.score_predicted is not None and r4.score_predicted is not None:
            print(
                f"  • 2B Score  : {r2.score_predicted:.2f} (Expected: {r2.score_expected:.2f}, MAE: {r2.score_abs_error:.2f})"
            )
            print(
                f"  • 4B Score  : {r4.score_predicted:.2f} (Expected: {r4.score_expected:.2f}, MAE: {r4.score_abs_error:.2f})"
            )

    print("\n" + "=" * 80)


# =====================================================================
# 5. Main Execution Entrypoint
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  Initializing Step 3: 2B vs 4B Scaling Experiment")
    print("=" * 80)

    # 1. Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timer = CUDATimer(device)
    tasks = build_benchmark_suite()

    # 2. Benchmark Model 1: 2B Baseline
    MODEL_2B_ID = "StrandsAgents/strands-decider-2B-hobson-v21"
    engine_2b, record_2b = load_model_and_measure_vram(
        model_id=MODEL_2B_ID,
        parameter_tier="2B",
        model_name="Strands Decider 2B (Qwen3.5 Torso)",
        device=device,
    )
    record_2b = benchmark_model_tier(engine_2b, record_2b, tasks, timer)

    # 3. Benchmark Model 2: 4B Scaled Model
    MODEL_4B_ID = "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
    engine_4b, record_4b = load_model_and_measure_vram(
        model_id=MODEL_4B_ID,
        parameter_tier="4B",
        model_name="Strands Decider E4B (Gemma 4 Torso)",
        device=device,
    )
    record_4b = benchmark_model_tier(engine_4b, record_4b, tasks, timer)

    # 4. Generate and display the comprehensive side-by-side report
    print_scaling_showdown(record_2b, record_4b, device=device, power_cap_watts=270.0)
