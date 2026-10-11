"""
Step 1: Baseline Verification Script (Part 1 - Device Setup & Model Loading)
---------------------------------------------------------------------------
This script verifies that the official pre-trained Strands Decider 2B model
can be loaded onto our local NVIDIA GeForce RTX 4090 GPU and executed.
"""

import time

import torch
from strands_decider.infer import EngineConfig, SystemOneEngine

# Import the core model loader and inference engine from our editable package
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)


def initialize_device() -> torch.device:
    """
    Detects and initializes the compute accelerator.
    We target our local NVIDIA RTX 4090 with 24 GB VRAM via CUDA.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ Compute Device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("⚠️ Warning: CUDA not available. Falling back to CPU.")
    return device


def load_baseline_model(model_id: str, device: torch.device) -> StrandsDeciderModel:
    """
    Downloads and instantiates the Strands Decider 2B checkpoint.
    """
    print(f"\n⏳ Loading checkpoint: '{model_id}'...")
    start_time = time.time()

    model = StrandsDeciderModel.load(model_id, device_map=str(device))
    model.eval()

    elapsed = time.time() - start_time
    print(f"✅ Model successfully loaded in {elapsed:.2f} seconds!")
    return model


# =====================================================================
# PART 2: System One Engine & Typed Decision Inference
# =====================================================================


def create_engine(model: StrandsDeciderModel, device: torch.device) -> SystemOneEngine:
    """
    Wraps the loaded model in the SystemOneEngine.

    The engine manages:
    - Tokenization and context window boundaries.
    - KV-cache reuse when evaluating multiple questions on the same state.
    - Single-forward-pass batching for sub-150ms execution.
    """
    cfg = EngineConfig(device=str(device), use_prefix_cache=True)
    return SystemOneEngine(model=model, config=cfg)


def build_sample_request() -> SystemOneRequest:
    """
    Constructs a realistic scenario from an autonomous agent loop:
    The agent encountered a database error and must make three distinct
    decisions in parallel:
    1. A 'choice' decision (Which tool/action to invoke next).
    2. A 'noul' decision (A binary safety gate: is it safe to auto-retry?).
    3. A 'score' decision (Ordinal grading: what is the severity level?).
    """
    state_content = """
    Agent Execution Context:
    Task: Process customer subscription renewal payment.
    Service: BillingService-v3
    Error Log: [DB_TIMEOUT_408] Read timeout after 5000ms while querying 
               billing_profiles table. Database replica pool latency is elevated (p99: 4200ms).
               Payment transaction TXN_982141 has NOT been submitted to Stripe yet.
    """

    questions = {
        # 1. Discrete Choice: Route among 4 distinct actions
        "route_action": ChoiceQuestion(
            instructions="What is the best immediate recovery action for the agent?",
            criteria={
                "retry_with_backoff": "Transient connection timeout; retry with exponential backoff.",
                "fallback_to_cache": "Read cached customer billing profile from local Redis store.",
                "escalate_to_human": "Database corruption or auth failure requiring manual review.",
                "abort_transaction": "Fatal unrecoverable payment failure; abort transaction.",
            },
        ),
        # 2. Binary NOUL: Yes/No safety classification
        "is_safe_to_retry": NoulQuestion(
            instructions="Is this operation safe to retry automatically without creating duplicate charges?",
            criteria={
                "true": "Transaction has not reached Stripe; retrying is idempotent and safe.",
                "false": "Risk of charging customer twice or producing inconsistent ledger state.",
            },
        ),
        # 3. Ordinal Score: Rated along an ordered rubric (Level 0 to Level 3)
        "incident_severity": ScoreQuestion(
            instructions="Rate the incident severity based on customer impact.",
            criteria=[
                "P3 - Minor (Internal delay, no customer impact)",
                "P2 - Moderate (Transient timeout, retryable)",
                "P1 - High (Payment failure visible to customer)",
                "P0 - Critical (Outage or financial inconsistency)",
            ],
        ),
    }

    return SystemOneRequest(state=state_content.strip(), questions=questions)


def run_inference_and_print(engine: SystemOneEngine, request: SystemOneRequest) -> None:
    """
    Executes the decisions in a single forward pass and prints the
    calibrated probabilities and confidence metrics.
    """
    print("\n" + "=" * 60)
    print("  Running Single-Pass Decision Inference on RTX 4090")
    print("=" * 60)

    # Measure wall-clock latency precisely
    start_time = time.perf_counter()
    response: SystemOneResponse = engine.evaluate(request)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n⚡ Total Inference Latency: {latency_ms:.1f} ms")
    print(
        f"📊 Tokens processed: {response.usage.input_tokens} input | {response.usage.output_tokens} output slots"
    )
    print("-" * 60)

    for q_name, answer in response.answers.items():
        print(f"\n▶ Question: [{q_name}] (Type: {answer.type})")

        if answer.type == "choice":
            print(f"  Selected Option : \033[1;32m{answer.choice}\033[0m")
            print(f"  Confidence      : {answer.confidence:.2%}")
            print("  Probabilities   :")
            for opt, prob in sorted(
                answer.probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {opt:<22}: {prob:.2%}")

        elif answer.type == "noul":
            prob_true = answer.noul
            verdict = "TRUE (Yes)" if prob_true >= 0.5 else "FALSE (No)"
            print(
                f"  Binary Verdict  : \033[1;34m{verdict}\033[0m (P(True) = {prob_true:.2%})"
            )

        elif answer.type == "score":
            print(f"  Predicted Level : \033[1;33m{answer.score:.2f}\033[0m")
            print(f"  Confidence      : {answer.confidence:.2%}")
            print("  Score Legend    :")
            for idx, label in answer.legend.items():
                prob = answer.probabilities.get(idx, 0.0)
                print(f"    [{idx}] {label:<45}: {prob:.2%}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Content Lab: Strands Decider 2B Baseline Verification")
    print("=" * 60)

    # 1. Detect hardware
    device = initialize_device()

    # 2. Specify the official release checkpoint
    MODEL_ID = "StrandsAgents/strands-decider-2B-hobson-v21"

    # 3. Load model weights into GPU memory
    model = load_baseline_model(MODEL_ID, device)

    # 4. Initialize engine
    engine = create_engine(model, device)

    # 5. Build our sample agent request with 3 questions
    request = build_sample_request()

    # 6. Execute and print the results
    run_inference_and_print(engine, request)
