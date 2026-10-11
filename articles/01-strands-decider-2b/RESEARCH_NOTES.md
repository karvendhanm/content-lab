# Technical Research Notes: Strands Decider 2B & 4B Exploration

## 1. Primary References & Repositories
* **Official Announcement**: [Introducing Strands Decider 2B: a small, open source, decision model (Strands Agents)](https://strandsagents.com/blog/introducing-strands-decider/)
* **GitHub Repository**: [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider)
* **Hugging Face Model Checkpoints**:
  * `StrandsAgents/strands-decider-2B-hobson-v21`
  * Checkpoints directory: [Hugging Face StrandsAgents Org](https://huggingface.co/StrandsAgents)
* **Key Authors**: Marc Brooker (VP/Distinguished Engineer, AWS), Mike Chambers, Fabio Nonato de Paula.
* **License**: Apache-2.0.

---

## 2. Core Architectural Breakdown ("Hobson" Design)

### The Classical Problem
In typical agent execution:
```text
User / Agent State -> Prompt -> Large LLM (Autoregressive Token Generation) -> Text Output ("tool_x")
```
* **Latency**: 400ms – 2500ms (dominated by token-by-token decoding step-by-step).
* **Cost**: Paid per input & output token on 8B–70B+ models.
* **Uncertainty**: The model produces pseudo-probabilities that are uncalibrated; hallucinated tool calls or format parse errors can derail the execution loop.

### The Strands Decider Alternative
```text
[Context + Question + Option Tokens]
                 │
                 ▼
      Qwen3.5-2B-Base Torso (LoRA rank-16)
                 │
      [Hidden States at <answer> and Candidate Option Tokens]
                 │
                 ▼
        Pointer Head (~1M params)
                 │
                 ▼
   Softmax -> Direct Discrete Probability Distribution
      (Single Forward Pass, ~115ms)
```

### Key Differences
1. **No Language Modeling Head**: The vocabulary projection matrix (e.g. 151,936 $\times$ $d$) is stripped off. It literally cannot output arbitrary text tokens.
2. **Pointer Head vs. Slot Head**:
   - *Slot Head (Failed Early Design)*: Projections mapped final representations to static classes $\{1, 2, \dots, K\}$. This failed because options in agent loops are dynamic strings (e.g., specific tool names or code snippets), not fixed static classes.
   - *Pointer Head (Winning Design)*: Computes bilinear or dot-product compatibility between the question representation (at a designated `<answer>` token) and the token representation at the termination of each option. This handles arbitrary strings and any number of dynamic choices.
3. **Question Types Supported**:
   - `choice`: Given question $Q$ and options $\{O_1, O_2, \dots, O_k\}$, returns $P(O_i)$.
   - `noul`: Binary decision (from Gaelic *no / uil* or boolean classification).
   - `score`: Probability distribution across an ordinal scale $[1 \dots 5]$.

---

## 3. Training Pipeline & Iteration Journey

### The Iteration Timeline
* **Early Versions (v1–v10)**: Exploration of torso models (encoder-decoder vs decoder-only, e.g. T5Gemma experiments) and head designs (slot head vs pointer head). Slot heads exhibited catastrophic accuracy drop on unseen prompt formats.
* **Middle Versions (v11–v18)**: Refinement of the pointer head architecture, transition to `Qwen3.5-2B-Base` torso with rank-16 LoRA adapter.
* **Checkpoint `v19`**: The first robust public release baseline.
* **Checkpoint `v21`**:
  - Addition of human-verified and synthetic question paraphrasing to eliminate prompt sensitivity.
  - Distillation using a `Qwen3.5-4B` teacher model to regularize target probability distributions.
* **Checkpoint `v22` / 4B Exploration**: Exploring larger parameter capacity for high-complexity triage.

### Training Data Mix
Documented in the repository's `data/sources.md`:
* 25 curated Hugging Face public datasets (including code reasoning and structured decision datasets).
* `ContractNLI`: Rigorous legal and condition-matching classification.
* `MuSiQue`: Multi-hop reasoning and passage selection.
* Distillation targets generated from `Qwen3.5-4B` open weights.

---

## 4. The 4B Model Experiment Plan

### Hypothesis
* **Research Question**: Does training a 4B decision model (`Qwen3.5-4B-Base` torso) provide a meaningful improvement in decision accuracy and calibration over the 2B model, or does it hit diminishing returns while doubling latency and memory requirements?
* **Anticipated Trade-offs**:
  - *Capacity*: 4B has wider hidden dimensions and more attention heads, which should improve multi-condition tool routing and subtle edge cases.
  - *Speed*: Latency is expected to scale from ~115ms (2B) to ~200–240ms (4B) on an RTX 3090 / equivalent GPU.
  - *Utility*: If 4B remains $< 250\text{ ms}$, is the extra accuracy worth it for high-stakes agent decisions (e.g., financial or security guardrails)?

### Experimental Blueprint
1. **Model Backbone**: `Qwen/Qwen3.5-4B-Base` vs `Qwen/Qwen3.5-2B-Base`.
2. **Head Configuration**: Identical pointer head (~1M parameters) attached to the 4B hidden dimension ($d_{model}$).
3. **Adapter**: LoRA rank $r=16$, $\alpha=32$, targeting query, key, value, and projection layers.
4. **Data**: Same data inventory recipe (`data/sources.md`).
5. **Evaluation Benchmark**: **JevBench** (and domain agent routing test cases).
6. **Key Evaluation Metrics**:
   - Accuracy / Top-1 Accuracy.
   - Expected Calibration Error (ECE) & Brier Score.
   - Latency Profile (p50, p90, p99 ms).
   - VRAM Memory Footprint during inference.

---

## 5. Article Action Items & Milestones

- [x] Establish `content-lab` vision and repository `README.md`.
- [x] Create detailed narrative outline (`OUTLINE.md`).
- [x] Gather core technical facts, architectural details, and iteration logs (`RESEARCH_NOTES.md`).
- [ ] Inspect code structure of `strands-decider` (loss functions, pointer head PyTorch implementation).
- [ ] Set up the experiment harness / training config for the 4B model.
- [ ] Draft Section 1 & 2 (The Problem & The Hobson Architecture).
- [ ] Draft Section 3 & 4 (Base Model Choice & The Iteration Graveyard).
- [ ] Run the evaluation & compile the 2B vs 4B benchmark comparison.
- [ ] Draft Section 5, 6 & 7 (Experimental Results, Lessons, and The Two-Tier Agent Architecture).
- [ ] Polish final article draft for publication.
