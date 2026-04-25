---
title: DriftEnv
emoji: 🌊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# DriftEnv

<!-- WRITE THIS: one-line hook in your voice -->

<!-- HERO: add GIF or screenshot tomorrow after recording demo -->

---

## The problem

<!-- WRITE THIS: 2-3 sentences, personal observation. Why you built this.
     "I've watched AI agents confidently follow stale instructions..."
     Judges weight storytelling at 30%. This paragraph is yours. -->

---

## What DriftEnv does

DriftEnv is an [OpenEnv](https://github.com/openenv)-compliant RL environment
that stress-tests AI agents on two real failure modes in ML workflows:

1. **Ambiguity** — the agent receives a deliberately vague instruction and must
   commit to a specific technical interpretation without asking clarifying questions.
2. **Context shift** — mid-task, requirements change. Does the agent pivot, or does
   it ignore the update and keep executing the original (now wrong) plan?

Episodes run across **three difficulty tiers**:

| Tier | Steps | Tests |
|------|-------|-------|
| `easy` | 1 | Interpret a vague instruction |
| `medium` | 2 | Interpret, then pivot on context shift |
| `hard` | 3 | Interpret, pivot, and produce a complete solution holding both in mind |

**25 scenarios** drawn from five real ML workflow domains: dataset preparation,
model selection, training configuration, evaluation criteria, and deployment
requirements. Five scenarios are held out for unseen evaluation (one per domain).

---

## How we made it teachable for RL

A single scalar reward wasn't enough — it gave no gradient signal about *why*
a response was wrong. We decomposed the step reward into **four independent components**:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| `R_format` | 0.1 | Conciseness — < 200 chars scores 1.0, < 500 scores 0.5, longer scores 0 |
| `R_interpretation` | 0.3 | Keyword overlap with the hidden correct interpretation (unique keywords only — words visible in the instruction are excluded to close the echo exploit) |
| `R_pivot` | 0.4 | Keyword overlap with the correct pivot **plus** lexical distance from the agent's own step-1 response — proving it actually changed course, not just got lucky |
| `R_no_stale` | 0.2 | Penalises responses that echo known wrong pivots or repeat step-1 content — the anti-reward-hacking signal |

**Total step reward = 0.1·R_format + 0.3·R_interpretation + 0.4·R_pivot + 0.2·R_no_stale**

The `R_pivot` and `R_no_stale` components specifically fight reward hacking:
an agent cannot score well by copying the visible instruction, by repeating
its prior response, or by pattern-matching to known wrong paths.
All four components are logged per step so training plots show four independent
learning curves rather than a single noisy scalar.

---

## Results

![Before vs After GRPO training on holdout scenarios](assets/before_after.png)

*Trained Qwen 1.5B vs untrained 1.5B vs Qwen 72B reference — 5 holdout scenarios unseen during training.*

<!-- FILL IN TOMORROW: headline number -->
<!-- e.g. "Trained Qwen 1.5B reaches X on holdout, within Y% of the untrained 72B
     reference at 1/50th the parameter count." -->

| Model | Holdout overall |
|-------|----------------|
| Qwen 72B — untrained (reference) | ~0.378 |
| Qwen 1.5B — untrained | <!-- fill in tomorrow --> |
| **Qwen 1.5B — trained (GRPO, 150 steps)** | **<!-- headline → fill in tomorrow -->** |

![GRPO reward component curves over training](assets/reward_curves.png)

*Four independent reward curves over 150 training steps. R_pivot (weight 0.4) carries the primary training signal.*

---

## What we learned

<!-- WRITE THIS: 2-3 honest observations that show engineering maturity.
     Judges love this section. This is yours — don't let an LLM write it.
     Things to draw from (in your own words):
     - Discovering the keyword echo exploit mid-build, patching it, seeing scores improve
     - Adding the holdout split after recognising the overfitting risk with 25 scenarios
     - Any surprises in how reward weights behaved during training -->

---

## What we'd do with more time

<!-- WRITE THIS: brief bullets, your priorities -->
<!-- Options to pick from:
     - Procedural scenario generation (not hand-authored)
     - Adversarial drift (prompt-injection-flavoured context shifts)
     - LLM-as-judge for richer process supervision signal
     - More domains beyond ML workflows
     - Multi-agent variant: one agent introduces drift, another detects it -->

---

## Training details

- **Algorithm:** GRPO (Group Relative Policy Optimization) via [HF TRL](https://github.com/huggingface/trl)
- **Model:** Qwen2.5-1.5B-Instruct, 4-bit quantized via [Unsloth](https://github.com/unslothai/unsloth)
- **LoRA rank:** 16 — target modules: q/k/v/o proj + gate/up/down proj
- **Steps:** 150 — batch 4, 4 generations per prompt (16 rollouts/step)
- **Training scenarios:** 20 (5 holdout reserved, never seen during training)
- **GPU:** A10G

---

## Quickstart

```bash
pip install openenv-core pydantic requests

# Hit the live Space
curl -X POST https://harims95-driftenv.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "medium"}'

curl -X POST https://harims95-driftenv.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"response": "your response here"}}'
```

Or run locally:

```bash
git clone https://github.com/harims95/driftenv
cd driftenv
pip install -r server/requirements.txt
python server/app.py
```

---

## Links

- **Live Space:** https://huggingface.co/spaces/harims95/driftenv
- **Trained adapter:** <!-- add HF Hub link after push_to_hub tomorrow -->
- **Training notebook:** <!-- add Colab link tomorrow -->
- **Demo video:** <!-- add YouTube link tomorrow -->

---

## Built by

Hariharan ([@harims95](https://github.com/harims95)) — solo entry,
Meta × HuggingFace × PyTorch OpenEnv Hackathon Grand Finale,
April 25–26 2026, Bangalore.
