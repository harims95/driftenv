# DriftEnv Build Progress Log

## Apr 25 evening session — full summary

### Completed (all on branch `multi-reward-v2`, NOT merged to main)

#### Multi-reward decomposition (`server/app.py`)
- [x] Replaced single `_score()` with 4 independent reward components:
  - `R_format` (0.1) — penalises verbose responses (>200 chars = 0.5, >500 = 0)
  - `R_interpretation` (0.3) — keyword overlap with `hidden_interpretation`
  - `R_pivot` (0.4) — keyword overlap with `correct_pivot` + lexical distance from step-1 response
  - `R_no_stale` (0.2) — penalises wrong-pivot echoes and step-1 repetition
- [x] Weighted total: `0.1·format + 0.3·interp + 0.4·pivot + 0.2·no_stale`
- [x] `prev_responses` tracked in `_state` for pivot/no_stale scoring
- [x] All 4 components logged in `info["rewards"]` per step — training plots ready

#### Anti-keyword-hack patch (`server/app.py`)
- [x] `_extract_unique_keywords(target, exclusion)` — strips words visible in
  `initial_instruction` + `context_shift` from keyword pools
- [x] Closes echo exploit: agent cannot game score by copying instruction words
- [x] Verified: v1 overall 0.436 → v2 overall 0.456 (+0.020), no reward sparsity

#### Holdout split (`server/dataset.json`, `server/app.py`, `inference.py`)
- [x] 5 holdout scenarios tagged `"holdout": true` — IDs **1, 3, 7, 14, 20**
  (one per domain, cleanest drift signals)
- [x] Remaining 20 scenarios tagged `"holdout": false`
- [x] `reset(holdout_only=False)` — default, samples from 20 training scenarios only
- [x] `reset(holdout_only=True)` — samples from 5 holdout scenarios only
- [x] `HOLDOUT_ONLY` env var in `inference.py` toggles eval mode
- [x] Zero leakage verified (10 default resets, 20 holdout resets)

#### Baseline files (`samples/`)
- [x] `samples/baseline_local.json` — dumb agent / 401 fallback, overall ~0.278
- [x] `samples/baseline_local_v2.txt` — after anti-hack patch, overall 0.456
- [x] `samples/baseline_train_v3.txt` — training set (20 scenarios), overall **0.327**
- [x] `samples/baseline_holdout_v3.txt` — holdout set (5 scenarios), overall **~0.378–0.429**

#### Training notebook (`training/driftenv_grpo_training.ipynb`)
- [x] 22-cell stub at `training/driftenv_grpo_training.ipynb`
- [x] Cells 1–4: title, installs, imports, config + Space pre-warm
- [x] Cells 5–6: Unsloth model load + LoRA (4-bit, rank-16, Qwen2.5 target modules)
- [x] Cells 7–8: DriftEnv HTTP client (`reset_env`, `step_env`, `holdout_only` support)
- [x] Cells 9–10: rollout/reward function stub (`for_inference` reminder included) **← FILL IN**
- [x] Cells 11–12: `GRPOConfig` (max_steps=150, batch 4, num_generations 4)
- [x] Cells 13–14: prompt dataset builder stub **← FILL IN**
- [x] Cells 15–16: `GRPOTrainer` + `trainer.train()` (commented out until 10+14 done)
- [x] Cells 17–18: LoRA adapter save + `push_to_hub`
- [x] Cells 19–20: holdout eval stub, 3-number comparison **← FILL IN**
- [x] Cells 21–22: matplotlib stubs — reward curves + before/after bar chart

#### README (`README.md`)
- [x] HF Spaces frontmatter preserved
- [x] Technical sections filled in (reward table, training details, quickstart)
- [x] Narrative sections left as `<!-- WRITE THIS -->` — Hariharan writes those
- [x] Plot embeds reference `assets/before_after.png` and `assets/reward_curves.png`
- [x] Results table has 72B reference (0.378) — trained/untrained 1.5B filled tomorrow

### Baseline summary (untrained Qwen 72B, multi-reward-v2 env + anti-hack patch)

| split | easy | medium | hard | overall |
|-------|------|--------|------|---------|
| training (20 scenarios) | 0.200 | 0.397 | 0.385 | **0.327** |
| holdout (5 scenarios)   | 0.307 | 0.370 | 0.611 | **~0.378–0.429** |

Best single step: holdout hard step 2 = **0.81** (scenario 20, serverless pivot)

### Branch state
- `main` — untouched, 1 commit (original v1), HF Space serving v1
- `multi-reward-v2` — 9 commits ahead of main, all tonight's work

### Files changed tonight
- `server/app.py` — multi-reward + anti-hack + holdout_only
- `server/dataset.json` — holdout flags on all 25 scenarios
- `inference.py` — HOLDOUT_ONLY env var
- `training/driftenv_grpo_training.ipynb` — 22-cell notebook stub
- `README.md` — scaffolded with technical sections
- `samples/` — 4 baseline files
- `assets/` — directory created (plots added tomorrow)
- `PROGRESS.md` — this file

### Files NOT changed
- `main` branch — untouched
- `inference.py` LLM target — still Qwen 72B (fine for baselines)

---

## Apr 26 — tomorrow's plan

### First action (8 AM)
Open `training/driftenv_grpo_training.ipynb` in Google Colab.
Set runtime to **T4 (free)**. Do NOT use A10G until dry run passes.

### Critical cells to fill before running
1. **Cell 10** — `driftenv_reward_fn`: call `step_env`, unpack `info["rewards"]`,
   return scalar, append to `reward_log`
2. **Cell 14** — `build_prompt_dataset`: loop `reset_env(holdout_only=False)`,
   format observation as prompt, return `Dataset`
3. **Cell 20** — `eval_on_holdout`: run both untrained and trained model against
   holdout scenarios, collect 4 components, save JSON

### Dry run checklist (T4, max_steps=3)
- [ ] Cell 2 installs complete without error
- [ ] Cell 6 prints trainable params (~10–15M)
- [ ] Cell 8 smoke test returns an instruction
- [ ] Cell 16 completes 3 steps, reward is non-zero float, no NaN
- [ ] `reward_log` has 3 entries after dry run

### After dry run passes → switch to A10G
- Change `max_steps=150`, re-run cells A–F
- Set phone timer, monitor every 30 min
- Stop if reward flat for 20+ steps — debug on T4

### Three numbers to record on holdout set
1. Untrained Qwen 1.5B (run before training, `HOLDOUT_ONLY=true`)
2. **Trained Qwen 1.5B — the headline metric**
3. Qwen 72B reference: **~0.378** (already recorded)

### Submission checklist (target: 4:30 PM hard stop)
- [ ] `assets/reward_curves.png` committed
- [ ] `assets/before_after.png` committed
- [ ] README narrative sections written (hook, problem, what we learned)
- [ ] README results table filled (3 numbers)
- [ ] README links added (HF Hub adapter, Colab, YouTube)
- [ ] Demo video recorded and uploaded (90 sec)
- [ ] `multi-reward-v2` merged to `main`
- [ ] HF Space rebuilt and live
- [ ] Submission form submitted by 4:30 PM
