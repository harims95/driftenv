# DriftEnv Build Progress Log

## Apr 25 evening session

### Completed
- [x] Branch `multi-reward-v2` created and pushed to GitHub
- [x] Multi-reward decomposition implemented in server/app.py
  - 4 components: R_format, R_interpretation, R_pivot, R_no_stale
  - Weighted sum: 0.1*format + 0.3*interp + 0.4*pivot + 0.2*no_stale
  - All components logged separately in info["rewards"]
  - `prev_responses` added to `_state` so R_pivot and R_no_stale can compare against step-1 response
- [x] Local server tested via uvicorn on port 7860
- [x] inference.py tested with Qwen2.5-72B baseline
- [x] Baseline output saved to samples/baseline_local.json

### Baseline numbers (Qwen2.5-72B, untrained, multi-reward-v2 env)
- easy: 0.275
- medium: 0.543
- hard: 0.597
- overall: 0.472

This is the floor we need to match or beat tomorrow with trained Qwen 1.5B.

### Files changed this session
- `server/app.py` — replaced single `_score()` with 4-component reward system
- `samples/baseline_local.json` — new file, untrained baseline rollouts

### Files NOT yet changed
- `inference.py` (untouched, still calls 72B via HF router)
- `README.md` (untouched, will rewrite tomorrow afternoon)
- `main` branch (untouched, HF Space still serves v1)

### Patch B (same session, late night)
- [x] `_extract_unique_keywords` helper added — strips words visible in instruction/context_shift from keyword pools
- [x] `_score_interpretation` and `_score_pivot` now use unique keywords only
- [x] Verified: v1 overall 0.436 → v2 overall 0.456 (+0.020), no reward sparsity
- [x] `samples/baseline_local_v2.txt` saved and committed

### Currently broken / known issues
- None

### Task A — holdout split (same session)
- [x] 5 holdout scenarios tagged: IDs 1,3,7,14,20 (one per domain, cleanest drift)
- [x] `reset(holdout_only=False)` — default samples from 20 training scenarios only
- [x] `reset(holdout_only=True)` — samples from 5 holdout scenarios only
- [x] `HOLDOUT_ONLY` env var wired into inference.py
- [x] Zero leakage verified (10 default resets, 20 holdout resets)
- [x] Baselines saved:
  - `samples/baseline_train_v3.txt` — training set overall: **0.327**
  - `samples/baseline_holdout_v3.txt` — holdout set overall: **0.429**
  - Best single step: holdout hard step 2 = 0.81 (scenario 20, serverless pivot)

### Baseline summary (untrained Qwen2.5-72B, multi-reward-v2 + anti-hack patch)
| split | easy | medium | hard | overall |
|-------|------|--------|------|---------|
| training (20) | 0.200 | 0.397 | 0.385 | 0.327 |
| holdout (5)   | 0.307 | 0.370 | 0.611 | 0.429 |

### Tomorrow's first action
Build GRPO training notebook in Colab per CLAUDE.md cells A–F.
T4 dry run first, then switch to A10G for real run.
Use `HOLDOUT_ONLY=false` during training, `HOLDOUT_ONLY=true` for final eval.

---
