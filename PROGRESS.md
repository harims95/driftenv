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

### Currently broken / known issues
- None

### Tomorrow's first action
Build GRPO training notebook in Colab per CLAUDE.md cells A–F.
T4 dry run first, then switch to A10G for real run.

---
