"""
DriftEnv — inference.py
"""

import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
HOLDOUT_ONLY = os.getenv("HOLDOUT_ONLY", "false").lower() == "true"
BENCHMARK    = "driftenv"

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert AI/ML engineer working on real AI development tasks.
    You will receive task instructions that may be vague or incomplete.
    Your job at each step:
      1. Carefully interpret what the instruction actually requires technically.
      2. If requirements change mid-task, immediately acknowledge and adapt.
      3. Always give a specific, technically detailed response.
    Be concise but precise. State your interpretation clearly.
    If the instruction is ambiguous, state your interpretation before proceeding.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", " ")[:200]
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def env_reset(task: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task, "holdout_only": HOLDOUT_ONLY}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": {"response": action}}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_action(client: OpenAI, messages: List[dict]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "I need more information to proceed."
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return "I need more information to proceed."

def build_prompt(obs: dict, step_num: int) -> str:
    instruction = obs.get("instruction", "")
    context_shift = obs.get("context_shift")
    history = obs.get("history", [])

    if step_num == 1:
        return f"""You have received this AI development task:
"{instruction}"
This instruction is vague. State your full technical interpretation and approach."""

    elif step_num == 2 and context_shift:
        prev = history[-1]["action"] if history else ""
        return f"""Original task: "{instruction}"
Your previous approach: {prev[:200]}

REQUIREMENTS CHANGED:
"{context_shift}"

Acknowledge what changed. Describe your new approach."""

    else:
        prev_steps = "\n".join([f"Step {h['step']} ({h['phase']}): {h['action'][:150]}" for h in history])
        return f"""Original task: "{instruction}"
Context update: "{context_shift or 'None'}"
Work so far:\n{prev_steps}

Provide your FINAL complete solution."""

def run_task(client: OpenAI, task: str) -> float:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    obs = {}

    try:
        obs = env_reset(task)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step_num in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break
            user_prompt = build_prompt(obs, step_num)
            messages.append({"role": "user", "content": user_prompt})
            action = get_action(client, messages)
            messages.append({"role": "assistant", "content": action})
            result = env_step(action)
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            error = result.get("info", {}).get("error")
            rewards.append(reward)
            steps_taken = step_num
            obs = result.get("observation", obs)
            log_step(step=step_num, action=action, reward=reward, done=done, error=error)
            if done:
                score = float(result.get("final_score") or (sum(rewards) / len(rewards)))
                score = min(max(score, 0.0), 1.0)
                break
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

def main() -> None:
    if not API_KEY:
        print("[DEBUG] Warning: HF_TOKEN not set.", flush=True)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []
    for task in TASKS:
        score = run_task(client, task)
        all_scores.append(score)
        print(f"[DEBUG] Task {task} final score: {score:.3f}", flush=True)
        print("", flush=True)
    overall = sum(all_scores) / len(all_scores)
    print(f"[DEBUG] Overall score: {overall:.3f}", flush=True)

if __name__ == "__main__":
    main()
