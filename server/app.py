"""
DriftEnv — app.py
"""

import json
import os
import random
from typing import Optional

from pydantic import BaseModel
from openenv.core import Action, Observation, State, Environment

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
with open(DATASET_PATH, "r") as f:
    SCENARIOS = json.load(f)

class DriftEnvObservation(BaseModel):
    instruction: str
    context_shift: Optional[str] = None
    step: int
    history: list
    done: bool
    task: str

class DriftEnvAction(BaseModel):
    response: str

class DriftEnvReward(BaseModel):
    reward: float
    final_score: Optional[float] = None
    phase: str
    feedback: str

_state = {
    "scenario": None,
    "task": "medium",
    "step_count": 0,
    "shift_triggered": False,
    "interpretation_score": None,
    "pivot_score": None,
    "completion_score": None,
    "done": False,
    "history": [],
    "prev_responses": [],  # agent responses per step, used for pivot/no_stale scoring
}

# ---------------------------------------------------------------------------
# Reward component helpers
# ---------------------------------------------------------------------------

def _score_format(response: str) -> float:
    """R_format: reward concise, structured responses."""
    n = len(response)
    if n < 200:
        return 1.0
    if n < 500:
        return 0.5
    return 0.0


def _extract_unique_keywords(target_text: str, exclusion_text: str) -> list:
    """Return words >4 letters in target_text that don't appear in exclusion_text.
    Prevents agents from gaming the score by echoing words from the visible instruction."""
    excl = set(exclusion_text.lower().split())
    return [w for w in target_text.lower().split() if len(w) > 4 and w not in excl]


def _score_interpretation(response: str, hidden_interpretation: str, exclusion_text: str = "") -> float:
    """R_interpretation: keyword overlap with hidden_interpretation, excluding
    words already visible in the instruction/context_shift."""
    resp_lower = response.lower()
    keywords = _extract_unique_keywords(hidden_interpretation, exclusion_text)
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in resp_lower)
    # Linear scale: ratio >= 0.4 → 1.0, proportional below
    return round(min(hits / len(keywords) / 0.4, 1.0), 4)


def _score_pivot(response: str, correct_pivot: str, step1_response: Optional[str], exclusion_text: str = "") -> float:
    """R_pivot: on step >= 2, keyword overlap with correct_pivot (excluding visible
    words) AND lexical distance from the agent's own step-1 response."""
    if step1_response is None:
        return 0.0

    resp_lower = response.lower()

    # (a) keyword overlap with correct pivot — unique words only
    keywords = _extract_unique_keywords(correct_pivot, exclusion_text)
    if keywords:
        hits = sum(1 for kw in keywords if kw in resp_lower)
        kw_score = min(hits / len(keywords) / 0.4, 1.0)
    else:
        kw_score = 0.0

    # (b) lexical distance from step-1 response (0 = identical, 1 = fully different)
    prev_words = set(w for w in step1_response.lower().split() if len(w) > 3)
    curr_words = set(w for w in resp_lower.split() if len(w) > 3)
    if prev_words:
        shared = len(prev_words & curr_words)
        lexical_dist = 1.0 - (shared / len(prev_words))
    else:
        lexical_dist = 1.0

    return round((kw_score + lexical_dist) / 2, 4)


def _score_no_stale(response: str, wrong_pivots: list, step1_response: Optional[str]) -> float:
    """R_no_stale: on step >= 2, penalise responses that match wrong_pivots or
    are too similar to the agent's step-1 response (anti-reward-hacking signal)."""
    if step1_response is None:
        return 1.0  # step 1 — nothing can be stale yet

    resp_lower = response.lower()

    # Hard zero if response echoes a known wrong pivot
    wrong_match = any(w.lower()[:30] in resp_lower for w in wrong_pivots)
    if wrong_match:
        return 0.0

    # Graded penalty for repeating step-1 content
    prev_words = set(w for w in step1_response.lower().split() if len(w) > 3)
    curr_words = set(w for w in resp_lower.split() if len(w) > 3)
    if prev_words:
        shared = len(prev_words & curr_words)
        similarity = shared / len(prev_words)
        if similarity > 0.7:
            return 0.0
        if similarity > 0.4:
            return round(1.0 - similarity, 4)

    return 1.0


def _compute_reward(
    action: str,
    scenario: dict,
    step1_response: Optional[str],
) -> tuple[float, dict]:
    """Compute all 4 reward components and return (weighted_total, components_dict)."""
    # Words visible to the agent — excluded from keyword pools to close the echo exploit
    exclusion_text = scenario["initial_instruction"] + " " + (scenario.get("context_shift") or "")

    r_fmt    = _score_format(action)
    r_interp = _score_interpretation(action, scenario["hidden_interpretation"], exclusion_text)
    r_pivot  = _score_pivot(action, scenario["correct_pivot"], step1_response, exclusion_text)
    r_stale  = _score_no_stale(action, scenario.get("wrong_pivots", []), step1_response)

    total = round(0.1 * r_fmt + 0.3 * r_interp + 0.4 * r_pivot + 0.2 * r_stale, 4)
    components = {
        "R_format": round(r_fmt, 4),
        "R_interpretation": round(r_interp, 4),
        "R_pivot": round(r_pivot, 4),
        "R_no_stale": round(r_stale, 4),
    }
    return total, components

# ---------------------------------------------------------------------------
# Core env logic
# ---------------------------------------------------------------------------

def _observe() -> dict:
    s = _state["scenario"]
    return {
        "instruction": s["initial_instruction"],
        "context_shift": s["context_shift"] if _state["shift_triggered"] else None,
        "step": _state["step_count"],
        "history": _state["history"],
        "done": _state["done"],
        "task": _state["task"],
    }


def reset(task: str = "medium") -> dict:
    global _state
    scenario = random.choice(SCENARIOS)
    _state = {
        "scenario": scenario,
        "task": task,
        "step_count": 0,
        "shift_triggered": False,
        "interpretation_score": None,
        "pivot_score": None,
        "completion_score": None,
        "done": False,
        "history": [],
        "prev_responses": [],
    }
    return _observe()


def step(action: str) -> dict:
    if _state["done"]:
        return {
            "observation": _observe(),
            "reward": 0.0,
            "final_score": None,
            "done": True,
            "info": {"error": "Episode finished. Call reset() to start a new episode."},
        }

    s = _state["scenario"]
    task = _state["task"]
    _state["step_count"] += 1
    n = _state["step_count"]

    # step-1 response is the pivot/no_stale reference for all subsequent steps
    step1_response = _state["prev_responses"][0] if _state["prev_responses"] else None
    _state["prev_responses"].append(action)

    reward, components = _compute_reward(action, s, step1_response)
    final_score = None
    phase = ""

    if n == 1:
        phase = "interpretation"
        _state["interpretation_score"] = reward
        if task == "easy":
            _state["done"] = True
            final_score = reward
        else:
            _state["shift_triggered"] = True

    elif n == 2:
        phase = "pivot"
        _state["pivot_score"] = reward
        if task == "medium":
            scores = [_state["interpretation_score"], _state["pivot_score"]]
            final_score = round(sum(scores) / len(scores), 4)
            _state["done"] = True

    elif n == 3:
        phase = "completion"
        # Hard mode: agent must score well on BOTH R_interpretation (remembered
        # original intent) and R_pivot (executed the shift) simultaneously.
        # The weighted formula already enforces this — no separate code needed.
        _state["completion_score"] = reward
        scores = [_state["interpretation_score"], _state["pivot_score"], _state["completion_score"]]
        final_score = round(sum(scores) / len(scores), 4)
        _state["done"] = True

    _state["history"].append({
        "step": n,
        "phase": phase,
        "action": action[:300],
        "reward": reward,
        "rewards": components,
    })

    info = {
        "phase": phase,
        "task": task,
        "rewards": components,
    }
    if _state["done"] and final_score is not None:
        info["final_score"] = final_score
        info["breakdown"] = {
            "interpretation": _state["interpretation_score"],
            "pivot": _state["pivot_score"],
            "completion": _state["completion_score"],
        }

    return {
        "observation": _observe(),
        "reward": reward,
        "final_score": final_score,
        "done": _state["done"],
        "info": info,
    }


def state() -> dict:
    s = _state["scenario"]
    return {
        "scenario_id": s["id"] if s else None,
        "domain": s["domain"] if s else None,
        "task": _state["task"],
        "step_count": _state["step_count"],
        "shift_triggered": _state["shift_triggered"],
        "scores": {
            "interpretation": _state["interpretation_score"],
            "pivot": _state["pivot_score"],
            "completion": _state["completion_score"],
        },
        "done": _state["done"],
        "history": _state["history"],
    }


# --- OpenEnv Environment subclass ---

class _DriftEnvAction(Action):
    response: str

class _DriftEnvObservation(Observation):
    instruction: str
    context_shift: Optional[str] = None
    step: int = 0
    history: list = []
    task: str = "medium"

class _DriftEnvState(State):
    scenario_id: Optional[int] = None
    domain: Optional[str] = None
    task: str = "medium"
    shift_triggered: bool = False
    scores: dict = {}
    history: list = []

class DriftEnvironment(Environment):
    def __init__(self):
        super().__init__()

    def _observe(self) -> _DriftEnvObservation:
        obs = _observe()
        return _DriftEnvObservation(
            instruction=obs["instruction"],
            context_shift=obs["context_shift"],
            step=obs["step"],
            history=obs["history"],
            done=obs["done"],
            task=obs["task"],
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> _DriftEnvObservation:
        task = kwargs.get("task", "medium")
        reset(task=task)
        return self._observe()

    def step(self, action: _DriftEnvAction, timeout_s=None, **kwargs) -> _DriftEnvObservation:
        result = step(action.response)
        obs = self._observe()
        final_score = result.get("final_score")
        obs.reward = final_score if final_score is not None else result.get("reward", 0.0)
        return obs

    @property
    def state(self) -> _DriftEnvState:
        s = state()
        return _DriftEnvState(
            scenario_id=s["scenario_id"],
            domain=s["domain"],
            task=s["task"],
            step_count=s["step_count"],
            shift_triggered=s["shift_triggered"],
            scores=s["scores"],
            history=s["history"],
        )


def create_health_app(app):
    @app.get("/")
    async def root():
        return {
            "name": "DriftEnv",
            "status": "running",
            "description": "RL environment for testing AI agent robustness under ambiguity and context shift",
            "tasks": ["easy", "medium", "hard"],
            "endpoints": {
                "reset": "POST /reset",
                "step": "POST /step",
                "state": "GET /state",
                "docs": "GET /docs"
            }
        }

    return app

def main():
    import uvicorn
    from openenv.core import create_app
    app = create_app(
        env=DriftEnvironment,
        action_cls=_DriftEnvAction,
        observation_cls=_DriftEnvObservation,
        env_name="driftenv",
    )
    create_health_app(app)  # register routes
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
