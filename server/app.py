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
}

def _score(agent_response: str, correct_answer: str, wrong_answers: list) -> tuple:
    agent_lower = agent_response.lower()
    correct_lower = correct_answer.lower()
    keywords = [w for w in correct_lower.split() if len(w) > 4]
    hits = sum(1 for kw in keywords if kw in agent_lower)
    ratio = hits / max(len(keywords), 1)
    wrong_match = any(w.lower()[:30] in agent_lower for w in wrong_answers)
    is_clarifying = "?" in agent_response and len(agent_response) < 400
    if wrong_match:
        return 0.0, "Response matched a known incorrect approach."
    if ratio >= 0.4:
        return 1.0, f"Strong match ({hits}/{len(keywords)} keywords)."
    if ratio >= 0.15 or is_clarifying:
        return 0.5, "Partial match or clarifying question."
    return 0.0, "Response did not match correct approach."

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
    reward = 0.0
    feedback = ""
    phase = ""
    final_score = None

    if n == 1:
        phase = "interpretation"
        reward, feedback = _score(action, s["hidden_interpretation"], s.get("wrong_pivots", []))
        _state["interpretation_score"] = reward
        if task == "easy":
            _state["done"] = True
            final_score = round(reward, 4)
        else:
            _state["shift_triggered"] = True

    elif n == 2:
        phase = "pivot"
        reward, feedback = _score(action, s["correct_pivot"], s.get("wrong_pivots", []))
        _state["pivot_score"] = reward
        if task == "medium":
            scores = [_state["interpretation_score"], _state["pivot_score"]]
            final_score = round(sum(scores) / len(scores), 4)
            _state["done"] = True

    elif n == 3:
        phase = "completion"
        combined = s["correct_pivot"] + " " + s["hidden_interpretation"]
        reward, feedback = _score(action, combined, s.get("wrong_pivots", []))
        _state["completion_score"] = reward
        scores = [_state["interpretation_score"], _state["pivot_score"], _state["completion_score"]]
        final_score = round(sum(scores) / len(scores), 4)
        _state["done"] = True

    _state["history"].append({"step": n, "phase": phase, "action": action[:300], "reward": reward})
    info = {"phase": phase, "feedback": feedback, "task": task}
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
