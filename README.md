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

**An RL environment for testing AI agent robustness under ambiguity and context shift.**

## What It Does

DriftEnv tests whether an AI agent can understand a task correctly when:
1. The instruction is deliberately vague
2. Requirements change mid-execution

## Episode Flow

EASY (1 step): Agent receives vague instruction, interprets it, gets scored.

MEDIUM (2 steps): Agent receives vague instruction, interprets it, gets scored. Context shift revealed, agent pivots, gets scored.

HARD (3 steps): All of the above plus agent produces final complete solution.

## Scoring

0.0 = Wrong interpretation or ignored context shift
0.5 = Partial match or correct clarifying question
1.0 = Correct interpretation or perfect pivot

Final score = average of all step scores, range 0.0 to 1.0

## Tasks

easy: 1 step, interpret vague instruction only, baseline score 0.40
medium: 2 steps, interpret plus handle context shift, baseline score 0.30
hard: 3 steps, interpret plus pivot plus complete correctly, baseline score 0.25

## Setup

pip install openenv-core pydantic requests openai

docker build -t driftenv .
docker run -p 8000:8000 driftenv

## Run Inference

export HF_TOKEN=your_token_here
python inference.py

## Validate

openenv validate

Built for the OpenEnv Hackathon. Meta x Hugging Face x PyTorch.

