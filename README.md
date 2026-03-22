# Sac-en-ciel
A comparative RL project: Rainbow DQN vs. Discrete SAC on Atari Breakout.

## Description
This project explores the performance gap between a custom Rainbow DQN 
(enhanced with λ-returns) and Discrete Soft Actor-Critic (SAC).
It leverages [CleanRL](https://docs.cleanrl.dev/)'s single-file implementation 
approach for direct source modification.

## Requirements
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- NVIDIA GPU with CUDA (recommended)

## Installation
```bash
git clone https://github.com/2521o/Sac-en-ciel
cd Sac-en-ciel
uv sync
```

## Usage

### Rainbow DQN
```bash
uv run python src/agents/rainbow_atari.py \
  --env-id BreakoutNoFrameskip-v4 \
  --cuda \
  --save-model \
  --buffer-size 100000 \
  --total-timesteps 1000000
```

### Discrete SAC
```bash
uv run python src/agents/sac_atari.py \
  --env-id BreakoutNoFrameskip-v4 \
  --cuda \
  --save-model \
  --total-timesteps 1000000
```

## Monitoring
```bash
source .venv/bin/activate
tensorboard --logdir runs/
```
Then open http://localhost:6006

## Pretrained models

You can use pretrained models by running:
```bash
uv run python src/evaluate.py --agent rainbowDQN_1M.pth --episode 5
```

This will automatically record a video of the agent playing Breakout, stored in `videos/`.

### Available agents

| Agent | Flag | Training steps |
|---|---|---|
| Rainbow DQN | `--agent rainbowDQN_1M.pth` (default)| 1M |