# Sac-en-ciel
A comparative RL project: Rainbow DQN vs. Discrete SAC on Atari Breakout.

# Implementation 
This project explores the performance gap between a custom Rainbow DQN (enhanced with $\lambda$-returns) and Discrete Soft Actor-Critic (SAC).
The project leverages [CleanRL](https://docs.cleanrl.dev/)'s single-file implementation approach. It allows for direct modifications of the source code, specifically for injecting $\lambda$-return logic into the Rainbow agent.

# Installation
Requirements : [uv](https://docs.astral.sh/uv/getting-started/installation/)
```
```bash
git clone https://github.com/2521o/Sac-en-ciel
cd Sac-en-ciel
uv sync  
```

