#!/bin/bash
# Installation automatique de uv si manquant
if ! command -v uv &> /dev/null; then
    echo "Installation de uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Synchronisation de l'environnement
uv sync

# Exécution
echo "Lancement de l'entraînement..."
uv run python src/agents/sac_atari.py \
  --env-id BreakoutNoFrameskip-v4 \
  # Flag --cuda à retirer si CUDA non présent
  --cuda \ 
  --save-model \
  --total-timesteps 1000000