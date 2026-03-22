import gymnasium as gym
import ale_py
import torch
import sys
import os
import argparse

parser = argparse.ArgumentParser(
    description="Run evaluation with a pretrained RL agent", prog="Evaluate SAC-EN-CIEL"
)
parser.add_argument("-a", "--agent", default="rainbowDQN_1M.pth")
parser.add_argument("-e", "--episode", default=1, type=int)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
sys.path.insert(0, os.path.dirname(__file__))

from agents.rainbow_atari import NoisyDuelingDistributionalNetwork, make_env

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charge l'environnement avec capture vidéo
env = gym.vector.SyncVectorEnv(
    [
        make_env(
            "BreakoutNoFrameskip-v4",
            seed=1,
            idx=0,
            capture_video=True,
            run_name=f"{args.agent}_{args.episode}ep",
        )
    ]
)

# Charge le modèle sauvegardé
model = NoisyDuelingDistributionalNetwork(env, n_atoms=51, v_min=-10, v_max=10).to(
    device
)

model.load_state_dict(
    torch.load(
        f"models/{args.agent}",
        map_location=device,
    )
)
model.eval()

obs, _ = env.reset()
episodes = 0
while episodes < args.episode:
    with torch.no_grad():
        q_dist = model(torch.Tensor(obs).to(device))
        q_values = torch.sum(q_dist * model.support, dim=2)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if terminations[0] or truncations[0]:
        episodes += 1
        print(f"Épisode {episodes} terminé")

env.close()
