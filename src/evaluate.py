import gymnasium as gym
import ale_py
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
sys.path.insert(0, os.path.dirname(__file__))

from agents.rainbow_atari import NoisyDuelingDistributionalNetwork, make_env

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charge l'environnement avec capture vidéo
env = gym.vector.SyncVectorEnv(
    [
        make_env(
            "BreakoutNoFrameskip-v4", seed=1, idx=0, capture_video=True, run_name="eval"
        )
    ]
)

# Charge le modèle sauvegardé
model = NoisyDuelingDistributionalNetwork(env, n_atoms=51, v_min=-10, v_max=10).to(
    device
)

model.load_state_dict(
    torch.load(
        "runs/BreakoutNoFrameskip-v4__rainbow_atari__1__1774130797/rainbow_atari.pth",
        map_location=device,
    )
)
model.eval()

obs, _ = env.reset()
episodes = 0
while episodes < 3:
    with torch.no_grad():
        q_dist = model(torch.Tensor(obs).to(device))
        q_values = torch.sum(q_dist * model.support, dim=2)
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if terminations[0] or truncations[0]:
        episodes += 1
        print(f"Épisode {episodes} terminé")

env.close()
