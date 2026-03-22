import gymnasium as gym
import ale_py
import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
sys.path.insert(0, os.path.dirname(__file__))

from agents.rainbow_atari import NoisyDuelingDistributionalNetwork, make_env
from agents.sac_atari import Actor

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(
    description="Run evaluation with a pretrained RL agent", prog="Evaluate SAC-EN-CIEL"
)
parser.add_argument("-a", "--agent", required=True)
parser.add_argument("-e", "--episode", default=1, type=int)
parser.add_argument("--algo", choices=["rainbow", "sac"], required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent_name = args.agent
run_name = f"{agent_name}_{args.episode}ep"

if args.algo == "rainbow":
    env = gym.vector.SyncVectorEnv(
        [
            make_env(
                "BreakoutNoFrameskip-v4",
                seed=1,
                idx=0,
                capture_video=True,
                run_name=run_name,
            )
        ]
    )
    model = NoisyDuelingDistributionalNetwork(env, n_atoms=51, v_min=-10, v_max=10).to(
        device
    )
    model.load_state_dict(torch.load(f"models/{agent_name}.pth", map_location=device))
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

elif args.algo == "sac":
    env = gym.vector.SyncVectorEnv(
        [
            make_env(
                "BreakoutNoFrameskip-v4",
                seed=1,
                idx=0,
                capture_video=True,
                run_name=run_name,
            )
        ]
    )
    model = Actor(env).to(device)
    model.load_state_dict(torch.load(f"models/{agent_name}.pth", map_location=device))
    model.eval()

    obs, _ = env.reset()
    episodes = 0
    while episodes < args.episode:
        with torch.no_grad():
            actions, _, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if terminations[0] or truncations[0]:
            episodes += 1
            print(f"Épisode {episodes} terminé")

env.close()
