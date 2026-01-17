import argparse
from typing import List

from cartpole_env import CartPoleEnv


def load_params(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as file:
        return [float(line.strip()) for line in file if line.strip()]


def evaluate(command: str, params: List[float], episodes: int = 100) -> int:
    success = 0
    for _ in range(episodes):
        env = CartPoleEnv(command)
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            score = sum(w * x for w, x in zip(params, obs))
            action = 1 if score > 0 else -1
            obs, done, reward = env.step(action)
            if obs is None:
                obs = [0.0, 0.0, 0.0, 0.0]
            total_reward += reward
        env.close()
        if total_reward >= 500:
            success += 1
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    params = load_params(args.params)
    success = evaluate(args.command, params, args.episodes)
    print(f"success={success}/{args.episodes}")


if __name__ == "__main__":
    main()
