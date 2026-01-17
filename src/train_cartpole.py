import argparse
from typing import List

from cartpole_env import CartPoleEnv
from cem import cem_optimize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True, help="cartpole host command")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--output", default="params.txt")
    args = parser.parse_args()

    def env_factory() -> CartPoleEnv:
        return CartPoleEnv(args.command)

    theta: List[float] = cem_optimize(
        env_factory=env_factory,
        initial_theta=[0.0, 0.0, 0.0, 0.0],
        iterations=args.iterations,
        population=args.population,
        elite_frac=args.elite_frac,
    )

    with open(args.output, "w", encoding="utf-8") as file:
        for value in theta:
            file.write(f"{value}\n")


if __name__ == "__main__":
    main()
