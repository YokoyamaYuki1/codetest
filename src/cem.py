import random
from typing import Callable, List, Tuple


def _sample_normal(rng: random.Random) -> float:
    if hasattr(rng, "gauss"):
        return rng.gauss(0.0, 1.0)
    return sum(rng.uniform(-1.0, 1.0) for _ in range(3))


def sample_noise(dim: int, rng: random.Random) -> List[float]:
    return [_sample_normal(rng) for _ in range(dim)]


def evaluate_episode(env, policy_fn: Callable[[List[float]], int]) -> float:
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = policy_fn(obs)
        obs, done, reward = env.step(action)
        total_reward += reward
    return total_reward


def cem_optimize(
    env_factory: Callable[[], object],
    initial_theta: List[float],
    iterations: int = 30,
    population: int = 100,
    elite_frac: float = 0.1,
    seed: int | None = None,
) -> List[float]:
    rng = random.Random(seed)
    theta = list(initial_theta)
    dim = len(theta)
    elite_count = max(1, int(population * elite_frac))

    for _ in range(iterations):
        candidates: List[Tuple[float, List[float]]] = []
        for _ in range(population):
            noise = sample_noise(dim, rng)
            candidate = [t + n for t, n in zip(theta, noise)]

            env = env_factory()

            def policy(obs: List[float]) -> int:
                score = sum(w * x for w, x in zip(candidate, obs))
                return 1 if score > 0 else -1

            reward = evaluate_episode(env, policy)
            candidates.append((reward, candidate))

        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = [params for _, params in candidates[:elite_count]]
        theta = [sum(values) / elite_count for values in zip(*elites)]

    return theta
