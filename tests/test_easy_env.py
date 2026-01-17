import sys

sys.path.append("src")

from easy_env import EasyEnv


def test_easy_env_steps():
    env = EasyEnv()
    obs = env.reset()
    assert len(obs) == 1
    done = False
    for _ in range(10):
        obs, done, _ = env.step(1)
    assert done
