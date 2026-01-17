import sys

sys.path.append("src")

from cartpole_env import CartPoleEnv


def test_cartpole_env_runs():
    command = f"{sys.executable} -u tests/fake_cartpole_host.py"
    env = CartPoleEnv(command, max_steps=5)
    obs = env.reset()
    assert len(obs) == 4
    done = False
    for _ in range(5):
        obs, done, reward = env.step(1)
        assert reward == 1.0
        if done:
            break
    assert done
    env.close()
