import sys

sys.path.append("src")

from cem import cem_optimize
from easy_env import EasyEnv


def test_cem_easy_env_runs():
    def env_factory() -> EasyEnv:
        return EasyEnv()

    result = cem_optimize(env_factory, [0.0], iterations=2, population=10, elite_frac=0.2, seed=0)
    assert len(result) == 1
