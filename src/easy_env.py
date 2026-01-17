import random
from typing import List, Tuple


class EasyEnv:
    def __init__(self) -> None:
        self._last_obs: float | None = None
        self._step_count = 0

    def reset(self) -> List[float]:
        self._step_count = 0
        self._last_obs = random.uniform(-1.0, 1.0)
        return [self._last_obs]

    def obs_dim(self) -> int:
        return 1

    def step(self, action: int) -> Tuple[List[float], bool, float]:
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        prev_obs = self._last_obs
        reward = action * prev_obs
        self._step_count += 1
        self._last_obs = random.uniform(-1.0, 1.0)
        done = self._step_count >= 10
        return [self._last_obs], done, reward
