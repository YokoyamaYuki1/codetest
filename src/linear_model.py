from typing import List


class LinearModel:
    def __init__(self, initial_param: List[float]) -> None:
        self.param = list(initial_param)

    def action(self, obs: List[float]) -> int:
        score = sum(w * x for w, x in zip(self.param, obs))
        return 1 if score > 0 else -1
