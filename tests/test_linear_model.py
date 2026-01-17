import sys

sys.path.append("src")

from linear_model import LinearModel


def test_linear_model_action():
    model = LinearModel([1.0, -2.0])
    assert model.action([1.0, 1.0]) == -1
    assert model.action([3.0, 1.0]) == 1
