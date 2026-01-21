import pytest

from my_proj import math


@pytest.mark.parametrize(
    "a,b",
    [
        (6, 7),
        (-5, 4),
        (100, 0),
        (1000000, 1000000),
    ],
)
def test_multiply(a, b):
    assert math.multiply(a, b) == a * b


@pytest.mark.parametrize(
    "x",
    [
        6,
        -5,
        100,
        1000000,
    ],
)
def test_square(x):
    assert math.square(x) == x**2
