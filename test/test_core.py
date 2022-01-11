from numpy.testing import assert_allclose
from jacobi import core


def test_steps():

    assert_allclose(core._steps(0, [0.2, 0.1], 4), [0.2, 0.02, 0.002, 0.0002])
    assert_allclose(core._steps(2, [0.2, 0.2], 3), [0.4, 0.08, 0.016])
    assert_allclose(core._steps(2, [0.2, 0.2], 3), [0.4, 0.08, 0.016])
    assert_allclose(core._steps(0, [0.2], 3), [0.2, 0.04, 0.008])
