import jacobi
from numpy.testing import assert_allclose


def test_complex_step_derivative():
    from jacobi.complex_step import jacobi

    y = jacobi(lambda x: x ** 2, [1, 2, 3])

    assert_allclose(y, [2, 4, 6])

    y = jacobi(lambda x: np.ones(2) * x ** 2, [1, 2, 3])

    assert_allclose(y, [[2, 4, 6], [2, 4, 6]])
