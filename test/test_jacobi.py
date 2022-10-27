from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pytest
from jacobi import jacobi


def test_squeeze():
    # also tests promotion of integer argument to float
    y, ye = jacobi(np.exp, 0)
    assert_allclose(y, 1)
    assert_allclose(ye, 0, atol=1e-10)


def f1(x, a):
    return x**a + 3


def fd1(x, a):
    return a * x ** (a - 1)


def f5(x):
    return np.mean(x**2)


def fd5(x):
    return 2 * x / len(x)


def f6(x):
    return np.outer(x, x)


def fd6(r):
    assert len(r) == 3
    x, y, z = r
    r = np.empty((3, 3, 3))
    r[..., 0] = [
        [2 * x, y, z],
        [y, 0, 0],
        [z, 0, 0],
    ]
    r[..., 1] = [
        [0, x, 0],
        [x, 2 * y, z],
        [0, z, 0],
    ]
    r[..., 2] = [
        [0, 0, x],
        [0, 0, y],
        [x, y, 2 * z],
    ]
    return r


f7_a = np.array([[1, 2, 3], [4, 5, 6]])


def f7(x):
    return f7_a * x**2


@pytest.mark.parametrize(
    "fn",
    [
        (lambda x: 1.0, lambda x: 0.0),
        (lambda x: np.exp(x), lambda x: np.diagflat(np.exp(x))),
        (lambda x: x**2, lambda x: np.diagflat(2 * x)),
        (lambda x: np.ones_like(x), lambda x: np.diagflat(np.zeros_like(x))),
        (lambda x: x**-1, lambda x: np.diagflat(-(x**-2))),
        (lambda x: (x + 1) ** 0.5, lambda x: np.diagflat(0.5 * (x + 1) ** -0.5)),
        (f5, fd5),
        (f6, fd6),
    ],
)
def test_1d(fn):
    x = np.array([1, 2, 3], dtype=float)
    f, fd = fn
    y, ye = jacobi(f, x)
    assert_allclose(y, fd(x))
    assert_allclose(ye, np.zeros_like(y), atol=1e-10)


def test_0d():
    x = 2
    y, ye = jacobi(f7, x)
    assert np.ndim(y) == 2
    assert_allclose(y, f7_a * 2 * x)
    assert_allclose(ye, np.zeros_like(y), atol=1e-10)


def test_2d():
    x = [[1, 2, 3], [3, 4, 5]]
    fd, _ = jacobi(f7, x)
    assert np.ndim(fd) == 4
    fd_ref = np.zeros((2, 3, 2, 3))
    for i in range(2):
        for j in range(3):
            fd_ref[i, j, i, j] = f7_a[i, j] * 2 * x[i][j]
    assert_allclose(fd, fd_ref)


def test_abs_at_zero():
    fp, fpe = jacobi(np.abs, 0)
    assert_equal(fp, 0)
    assert_equal(fpe, 0)


def test_method_auto():
    d = {}
    fp, fpe = jacobi(lambda x: x, 0, diagnostic=d)
    assert_equal(d["method"], [0])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x >= 0 else np.nan, 0, diagnostic=d)
    assert_equal(d["method"], [1])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x <= 0 else np.nan, 0, diagnostic=d)
    assert_equal(d["method"], [-1])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)


def test_mask():
    x = np.array([1, 2, 3, 4])
    mask = np.array([True, False, False, True])
    jac = jacobi(f1, x, 3, mask=mask)[0]

    assert jac.shape == (4, 4)
    jac_ref = np.diag(fd1(x, 3))
    for i, mi in enumerate(mask):
        if not mi:
            jac_ref[i] = 0

    assert_allclose(jac, jac_ref)


@pytest.mark.parametrize("method", (-1, 0, 1))
def test_method(method):
    d = {}
    fp, fpe = jacobi(lambda x: x, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x >= 0 else np.nan, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    if method == 1:
        assert_allclose(fp, 1)
        assert_allclose(fpe, 0, atol=1e-10)
    else:
        assert_equal(fp, np.nan)
        assert_equal(fpe, np.inf)

    fp, fpe = jacobi(lambda x: x if x <= 0 else np.nan, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    if method == -1:
        assert_allclose(fp, 1)
        assert_allclose(fpe, 0, atol=1e-10)
    else:
        assert_equal(fp, np.nan)
        assert_equal(fpe, np.inf)


@pytest.mark.parametrize("x", [2, [1, 2]])
def test_diagonal(x):
    jac = jacobi(f1, x, 3, diagonal=True)[0]

    assert jac.ndim == np.ndim(x)
    assert_allclose(jac, fd1(np.atleast_1d(x), 3))


@pytest.mark.parametrize("maxgrad", (0, 2, 5))
def test_maxgrad(maxgrad):
    x = [1, 2, 3]

    jac = jacobi(f1, x, 3, maxgrad=maxgrad)[0]

    jac_ref = np.diag(fd1(np.atleast_1d(x), 3))
    assert_allclose(jac, jac_ref, atol=1e-5)


def test_rtol():
    x = [1, 2, 3]

    jac, jace = jacobi(f1, x, 3, rtol=0.1)
    jac_ref, jace_ref = jacobi(f1, x, 3)

    assert np.all(jace >= jace_ref)
    assert_allclose(jac, jac_ref, rtol=0.03)


@pytest.mark.parametrize("maxiter", (0, -1))
def test_bad_maxiter(maxiter):
    with pytest.raises(ValueError):
        jacobi(f1, 1, 3, maxiter=maxiter)


@pytest.mark.parametrize("maxgrad", (-1, -0.1))
def test_bad_maxgrad(maxgrad):
    with pytest.raises(ValueError):
        jacobi(f1, 1, 3, maxgrad=maxgrad)


@pytest.mark.parametrize("step", [0, -1, 1, 0.5])
def test_bad_step_0(step):
    with pytest.raises(ValueError, match=r"step\[0\]"):
        jacobi(f1, 1, 3, step=(step, 0.5))


@pytest.mark.parametrize("step", [0, -1, 1])
def test_bad_step_1(step):
    with pytest.raises(ValueError, match=r"step\[1\]"):
        jacobi(f1, 1, 3, step=(0.25, step))


@pytest.mark.parametrize("method", (-2, -3, 2, 3))
def test_bad_method(method):
    with pytest.raises(ValueError, match="method"):
        jacobi(f1, 1, 3, method=method)


def test_on_nan():
    x = np.array([2.0, np.nan, 3.0])

    d = {}
    yd, yde = jacobi(f1, x, 2, diagnostic=d)

    assert d["iteration"][1] == 1
    assert_allclose(
        yd, [[4.0, np.nan, 0.0], [np.nan, np.nan, np.nan], [0.0, np.nan, 6.0]]
    )
    assert_allclose(
        yde,
        [[0.0, np.inf, 0.0], [np.inf, np.inf, np.inf], [0.0, np.inf, 0.0]],
        atol=1e-15,
    )
