from jacobi import propagate
from pathlib import Path
import numpy as np
from numpy.testing import assert_equal

dir = Path(__file__).parent

rng = np.random.default_rng(1)


def test_nan():
    nan = np.nan
    a = np.array([4303.16536081, nan, 2586.42395464, nan, 2010.31141544, nan, nan, nan])

    a_var = np.array(
        [7.89977628e04, nan, 1.87676043e22, nan, 8.70294972e04, nan, nan, nan]
    )

    b = np.array([0.48358779, 0.0, 0.29371395, 0.0, 0.29838083, 0.58419942, 0.0, 0.0])

    b_var = np.array(
        [
            2.31907643e-05,
            0.00000000e00,
            2.17812131e-05,
            0.00000000e00,
            2.82526004e-05,
            1.66067899e-03,
            0.00000000e00,
            0.00000000e00,
        ]
    )

    mask = np.isnan(a) | np.isnan(b)
    mask_var = mask | np.isnan(a_var) | np.isnan(b_var)

    assert np.mean(mask_var) < 1

    def f(a, b):
        return a * b

    c, c_var = propagate(f, a, a_var, b, b_var, diagonal=True)

    assert_equal(np.isnan(c), mask)
    assert_equal(np.isnan(c_var), mask_var)
