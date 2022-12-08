from jacobi import propagate
from pathlib import Path
import numpy as np
from numpy.testing import assert_equal

dir = Path(__file__).parent

rng = np.random.default_rng(1)


def test_nan():
    nan = np.nan
    a = np.array(
        [
            nan,
            nan,
            2.92376310e03,
            1.94782244e02,
            nan,
            nan,
            nan,
            7.88660887e-08,
            nan,
            nan,
            4.30316536e03,
            4.80521360e03,
            6.16510069e02,
            nan,
            2.58642395e03,
            4.88147236e02,
            nan,
            nan,
            nan,
            nan,
            3.89431601e02,
            2.01031142e03,
            8.55352504e02,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            2.33492317e02,
            nan,
        ]
    )

    a_var = np.array(
        [
            nan,
            nan,
            1.99295074e25,
            2.48354599e02,
            nan,
            nan,
            nan,
            9.58082767e-04,
            nan,
            nan,
            7.89977628e04,
            1.39616829e04,
            3.57891928e03,
            nan,
            1.87676043e22,
            2.47813083e03,
            nan,
            nan,
            nan,
            nan,
            2.80102114e20,
            8.70294972e04,
            5.89370565e21,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            1.28489972e13,
            nan,
        ]
    )

    b = np.array(
        [
            0.10805038,
            0.10170801,
            0.28121607,
            0.5948512,
            0.23283663,
            0.24067257,
            0.17606588,
            0.21121785,
            0.12724757,
            0.10896939,
            0.48358779,
            0.53638651,
            0.68426319,
            0.0,
            0.29371395,
            0.67426487,
            0.0,
            0.0,
            0.0,
            0.0,
            0.03664308,
            0.29838083,
            0.48790491,
            0.58419942,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.24093129,
            0.3789397,
        ]
    )

    b_var = np.array(
        [
            1.04671868e-03,
            4.64290157e-05,
            1.82166247e-05,
            6.81167241e-04,
            2.45354167e-03,
            4.78091395e-04,
            1.67943003e-04,
            4.10409049e-05,
            1.24692735e-04,
            5.26068355e-05,
            2.31907643e-05,
            2.63940402e-05,
            2.28672527e-04,
            0.00000000e00,
            2.17812131e-05,
            2.92027443e-04,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            4.17388538e-06,
            2.82526004e-05,
            1.25796566e-04,
            1.66067899e-03,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.86618707e-04,
            7.84779137e-04,
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
