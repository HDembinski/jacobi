import jacobi


def test_complex_step_derivative():
    from jacobi._cpp import complex_step_derivative

    assert complex_step_derivative(1, 2) == 3
