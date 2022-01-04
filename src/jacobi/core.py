import numpy as np


def _steps(p, h0, factor, maxiter):
    h = p * h0
    if h == 0:
        h = h0
    return h * factor ** np.arange(maxiter)


def _central(f, x, i, h):
    xp = x.copy()
    xm = x.copy()
    xp[i] += h
    xm[i] -= h
    return (f(x + h) - f(x - h)) * (0.5 / h)


def jacobi(
    f,
    x,
    *,
    rtol=0,
    maxiter=10,
    step=(0.5, 0.3090169943749474),
    mask=None,
    diagnostic=None
):
    assert maxiter > 0

    squeeze = np.ndim(x) == 0
    x = np.atleast_1d(x)
    x_shape = np.shape(x)

    x_indices = np.arange(len(x))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        x_indices = x_indices[mask]

    # TODO fill diagnostic

    jac = []
    err = []
    for k in x_indices:
        h = _steps(x[k], *step, maxiter)

        r = np.atleast_1d(_central(f, x, k, h[0]))
        r_shape = np.shape(r)
        re = np.full(r_shape, np.inf)
        todo = np.ones(r_shape, dtype=bool)
        fd = [r]

        squeeze &= r_shape == x_shape

        for i in range(1, len(h)):
            fdi = np.atleast_1d(_central(f, x, k, h[i]))
            fd.append(fdi[todo])

            # polynomial fit with one extra degree of freedom
            grad = min(i - 1, 3)
            start = i - (grad + 1)
            stop = i + 1
            q, c = np.polyfit(
                h[start:stop] ** 2, fd[start:], grad, rcond=None, cov=True
            )
            ri = q[-1]
            # pulls have roughly unit variance, however,
            # the pull distribution is not gaussian and looks
            # more like student's t
            rei = c[-1, -1] ** 0.5

            # update estimates that have significantly smaller error
            sub_todo = rei < (re[todo] * 2 * step[1] ** 2)
            todo1 = todo.copy()
            todo[todo1] = sub_todo
            r[todo] = ri[sub_todo]
            re[todo] = rei[sub_todo]

            # do not improve estimates further which meet the tolerance
            if rtol > 0:
                sub_todo &= rei > rtol * np.abs(ri)
                todo[todo1] = sub_todo

            if np.sum(todo) == 0:
                break

            # shrink previous vectors of estimates
            fd = [fdi[sub_todo] for fdi in fd]

        jac.append(np.squeeze(r) if squeeze else r)
        err.append(np.squeeze(re) if squeeze else re)

    return np.transpose(jac), np.transpose(err)
