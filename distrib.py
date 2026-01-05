import math
import numpy as np

import numpy as np


def distrib(ncluster, cntrlpt, arclen, ds, ndim=7200):
    """
    Python version of distrib that RETURNS out.

    Parameters
    ----------
    ncluster : int
        Number of clustering control points
    cntrlpt : array-like (1-based)
        Indices where clustering is specified
    arclen : array-like (1-based)
        Arc-length values at control points
    ds : array-like (1-based)
        Desired spacings at control points
    ndim : int
        Maximum number of points (Fortran parameter)

    Returns
    -------
    out : numpy.ndarray (1-based indexing)
        Distributed arc-length coordinates
    """

    # allocate output (1-based)
    out = np.zeros(ndim + 1)

    # temporary workspace
    arcdist = np.zeros(ndim + 1)

    for n in range(1, ncluster):
        node0 = cntrlpt[n]
        node1 = cntrlpt[n + 1]

        alength = arclen[n + 1] - arclen[n]
        ds0 = ds[n] / alength
        ds1 = ds[n + 1] / alength

        clust(arcdist, ds0, ds1, node0, node1)

        for i in range(node0, node1 + 1):
            out[i] = arclen[n] + alength * arcdist[i]

    return out


def clust(y, ds0, ds1, jmin, jmax):
    """
    One-step corrector for clustering.
    """

    del_ = 1.0 / (jmax - jmin)

    s0 = del_ / ds0
    s1 = del_ / ds1

    clust2(y, s0, s1, jmin, jmax)

    alpha0 = (y[jmin + 1] - y[jmin]) / ds0
    alpha1 = (y[jmax] - y[jmax - 1]) / ds1

    s0 = alpha0 * del_ / ds0
    s1 = alpha1 * del_ / ds1

    clust2(y, s0, s1, jmin, jmax)


def clust2(y, s0, s1, jmin, jmax):
    """
    Vinokur endpoint clustering function.
    Produces y(j) in [0,1].
    """

    eps = 1.0e-3

    jm1 = jmax - 1
    jp1 = jmin + 1
    del_ = 1.0 / (jmax - jmin)

    b = math.sqrt(s0 * s1)
    a = b / s1

    y[jmin] = 0.0
    y[jmax] = 1.0

    if b > 1.0 + eps:
        dz = asinhf(b)
        coshdz = math.cosh(dz)
        omacdz = 1.0 - a * coshdz
        asindz = a * math.sqrt(coshdz * coshdz - 1.0)

        for j in range(jp1, jm1 + 1):
            u = math.tanh(dz * (j - jmin) * del_)
            y[j] = u / (asindz + omacdz * u)

    else:
        if b > 1.0 - eps:
            twobmo = 2.0 * (b - 1.0)
            onema = 1.0 - a

            for j in range(jp1, jm1 + 1):
                x = (j - jmin) * del_
                u = x * (1.0 + twobmo * (x - 0.5) * (1.0 - x))
                y[j] = u / (a + onema * u)

        else:
            dz = asinf(b)
            cosdz = math.cos(dz)
            cscdz = 1.0 / math.sqrt(1.0 - cosdz * cosdz)

            w0 = cscdz * (cosdz - 1.0 / a)
            z0 = math.atan(w0)
            dw = cscdz * (a - cosdz) - w0
            odw = 1.0 / dw

            for j in range(jp1, jm1 + 1):
                y[j] = odw * (math.tan(dz * (j - jmin) * del_ + z0) - w0)


def asinhf(u):
    """
    Polynomial approximation of inverse hyperbolic sine
    (matches Fortran implementation).
    """

    a1 = -0.15
    a2 = 0.0573214285714
    a3 = -0.024907294878
    a4 = 0.00774244601899
    a5 = -0.0010794122691

    b0 = -0.0204176930892
    b1 = 0.2490272170591
    b2 = 1.9496443322775
    b3 = -2.629454725241
    b4 = 8.5679591096315

    u1 = 2.7829681178603
    u2 = 35.0539798452776

    if u <= u1:
        ub = u - 1.0
        return math.sqrt(6.0 * ub) * (
            (((((a5 * ub + a4) * ub + a3) * ub + a2) * ub + a1) * ub + 1.0)
        )
    else:
        v = math.log(u)
        w = 1.0 / u - 1.0 / u2
        return (
            v
            + math.log(2.0 * v) * (1.0 + 1.0 / v)
            + (((b4 * w + b3) * w + b2) * w + b1) * w
            + b0
        )


def asinf(u):
    """
    Polynomial approximation of inverse sine
    (matches Fortran implementation).
    """

    a1 = 0.15
    a2 = 0.0573214285714
    a3 = 0.0489742834696
    a4 = -0.053337753213
    a5 = 0.0758451335824

    b3 = -2.6449340668482
    b4 = 6.7947319658321
    b5 = -13.2055008110734
    b6 = 11.7260952338351

    u1 = 0.2693897165164
    pi = math.pi

    if u <= u1:
        return pi * ((((((b6 * u + b5) * u + b4) * u + b3) * u + 1.0) * u - 1.0) * u + 1.0)
    else:
        ub = 1.0 - u
        return math.sqrt(6.0 * ub) * (
            (((((a5 * ub + a4) * ub + a3) * ub + a2) * ub + a1) * ub + 1.0)
        )

