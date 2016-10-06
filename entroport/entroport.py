from __future__ import division, absolute_import
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

from entroport.utils import window
from entroport.utils import arrmap

MAX_OPT_ATTEMPTS = 10

class _Debug():
    def __init__(self):
        self.x = []
        self.y = []

_debug = _Debug()

def _kernel(theta, R):
    """ theta : [N,] ndarray, R : [T, N] ndarray
    """
    return np.exp(np.sum(theta * R, axis=1))

def _goalfun(theta, *args):
    R = args[0]
    comm = _kernel(theta, R)
    fun = np.sum(comm)
    grad = np.sum(comm * R.T, axis=1)

    return fun, grad

def _get_thetas(R, pcached_startval):
    Nassets = R.shape[1]
    success = False
    i = 0
    # Quirky but works. Look at Convex.jl. cvxopt too slow.
    while not success and i < MAX_OPT_ATTEMPTS:
        optres = minimize(fun=_goalfun,
                          x0=pcached_startval,
                          args=R,
                          jac=True,
                          method='BFGS')
        success = optres.success
        # if this run was unsuccessful reset the start values
        pcached_startval[:] = np.random.rand(Nassets)
        i += 1

    if not success:
        raise RuntimeError("fmin failed (SDF ML)", optres.message)

    theta = optres.x
    pcached_startval[:] = theta # Cleaner way to do this?

    return theta

# def scorer(estimator, X, y):
#     coef = estimator.coef_
#     coefsum = -np.abs(np.sum(coef))
#     if np.isclose(coefsum, 0.0):
#         return np.inf
#     coef /= coefsum
#     intercept = estimator.intercept_
#     return np.average( ((X * coef).sum(axis=1) + intercept - y)**2 )

def _get_weights(R, theta, regularization, lmin, lmax, lnum, nfolds):
    sdf_is = _kernel(theta, R)
    if not regularization:
        reg = sm.OLS(sdf_is, sm.add_constant(R)).fit()
        weights = reg.params[1:]
    else:
        est = RidgeCV(alphas=np.linspace(lmin, lmax, num=lnum),
        fit_intercept=True,
        normalize=False,
        cv=nfolds, scoring=None).fit(R, sdf_is)

        weights = est.coef_
        _debug.x.append(est.alpha_)

    weights /= -np.abs(np.sum(weights))

    return weights

class EntroPort(object):
    r""" Portfolio allocation with relative entropy minimization

    Estimates portfolio weights on a rolling out of sample basis by projecting
    past returns on an estimated stochastic discount factor.

    Parameters
    ----------
    df : DataFrame
        Portfolio time series, net log (excess) returns

    estlength : int
        Length of the moving estimation window

    step : int
        The number of observations in the out of sample
        estimation window (default is 1) - a step size of for ex. 10 would
        be the same as a 10 period rebalancing.

    Attributes
    ----------
    `theta_` : DataFrame
        Estimated thetas

    `weights_` : DataFrame
        Estimated weights

    `pfs_` : DataFrame
        The time series of the estimated stochastic discount factor (`sdf`) and
        information portfolio (`ip`)

    Notes
    -----
    In detail:
    for each estimation window of size **T**, estimate

    .. math::

        \boldsymbol{\hat \theta} = \arg\min_{\boldsymbol{\theta}}
         \sum_{t=1}^{T}\mathrm{kernel}(\boldsymbol{\theta}, \mathbf{R}_t)

    where

    .. math::

        \mathrm{kernel}(\boldsymbol{\theta}, \mathbf{R}_t) =
        e^{\boldsymbol{\theta'} \mathbf{R}_t}

    The estimated portfolio weights are the coefficients (normalized) obtained
    from regressing the returns in the estimation window on the **kernel**
    evaluated at :math:`\boldsymbol{\hat \theta}`

    The out of sample *information portfolio* is these weights multiplied by
    the out of sample returns.

    The out of sample *stochastic discount factor* is the estimated **kernel**
    evaluated at the out of sample returns.

    References
    ----------
    .. [1] Ghosh, Julliard, and Taylor A, "One Factor Benchmark Model
        for Asset Pricing", (2015 wp)

    """

    def __init__(self, df, estlength, step=1, regularization=False,
                    lmin=0.1, lmax = 5, lnum=10, nfolds=5):
        self.df = df
        self.estlength = estlength
        self.step = step
        self.regularization = regularization
        self.lmin = lmin
        self.lmax = lmax
        self.lnum = lnum
        self.nfolds = nfolds

        self.Nobs = df.shape[0]
        assert step < estlength < self.Nobs
        self.estidx = list(window(range(self.Nobs), estlength, step))
        self.oosidx = map(lambda x: range(x[-1] + 1, x[-1] + 1 + step),
                          self.estidx)

        # Taking care of the last estimation and out of sample indices
        self.oosidx[-1] = list(set(self.oosidx[-1]).intersection(range(self.Nobs)))
        if not self.oosidx[-1]:
            self.oosidx.pop()
            self.estidx.pop()

        self._pcached_startval = np.random.rand(df.shape[1])

    def _fit_one_period(self, idx):
        estwindow = self.estidx[idx]
        ooswindow = self.oosidx[idx]

        R = self.df.iloc[estwindow].values

        theta = _get_thetas(R, self._pcached_startval)
        weights = _get_weights(R, theta, self.regularization,
                                self.lmin, self.lmax, self.lnum, self.nfolds)

        # b/c not necessarily equal to `self.step` in the last period
        reps = len(ooswindow)

        return np.tile(theta, (reps, 1)), np.tile(weights, (reps, 1))

    def fit(self):
        """Fit this model.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        theta, weights = arrmap(self._fit_one_period, range(len(self.estidx)))

        index = self.df.index[self.oosidx[0][0]:self.Nobs]
        colnames = self.df.columns.tolist()

        self.thetas_ = pd.DataFrame(theta, index=index, columns=colnames)
        self.weights_ = pd.DataFrame(weights, index=index, columns=colnames)

        sdf = map(lambda theta, R: _kernel(theta, R[None, :])[0],
                  theta, self.df.loc[index].values)
        ip = (weights * self.df.loc[index]).sum(axis=1)

        self.pfs_ = pd.DataFrame({'sdf': sdf, 'ip': ip})

        return self
