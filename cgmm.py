"""
Gaussian Mixture Models.

This implementation corresponds to frequentist (non-Bayesian) formulation
of Gaussian Mixture Models.
"""

# Author: Eric Battenberg <ebattenberg@gmail.com>
# Based on gmm.py from sklearn

import numbers
import numpy as np
from scipy import linalg

EPS = np.finfo(float).eps


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def pinvh(a, cond=None, rcond=None, lower=True):
    """Compute the (Moore-Penrose) pseudo-inverse of a hermetian matrix.

    Calculate a generalized inverse of a symmetric matrix using its
    eigenvalue decomposition and including all 'large' eigenvalues.

    Parameters
    ----------
    a : array, shape (N, N)
        Real symmetric or complex hermetian matrix to be pseudo-inverted
    cond, rcond : float or None
        Cutoff for 'small' eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are considered
        zero.

        If None or -1, suitable machine precision is used.
    lower : boolean
        Whether the pertinent array data is taken from the lower or upper
        triangle of a. (Default: lower)

    Returns
    -------
    B : array, shape (N, N)

    Raises
    ------
    LinAlgError
        If eigenvalue does not converge

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(9, 6)
    >>> a = np.dot(a, a.T)
    >>> B = pinvh(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

    """
    a = np.asarray_chkfinite(a)
    s, u = linalg.eigh(a, lower=lower)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps

    # unlike svd case, eigh can lead to negative eigenvalues
    above_cutoff = (abs(s) > cond * np.max(abs(s)))
    psigma_diag = np.zeros_like(s)
    psigma_diag[above_cutoff] = 1.0 / s[above_cutoff]

    return np.dot(u * psigma_diag, np.conjugate(u).T)


def sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                    random_state=None):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean of the distribution.

    covars : array_like, optional
        Covariance of the distribution. The shape depends on `covariance_type`:
            scalar if 'spherical',
            (n_features) if 'diag',
            (n_features, n_features)  if 'tied', or 'full'

    covariance_type : string, optional
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    n_samples : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    X : array, shape (n_features, n_samples)
        Randomly generated sample
    """
    rng = check_random_state(random_state)
    n_dim = len(mean)
    rand = rng.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    if covariance_type == 'spherical':
        rand *= np.sqrt(covar)
    elif covariance_type == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        s, U = linalg.eigh(covar)
        s.clip(0, out=s)        # get rid of tiny negatives
        np.sqrt(s, out=s)
        U *= s
        rand = np.dot(U, rand)

    return (rand.T + mean).T


class GMM(object):
    """Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.


    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    thresh : float, optional
        Convergence threshold.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    update_params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    Attributes
    ----------
    `weights` : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    `means` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    `covars_` : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    Output: returns bool converged





    """

    def __init__(self, n_components, n_dimensions,
                covariance_type='diag',
                min_covar=1e-3,
                verbose=False):
               

        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.verbose = verbose

        if not covariance_type in ['diag']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

        self.weights = None
        self.means = None
        self.covars = None

    def set_weights(self,weights):
        if weights.shape != (self.n_components,):
            raise ValueError, 'input weight vector is of shape %s, should be %s' % (
                    weights.shape,(self.n_components,))
        if np.abs(weights.sum()-1.0) > 1e-6:
            raise ValueError, 'input weight vector must sum to 1.0'
        if np.any(weights < 0.0):
            raise ValueError, 'input weight values must be non-negative'
        self.weights = weights.copy()

    def set_means(self,means):
        if means.shape != (self.n_components,self.n_dimensions):
            raise ValueError, 'input mean matrix is of shape %s, should be %s' % (
                    means.shape,(self.n_components,self.n_dimensions))
        self.means = means.copy()

    def set_covars(self,covars):
        if covars.shape != (self.n_components,self.n_dimensions):
            raise ValueError, 'input covars matrix is of shape %s, should be %s' % (
                    covars.shape,(self.n_components,self.n_dimensions))
        self.covars = covars.copy()
        if np.any(self.covars < 0):
            raise ValueError, 'input covars must be non-negative'
        if np.any(self.covars < self.min_covar):
            self.covars[self.covars < self.min_covar] = self.min_covar
            if self.verbose:
                print 'input covars less than min_covar (%g) have been set to %g' % (self.min_covar,self.min_covar)

    def get_weights(self):
        return self.weights

    def get_means(self):
        return self.means

    def get_covars(self):
        return self.covars

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        if None in (self.weights,self.means,self.covars):
            raise ValueError, 'GMM parameters have not been initialized'

        if X.shape[1] != self.n_dimensions:
            raise ValueError, 'input data matrix X is of shape %s, should be %s' % (
                    X.shape,(X.shape[0],self.n_dimensions))

        X = np.asarray(X, dtype=np.float)
        n_observations = X.shape[0]

        lpr = (log_multivariate_normal_density(X, self.means, self.covars,
                                               self.covariance_type)
               + np.log(self.weights))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        weight_cdf = np.cumsum(self.weights)

        X = np.empty((n_samples, self.means.shape[1]))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in xrange(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                if self.covariance_type == 'tied':
                    cv = self.covars
                elif self.covariance_type == 'spherical':
                    cv = self.covars[comp][0]
                else:
                    cv = self.covars[comp]
                X[comp_in_X] = sample_gaussian(
                    self.means[comp], cv, self.covariance_type,
                    num_comp_in_X, random_state=random_state).T
        return X

    def fit(self, X,
            thresh=1e-2, n_iter=100, n_init=1,
            update_params='wmc', init_params='wmc',
            random_state=None,verbose=None):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        if verbose is None:
            verbose = self.verbose

        if random_state is None:
            random_state = np.random.RandomState()
        else:
            check_random_state(random_state)

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')
        if X.shape[1] != self.n_dimensions:
            raise ValueError, 'input data matrix X is of shape %s, should be %s' % (
                    X.shape,(X.shape[0],self.n_dimensions))

        X = np.asarray(X, dtype=np.float)
        n_observations = X.shape[0]

        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        for _ in xrange(n_init):
            if 'm' in init_params or self.means is None:
                perm = random_state.permutation(n_observations)
                self.means = X[perm[:self.n_components]].copy()

            if 'w' in init_params or self.weights is None:
                self.weights = (1.0/self.n_components)*np.ones(self.n_components)

            if 'c' in init_params or self.covars is None:
                if self.covariance_type == 'diag':
                    cv = np.var(X,axis=0) + self.min_covar
                    self.covars = np.tile(cv, (self.n_components,1))
                else:
                    raise ValueError, 'unsupported covariance type: %s' % self.covariance_type

            # EM algorithms
            log_likelihood = []
            converged = False
            for i in xrange(n_iter):
                # Expectation step
                curr_log_likelihood, responsibilities = self.score_samples(X)
                curr_log_likelihood_sum = curr_log_likelihood.sum()
                log_likelihood.append(curr_log_likelihood_sum)
                if verbose:
                    print 'Iter: %u, log-likelihood: %g' % (i,curr_log_likelihood_sum)

                # Check for convergence.
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                        thresh:
                    converged = True
                    break

                # Maximization step
                self._do_mstep(X, responsibilities, update_params,
                               self.min_covar)

            # if the results are better, keep it
            if n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights,
                                   'means': self.means,
                                   'covars': self.covars}
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # n_iter == 0 occurs when using GMM within HMM
        if n_iter:
            self.covars = best_params['covars']
            self.means = best_params['means']
            self.weights = best_params['weights']

        return converged

    def _do_mstep(self, X, responsibilities, update_params, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weihgts.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in update_params:
            self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in update_params:
            self.means = weighted_X_sum * inverse_weights
        if 'c' in update_params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()


#########################################################################
## some helper routines
#########################################################################


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model"""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model"""
    n_samples, n_dim = X.shape
    icv = pinvh(covars)
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(linalg.det(covars) + 0.1)
                  + np.sum(X * np.dot(X, icv), 1)[:, np.newaxis]
                  - 2 * np.dot(np.dot(X, icv), means.T)
                  + np.sum(means * np.dot(means, icv), 1))
    return lpr


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices.
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values
    """
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template
    """
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv


def _covar_mstep_diag(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for diagonal cases"""
    avg_X2 = np.dot(responsibilities.T, X * X) * norm
    avg_means2 = gmm.means ** 2
    avg_X_means = gmm.means * weighted_X_sum * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _covar_mstep_spherical(*args):
    """Performing the covariance M step for spherical cases"""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))


def _covar_mstep_full(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for full cases"""
    # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    n_features = X.shape[1]
    cv = np.empty((gmm.n_components, n_features, n_features))
    for c in xrange(gmm.n_components):
        post = responsibilities[:, c]
        # Underflow Errors in doing post * X.T are  not important
        np.seterr(under='ignore')
        avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
        mu = gmm.means[c][np.newaxis]
        cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
    return cv


def _covar_mstep_tied(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
    n_features = X.shape[1]
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(gmm.means.T, weighted_X_sum)
    return (avg_X2 - avg_means2 + min_covar * np.eye(n_features)) / X.shape[0]


_covar_mstep_funcs = {'spherical': _covar_mstep_spherical,
                      'diag': _covar_mstep_diag,
                      'tied': _covar_mstep_tied,
                      'full': _covar_mstep_full,
                      }
