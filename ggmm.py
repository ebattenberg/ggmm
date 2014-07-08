"""
Gaussian Mixture Models.

This implementation corresponds to frequentist (non-Bayesian) formulation
of Gaussian Mixture Models.
"""

# Author: Eric Battenberg <ebattenberg@gmail.com>
# Based on gmm.py from sklearn

import cudamat as cm
import logging
import numbers
import numpy as np
import sys

from scipy import linalg

EPS = np.finfo(float).eps

logger = logging.getLogger(__name__)

def init(max_ones=(1024*256)):
    cm.init(max_ones)

def shutdown():
    cm.shutdown()

def log_multivariate_normal_density(X, means, covars, covariance_type='diag',temp_gpu_mem=None):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (K, n_features)
        List of n_features-dimensional mean vectors for K Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of K covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (K, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (K, n_features)    if 'diag',
            (K, n_features, n_features) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, K)
        Array containing the log probabilities of each data point in
        X under each of the K multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        #'spherical': _log_multivariate_normal_density_spherical,
        #'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        #'full': _log_multivariate_normal_density_full
        }
    if temp_gpu_mem is None:
        N,D = X.shape
        K = means.shape[0]
        temp_gpu_mem = TempGPUMem()
        temp_gpu_mem.alloc(N,K,D)
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars,temp_gpu_mem)


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

def return_CUDAMatrix(input_array):
    '''
    If input is a numpy_array, convert to CUDAMatrix.
    If input is already CUDAMatrix, return input
    '''
    if isinstance(input_array,np.ndarray):
        if input_array.ndim == 1:
            return cm.CUDAMatrix(input_array[:,np.newaxis])
        else:
            return cm.CUDAMatrix(input_array)
    elif isinstance(input_array,cm.CUDAMatrix):
        return input_array
    else:
        raise ValueError, 'cannot handle input of type: %s' % (type(input_array),)

class TempGPUMem(dict):

    def alloc(self, N, K, D, key_shape_mapping=None):
        '''
        make sure temporary gpu memory is of correct size
        '''
        if key_shape_mapping is None:
            key_shape_mapping = {
                    'NxK' :         (N,K),
                    'NxK(2)' :      (N,K),
                    'KxD' :         (K,D),
                    'KxD(2)' :      (K,D),
                    'Kx1' :         (K,1),
                    'Nx1' :         (N,1),
                    'Nx1(2)' :      (N,1),
                    'NxD' :         (N,D),
            }
        # deallocate unneeded memory
        for key,shape in self.iteritems():
            if (not key_shape_mapping.has_key(key) 
                    or shape != key_shape_mapping[key]):
                self.pop(key)
                logger.debug('%s: removed %s at key %s' % (
                    sys._getframe().f_code.co_name,
                    shape,
                    key))

        # allocate memory
        for key,shape in key_shape_mapping.iteritems():
            if not self.has_key(key):
                logger.debug('%s: created %s at key %s' % (
                    sys._getframe().f_code.co_name,
                    shape,
                    key))
                self[key] = cm.empty(shape)


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
    K : int, optional
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
    `weights_` : array, shape (`K`,)
        This attribute stores the mixing weights for each mixture component.

    `means_` : array, shape (`K`, `n_features`)
        Mean parameters for each mixture component.

    `covars_` : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (K, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (K, n_features)             if 'diag',
            (K, n_features, n_features) if 'full'

    Output: returns bool converged





    """

    def __init__(self, K, D,
                covariance_type='diag',
                min_covar=1e-3,
                verbose=False):
               

        self.K = K
        self.D = D
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
        if weights.shape != (self.K,):
            raise ValueError, 'input weight vector is of shape %s, should be %s' % (
                    weights.shape,(self.K,))
        if np.abs(weights.sum()-1.0) > 1e-6:
            raise ValueError, 'input weight vector must sum to 1.0'
        if np.any(weights < 0.0):
            raise ValueError, 'input weight values must be non-negative'
        self.weights = return_CUDAMatrix(weights)

    def set_means(self,means):
        if means.shape != (self.K,self.D):
            raise ValueError, 'input mean matrix is of shape %s, should be %s' % (
                    means.shape,(self.K,self.D))
        self.means = return_CUDAMatrix(means)

    def set_covars(self,covars):
        if covars.shape != (self.K,self.D):
            raise ValueError, 'input covars matrix is of shape %s, should be %s' % (
                    covars.shape,(self.K,self.D))
        covars_ = covars.copy()
        if np.any(covars_ < 0):
            raise ValueError, 'input covars must be non-negative'
        if np.any(covars_ < self.min_covar):
            covars_[covars_ < self.min_covar] = self.min_covar
            if self.verbose:
                print 'input covars less than min_covar (%g) have been set to %g' % (self.min_covar,self.min_covar)

        self.covars = return_CUDAMatrix(covars_)

    def get_weights(self):
        return self.weights.asarray()

    def get_means(self):
        return self.means.asarray()

    def get_covars(self):
        return self.covars.asarray()

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (posteriors) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob_Nx1 : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        posteriors : array_like, shape (n_samples, K)
            Posterior probabilities of each mixture component for each
            observation
        """
        if None in (self.weights,self.means,self.covars):
            raise ValueError, 'GMM parameters have not been initialized'

        if X.shape[1] != self.D:
            raise ValueError, 'input data matrix X is of shape %s, should be %s' % (
                    X.shape,(X.shape[0],self.D))

        N = X.shape[0]
        X_gpu = return_CUDAMatrix(X)

        #maintain_temp_gpu_mem(self.temp_gpu_mem, N, self.K, self.D)

        # lpr = log_multivariate_normal_density() + np.log(self.weights)[None,:]
        # -----------------------------------------------------
        posteriors_NxK = log_multivariate_normal_density(
                X_gpu, self.means, self.covars, 
                self.covariance_type,self.temp_gpu_mem)
        # lpr += np.log(self.weights)
        #temp_Kx1 = self.temp_gpu_mem['Kx1']
        temp_Kx1 = self.temp_gpu_mem.get_mem((self.K,1), 'temp_Kx1')
        cm.log(self.weights, target=temp_Kx1)
        posteriors_NxK.add_row_vec(temp_Kx1.reshape((1,self.K)))
        temp_Kx1.reshape((self.K,1))
        # in use: lpr -> 'NxK'


        #logprob_Nx1 = np.log(np.sum(np.exp(lpr - vmax), axis=1))
        #logprob_Nx1 += vmax
        # ---------------------------------------------------------
        #vmax = self.temp_gpu_mem['Nx1']
        vmax_Nx1 = self.temp_gpu_mem.get_mem((N,1), 'vmax_Nx1')
        #logprob_Nx1 = self.temp_gpu_mem['Nx1(2)']
        logprob_Nx1 = self.temp_gpu_mem.get_mem((N,1), 'logprob_Nx1')
        # vmax_Nx1 = np.max(lpr,axis=1)
        posteriors_NxK.max(axis=1, target=vmax_Nx1)
        #temp_NxK = self.temp_gpu_mem['NxK(2)']
        temp_NxK = self.temp_gpu_mem.get_mem((N,self.K), 'temp_NxK')
        # temp_NxK = lpr - vmax_Nx1[:,None]
        posteriors_NxK.add_col_mult(vmax_Nx1, -1.0, target=temp_NxK)
        # temp_NxK = np.exp(temp_NxK)
        cm.exp(temp_NxK,target=temp_NxK)
        # logprob_Nx1 = np.sum(temp_NxK, axis=1)
        temp_NxK.sum(axis=1, target=logprob_Nx1)
        # logprob_Nx1 = np.log(logprob_Nx1)
        cm.log(logprob_Nx1, target=logprob_Nx1)
        # logprob_Nx1 += vmax_Nx1
        logprob_Nx1.add(vmax_Nx1)
        # in use: logprob_Nx1 -> 'Nx1(2)'


        # posteriors = np.exp(lpr - logprob_Nx1[:, np.newaxis])
        # ---------------------------------------------------------
        # lpr = lpr - logprob_Nx1[:,None]
        posteriors_NxK.add_col_mult(logprob_Nx1, mult=-1.0)
        cm.exp(posteriors_NxK, target=posteriors_NxK)

        return logprob_Nx1, posteriors_NxK

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob_Nx1 : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob_Nx1, _ = self.score_samples(X)
        return logprob_Nx1

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob_Nx1, posteriors = self.score_samples(X)
        return posteriors.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        posteriors : array-like, shape = (n_samples, K)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob_Nx1, posteriors = self.score_samples(X)
        return posteriors

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
        for comp in xrange(self.K):
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
        if X.shape[1] != self.D:
            raise ValueError, 'input data matrix X is of shape %s, should be %s' % (
                    X.shape,(X.shape[0],self.D))

        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]

        if X.shape[0] < self.K:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.K, X.shape[0]))



        # copy observations to GPU
        X_gpu = return_CUDAMatrix(X)

        max_log_prob = -np.infty

        for _ in xrange(n_init):
            if 'm' in init_params or self.means is None:
                perm = random_state.permutation(N)
                self.means = return_CUDAMatrix(X[perm[:self.K]])

            if 'w' in init_params or self.weights is None:
                self.weights = return_CUDAMatrix((1.0/self.K)*np.ones(self.K))

            if 'c' in init_params or self.covars is None:
                if self.covariance_type == 'diag':
                    cv = np.var(X,axis=0) + self.min_covar
                    self.covars = return_CUDAMatrix(np.tile(cv, (self.K,1)))
                else:
                    raise ValueError, 'unsupported covariance type: %s' % self.covariance_type

            # EM algorithms
            log_likelihood = []
            converged = False
            for i in xrange(n_iter):
                # Expectation step
                curr_log_likelihood, posteriors = self.score_samples(X_gpu)
                curr_log_likelihood_sum = curr_log_likelihood.sum(axis=0).asarray()[0,0]
                log_likelihood.append(curr_log_likelihood_sum)
                if verbose:
                    print 'Iter: %u, log-likelihood: %g' % (i,curr_log_likelihood_sum)

                # Check for convergence.
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                        thresh:
                    converged = True
                    break

                # Maximization step
                self._do_mstep(X_gpu, posteriors, update_params,
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

    def _do_mstep(self, X, posteriors, update_params, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weights.
        """

        weights = self.temp_gpu_mem.get_mem((1,self.K), 'temp_1xK')
        posteriors.sum(axis=0, target=weights) # 1xK
        weighted_X_sum = self.temp_gpu_mem.get_mem((self.K,self.D), 'temp_KxD')
        cm.dot(posteriors.T, X, target=weighted_X_sum)
        #inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        inverse_weights = self.temp_gpu_mem.get_mem((1,self.K), 'inverse_weights')
        inverse_weights.assign(1.0)
        inverse_weights.divide(weights)

        if 'w' in update_params:
            #self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
            weights.divide(weights.sum(axis=1), target=self.weights)
        if 'm' in update_params:
            #self.means = weighted_X_sum * inverse_weights
            weighted_X_sum.mult_by_col(inverse_weights.reshape((self.K,1)), target=self.means)
            inverse_weights.reshape((1,self.K))
        if 'c' in update_params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars = covar_mstep_func(
                self, X, posteriors, weighted_X_sum, inverse_weights,
                min_covar)
        return weights

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means.shape[1]
        if self.covariance_type == 'full':
            cov_params = self.K * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.K * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.K
        mean_params = ndim * self.K
        return int(cov_params + mean_params + self.K - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, D)

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
        X : array of shape(n_samples, D)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()


#########################################################################
## some helper routines
#########################################################################


def _log_multivariate_normal_density_diag(X, means, covars, temp_gpu_mem):
    """Compute Gaussian log-density at X for a diagonal model"""

    N, D = X.shape
    K = means.shape[0]

    #maintain_temp_gpu_mem(temp_gpu_mem,N,K,D)

    lpr_NxK = temp_gpu_mem.get_mem((N,K),'posteriors_NxK')
    temp_KxD = temp_gpu_mem.get_mem((K,D),'temp_KxD')
    inv_covars_KxD = temp_gpu_mem.get_mem((K,D),'inv_covars_KxD')
    temp_Kx1 = temp_gpu_mem.get_mem((K,1), 'temp_Kx1')
    temp_NxD = temp_gpu_mem.get_mem((N,D), 'temp_NxD')

    # compute inverse variances
    inv_covars_KxD.assign(1.0)
    inv_covars_KxD.divide(covars)

    # lpr = D * np.log(2*np.pi)
    lpr_NxK.assign(D * np.log(2*np.pi))

    # temp_Kx1 =  np.sum(np.log(covars), 1)
    cm.log(covars,target=temp_KxD)
    temp_KxD.sum(axis=1, target=temp_Kx1)

    # temp_Kx1 += np.sum((means**2)/covars, 1)
    means.mult(means, target=temp_KxD)
    temp_KxD.mult(inv_covars_KxD)
    temp_Kx1.add_sums(temp_KxD,axis=1)

    # lpr += temp_Kx1
    lpr_NxK.add_row_vec(temp_Kx1.reshape((1,K)))
    temp_Kx1.reshape((K,1)) # return to original shape

    # lpr += -2*np.dot(X, (means / covars).T)
    temp_KxD.assign(means)
    temp_KxD.mult(inv_covars_KxD)
    lpr_NxK.add_dot(X,temp_KxD.T, mult=-2.)

    # lpr += np.dot(X**2, (1.0 / covars).T)
    temp_NxD.assign(X)
    temp_NxD.mult(temp_NxD)
    lpr_NxK.add_dot(temp_NxD, inv_covars_KxD.T)

    # lpr *= -0.5
    lpr_NxK.mult(-0.5)

    # lpr_NxK still in use

    '''
    lpr = -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    '''

    return lpr_NxK

def _validate_covars(covars, covariance_type, K):
    """Do basic checks on matrix covariance sizes and values
    """
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != K:
            raise ValueError("'spherical' covars have length K")
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
                             "(K, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(K, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(K, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


def _covar_mstep_diag(gmm, X, posteriors, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for diagonal cases"""
    avg_X2 = np.dot(posteriors.T, X * X) * norm
    avg_means2 = gmm.means_ ** 2
    avg_X_means = gmm.means_ * weighted_X_sum * norm
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _covar_mstep_spherical(*args):
    """Performing the covariance M step for spherical cases"""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))


def _covar_mstep_full(gmm, X, posteriors, weighted_X_sum, norm,
                      min_covar):
    """Performing the covariance M step for full cases"""
    # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
    # Distribution"
    n_features = X.shape[1]
    cv = np.empty((gmm.K, n_features, n_features))
    for c in xrange(gmm.K):
        post = posteriors[:, c]
        # Underflow Errors in doing post * X.T are  not important
        np.seterr(under='ignore')
        avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
        mu = gmm.means_[c][np.newaxis]
        cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
    return cv


def _covar_mstep_tied(gmm, X, posteriors, weighted_X_sum, norm,
                      min_covar):
    # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
    n_features = X.shape[1]
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(gmm.means_.T, weighted_X_sum)
    return (avg_X2 - avg_means2 + min_covar * np.eye(n_features)) / X.shape[0]


_covar_mstep_funcs = {
    #'spherical': _covar_mstep_spherical,
    'diag': _covar_mstep_diag,
    #'tied': _covar_mstep_tied,
    #'full': _covar_mstep_full,
}
