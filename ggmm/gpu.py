'''
GPU/CUDAMat backend for GMM training and inference
'''

# Author: Eric Battenberg <ebattenberg@gmail.com>
# Based on gmm.py from sklearn

# python2 compatibility
from __future__ import print_function
from future.utils import iteritems
from builtins import range

import cudamat as cm
import logging
import numbers
import numpy as np
import sys

from scipy import linalg

EPS = np.finfo(float).eps

logger = logging.getLogger(__name__)


def init(max_ones=(1024*256)):
    '''
    Initialize GPU resources.

    Parameters
    -----------
    max_ones : int, optional
        Allocate enough memory for a sum of up to 'max_ones' length
    '''
    cm.init(max_ones)


def shutdown():
    '''Free GPU resources'''
    cm.shutdown()


def log_multivariate_normal_density(
        X, means, covars, covariance_type='diag', temp_gpu_mem=None):
    '''Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_dimensions)
        List of 'n_samples' data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_dimensions)
        List of 'n_components' mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_dimensions)      if 'spherical',
            (n_dimensions, n_dimensions)    if 'tied',
            (n_components, n_dimensions)    if 'diag',
            (n_components, n_dimensions, n_dimensions) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    '''
    log_multivariate_normal_density_dict = {
        # 'spherical': _log_multivariate_normal_density_spherical,
        # 'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        # 'full': _log_multivariate_normal_density_full
    }
    if temp_gpu_mem is None:
        N, D = X.shape
        K = means.shape[0]
        temp_gpu_mem = TempGPUMem()
        temp_gpu_mem.alloc(N, K, D)
    return log_multivariate_normal_density_dict[covariance_type](
        return_CUDAMatrix(X),
        return_CUDAMatrix(means),
        return_CUDAMatrix(covars),
        temp_gpu_mem)


def check_random_state(seed):
    '''Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def pinvh(a, cond=None, rcond=None, lower=True):
    '''Compute the (Moore-Penrose) pseudo-inverse of a hermetian matrix.

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

    '''
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
    if isinstance(input_array, np.ndarray):
        if input_array.ndim == 1:
            return cm.CUDAMatrix(input_array[:, np.newaxis])
        else:
            return cm.CUDAMatrix(input_array)
    elif isinstance(input_array, cm.CUDAMatrix):
        return input_array
    else:
        raise ValueError('cannot handle input of type: %s'
                         % (type(input_array),))


def return_NumpyArray(input_array):
    '''
    If input is a numpy_array, return input
    If input is a CUDAMatrix, return numpy array
    '''
    if isinstance(input_array, np.ndarray):
        return input_array
    elif isinstance(input_array, cm.CUDAMatrix):
        if input_array.shape[1] == 1:
            return input_array.asarray().flatten()
        else:
            return input_array.asarray()
        return input_array
    else:
        raise ValueError('cannot handle input of type: %s'
                         % (type(input_array),))


class TempGPUMem(dict):

    def alloc(self, *args):
        '''
        Allocate temporary GPU memory

        alloc(N, K, D)
        allo(key_shape_mapping)
        '''
        if len(args) == 3:
            N, K, D = args
            key_shape_mapping = {
                'posteriors_NxK':      (N, K),  # big
                'weighted_X_sum_KxD':  (K, D),  # medium
                'vmax_Nx1':            (N, 1),
                'logprob_Nx1':         (N, 1),
                'inv_weights_Kx1':     (K, 1),
                'temp_NxD':            (N, D),  # big
                'temp_KxD':            (K, D),  # medium
                'temp_KxD_2':          (K, D),  # medium
                'temp_Kx1':            (K, 1),
                'temp_Kx1_2':          (K, 1),
            }
        elif len(args) == 1:
            key_shape_mapping = args[0]
        else:
            ValueError(
                'TempGPUMem: alloc(N, K, D) or alloc(key_shape_mapping)')

        # allocate memory
        for key, shape in iteritems(key_shape_mapping):
            if key not in self:
                logger.debug('%s: created %s at key %s',
                             sys._getframe().f_code.co_name,
                             shape,
                             key)
                self[key] = cm.empty(shape)
            elif self[key].shape != shape:
                logger.debug('%s: reshaped %s from %s to %s',
                             sys._getframe().f_code.co_name,
                             key,
                             self[key].shape,
                             shape)
                self[key] = cm.empty(shape)

    def dealloc(self):
        self.clear()


class GMM(object):
    '''Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.


    Parameters
    ----------
    n_components : int, required
        Number of mixture components.

    n_dimensions : int, required
        Number of data dimensions.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  For now, only 'diag' is supported.
        Defaults to 'diag'.

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    verbose : bool, optional
        Whether to print EM iteration information during training
    '''

    def __init__(self, n_components, n_dimensions,
                 covariance_type='diag',
                 min_covar=1e-3,
                 verbose=False):

        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.verbose = verbose

        if covariance_type not in ['diag']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)

        self.weights = None
        self.means = None
        self.covars = None

    def set_weights(self, weights):
        '''
        Set weight vector with numpy array.

        Parameters
        ----------
        weights: numpy.ndarray, shape (n_components,)
        '''
        if weights.shape != (self.n_components,):
            raise ValueError(
                'input weight vector is of shape %s, should be %s'
                % (weights.shape, (self.n_components,)))
        if np.abs(weights.sum()-1.0) > 1e-6:
            raise ValueError('input weight vector must sum to 1.0')
        if np.any(weights < 0.0):
            raise ValueError('input weight values must be non-negative')
        self.weights = return_CUDAMatrix(weights)

    def set_means(self, means):
        '''
        Set mean vectors with numpy array.

        Parameters
        ----------
        means: numpy.ndarray, shape (n_components, n_dimensions)
        '''
        if means.shape != (self.n_components, self.n_dimensions):
            raise ValueError(
                'input mean matrix is of shape %s, should be %s'
                % (means.shape, (self.n_components, self.n_dimensions)))
        self.means = return_CUDAMatrix(means)

    def set_covars(self, covars):
        '''
        Set covariance matrices with numpy array

        Parameters
        ----------
        covars: numpy.ndarray, shape (n_components, n_dimensions)
            (for now only diagonal covariance matrices are supported)
        '''
        if covars.shape != (self.n_components, self.n_dimensions):
            raise ValueError(
                'input covars matrix is of shape %s, should be %s'
                % (covars.shape, (self.n_components, self.n_dimensions)))
        covars_ = return_NumpyArray(covars).copy()
        if np.any(covars_ < 0):
            raise ValueError('input covars must be non-negative')
        if np.any(covars_ < self.min_covar):
            covars_[covars_ < self.min_covar] = self.min_covar
            if self.verbose:
                print('input covars less than min_covar (%g) ' \
                    'have been set to %g' % (self.min_covar, self.min_covar))

        self.covars = return_CUDAMatrix(covars_)

    def get_weights(self):
        '''
        Return current weight vector as numpy array

        Returns
        -------
        weights : np.ndarray, shape (n_components,)
        '''
        return self.weights.asarray().flatten()

    def get_means(self):
        '''
        Return current means as numpy array

        Returns
        -------
        means : np.ndarray, shape (n_components, n_dimensions)
        '''
        return self.means.asarray()

    def get_covars(self):
        '''
        Return current means as numpy array

        Returns
        -------
        covars : np.ndarray, shape (n_components, n_dimensions)
            (for now only diagonal covariance matrices are supported)
        '''
        return self.covars.asarray()

    def score_samples(self, X, temp_gpu_mem=None):
        '''Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior probability of each
        mixture component for each element of X.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_dimensions)
            Array of n_samples data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob_Nx1 : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        posteriors : array_like, shape (n_samples, n_components)
            Posterior probability of each mixture component for each
            sample
        '''
        if self.weights is None or self.means is None or self.covars is None:
            raise ValueError('GMM parameters have not been initialized')

        if X.shape[1] != self.n_dimensions:
            raise ValueError(
                'input data matrix X is of shape %s, should be %s'
                % (X.shape, (X.shape[0], self.n_dimensions)))

        N = X.shape[0]

        if temp_gpu_mem is None:
            temp_gpu_mem = TempGPUMem()
        temp_gpu_mem.alloc(N, self.n_components, self.n_dimensions)

        # lpr = log_multivariate_normal_density()
        #        + np.log(self.weights)[None, :]
        # -----------------------------------------------------
        posteriors_NxK = log_multivariate_normal_density(
            X, self.means, self.covars,
            self.covariance_type, temp_gpu_mem)
        # lpr += np.log(self.weights)
        temp_Kx1 = temp_gpu_mem['temp_Kx1']
        cm.log(self.weights, target=temp_Kx1)
        temp_Kx1.reshape((1, self.n_components))  # transpose
        posteriors_NxK.add_row_vec(temp_Kx1)
        temp_Kx1.reshape((self.n_components, 1))  # original shape
        # in use: lpr -> 'NxK'

        # logprob_Nx1 = np.log(np.sum(np.exp(lpr - vmax), axis=1))
        # logprob_Nx1 += vmax
        # ---------------------------------------------------------
        vmax_Nx1 = temp_gpu_mem['vmax_Nx1']
        logprob_Nx1 = temp_gpu_mem['logprob_Nx1']
        # vmax_Nx1 = np.max(lpr, axis=1)
        posteriors_NxK.max(axis=1, target=vmax_Nx1)
        # lpr -= vmax_Nx1[:, None]
        posteriors_NxK.add_col_mult(vmax_Nx1, -1.0)
        # posteriors_NxK = np.exp(posteriors_NxK)
        cm.exp(posteriors_NxK)
        # logprob_Nx1 = np.sum(posteriors_NxK, axis=1)
        posteriors_NxK.sum(axis=1, target=logprob_Nx1)
        # posteriors_NxK /= logprob_Nx1[:, None]
        posteriors_NxK.div_by_col(logprob_Nx1)

        # logprob_Nx1 = np.log(logprob_Nx1)
        cm.log(logprob_Nx1, target=logprob_Nx1)
        # logprob_Nx1 += vmax_Nx1
        logprob_Nx1.add(vmax_Nx1)

        return logprob_Nx1, posteriors_NxK

    def score(self, X):
        '''Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_dimensions)
            List of 'n_samples' data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob_Nx1 : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        '''
        logprob_Nx1, _ = self.score_samples(X)
        return logprob_Nx1

    def predict(self, X):
        '''Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_dimensions]

        Returns
        -------
        C : array, shape = (n_samples,)
        '''
        _, posteriors = self.score_samples(X)
        return posteriors.argmax(axis=1)

    def compute_posteriors(self, X):
        '''Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_dimensions]

        Returns
        -------
        posteriors : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        '''
        _, posteriors = self.score_samples(X)
        return posteriors

    def fit(self, X,
            thresh=1e-2, n_iter=100, n_init=1,
            update_params='wmc', init_params='',
            random_state=None, verbose=None):
        '''Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_dimensions)
            List of 'n_samples' data points.  Each row
            corresponds to a single data point.
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
            'm' for means, and 'c' for covars.  Defaults to ''.
        random_state: numpy.random.RandomState
        verbose: bool, optional
            Whether to print EM iteration information during training
        '''
        if verbose is None:
            verbose = self.verbose

        if random_state is None:
            random_state = np.random.RandomState()
        else:
            check_random_state(random_state)

        if n_init < 1:
            raise ValueError('GMM estimation requires at least one run')
        if X.shape[1] != self.n_dimensions:
            raise ValueError(
                'input data matrix X is of shape %s, should be %s'
                % (X.shape, (X.shape[0], self.n_dimensions)))

        # copy observations to GPU
        X_gpu = return_CUDAMatrix(X)
        n_samples = X.shape[0]

        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        # allocate TempGPUMem
        temp_gpu_mem = TempGPUMem()
        temp_gpu_mem.alloc(n_samples, self.n_components, self.n_dimensions)

        max_log_prob = -np.infty

        for _ in range(n_init):
            if 'm' in init_params or self.means is None:
                perm = random_state.permutation(n_samples)
                self.means = return_CUDAMatrix(X[perm[:self.n_components]])

            if 'w' in init_params or self.weights is None:
                self.weights = return_CUDAMatrix(
                    (1.0/self.n_components)*np.ones(self.n_components))

            if 'c' in init_params or self.covars is None:
                if self.covariance_type == 'diag':
                    cv = np.var(X, axis=0) + self.min_covar
                    self.covars = return_CUDAMatrix(
                        np.tile(cv, (self.n_components, 1)))
                else:
                    raise ValueError('unsupported covariance type: %s'
                                     % self.covariance_type)

            # EM algorithms
            log_likelihood = []
            converged = False
            for i in range(n_iter):
                # Expectation step
                curr_log_likelihood, posteriors = self.score_samples(
                    X_gpu, temp_gpu_mem)
                curr_log_likelihood_sum = curr_log_likelihood.sum(
                    axis=0).asarray()[0, 0]
                log_likelihood.append(curr_log_likelihood_sum)
                if i > 0:
                    change = log_likelihood[-1] - log_likelihood[-2]
                else:
                    change = np.inf
                if verbose:
                    print('Iter: %u, log-likelihood: %g ' \
                        '(change = %g)' % (
                            i, curr_log_likelihood_sum, change))

                # Check for convergence.
                if change < thresh:
                    converged = True
                    break

                # Maximization step
                self._do_mstep(X_gpu, posteriors, update_params,
                               self.min_covar,
                               temp_gpu_mem)

            # if the results are better, keep it
            if n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                if n_init > 1:
                    best_params = {
                        'weights': self.get_weights(),
                        'means': self.get_means(),
                        'covars': self.get_covars()
                    }
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # n_iter == 0 occurs when using GMM within HMM
        if n_iter and n_init > 1:
            self.covars = return_CUDAMatrix(best_params['covars'])
            self.means = return_CUDAMatrix(best_params['means'])
            self.weights = return_CUDAMatrix(best_params['weights'])

        return converged

    def _do_mstep(
            self, X, posteriors,
            update_params, min_covar=0,
            temp_gpu_mem=None):
        ''' Perform the Mstep of the EM algorithm and return the class weights.
        '''

        N = X.shape[0]
        K, D = self.n_components, self.n_dimensions

        X = return_CUDAMatrix(X)

        if temp_gpu_mem is None:
            temp_gpu_mem = TempGPUMem()
        temp_gpu_mem.alloc(N, K, D)

        weights = temp_gpu_mem['temp_Kx1']

        weights.reshape((1, K))
        posteriors.sum(axis=0, target=weights)
        weights.reshape((K, 1))

        weighted_X_sum = temp_gpu_mem['weighted_X_sum_KxD']
        cm.dot(posteriors.T, X, target=weighted_X_sum)  # [KxN]x[NxD] -> [KxD]
        # inv_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        inv_weights = temp_gpu_mem['inv_weights_Kx1']
        denom = temp_gpu_mem['temp_Kx1_2']
        weights.add(10*EPS, target=denom)
        inv_weights.assign(1.0)
        inv_weights.divide(denom)

        if 'w' in update_params:
            # self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
            weights.div_by_row(
                weights.sum(axis=0).add(10*EPS),
                target=self.weights)
            weights.add(EPS)
        if 'm' in update_params:
            # self.means = weighted_X_sum * inv_weights
            # [KxD].*[Kx1]
            weighted_X_sum.mult_by_col(inv_weights, target=self.means)
        if 'c' in update_params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            temp_result = covar_mstep_func(
                self, X, posteriors, weighted_X_sum, inv_weights,
                min_covar, temp_gpu_mem)
            self.covars.assign(temp_result)

        return weights

    def _n_parameters(self):
        '''Return the number of free parameters in the model.'''
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


#########################################################################
# some helper routines
#########################################################################


def _log_multivariate_normal_density_diag(X, means, covars, temp_gpu_mem):
    '''Compute Gaussian log-density at X for a diagonal model'''

    N, D = X.shape
    K = means.shape[0]

    lpr_NxK = temp_gpu_mem['posteriors_NxK']
    inv_covars_KxD = temp_gpu_mem['temp_KxD_2']
    temp_NxD = temp_gpu_mem['temp_NxD']
    temp_KxD = temp_gpu_mem['temp_KxD']
    temp_Kx1 = temp_gpu_mem['temp_Kx1']

    # compute inverse variances
    inv_covars_KxD.assign(1.0)
    inv_covars_KxD.divide(covars)

    # lpr = D * np.log(2*np.pi)
    lpr_NxK.assign(D * np.log(2*np.pi))

    # temp_Kx1 =  np.sum(np.log(covars), 1)
    cm.log(covars, target=temp_KxD)
    temp_KxD.sum(axis=1, target=temp_Kx1)

    # temp_Kx1 += np.sum((means**2)/covars, 1)
    means.mult(means, target=temp_KxD)
    temp_KxD.mult(inv_covars_KxD)
    temp_Kx1.add_sums(temp_KxD, axis=1)

    # lpr += temp_Kx1
    temp_Kx1.reshape((1, K))  # transpose
    lpr_NxK.add_row_vec(temp_Kx1)
    temp_Kx1.reshape((K, 1))  # return to original shape

    # lpr += -2*np.dot(X, (means / covars).T)
    temp_KxD.assign(means)
    temp_KxD.mult(inv_covars_KxD)
    lpr_NxK.add_dot(X, temp_KxD.T, mult=-2.)

    # lpr += np.dot(X**2, (1.0 / covars).T)
    temp_NxD.assign(X)
    temp_NxD.mult(temp_NxD)
    lpr_NxK.add_dot(temp_NxD, inv_covars_KxD.T)

    # lpr *= -0.5
    lpr_NxK.mult(-0.5)

    # lpr_NxK still in use

    # lpr = -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
    #              + np.sum((means ** 2) / covars, 1)
    #              - 2 * np.dot(X, (means / covars).T)
    #              + np.dot(X ** 2, (1.0 / covars).T))

    return lpr_NxK


def _covar_mstep_diag(gmm, X, posteriors, weighted_X_sum, inv_weights,
                      min_covar, temp_gpu_mem):
    '''Performing the covariance M step for diagonal cases'''

    X2 = temp_gpu_mem['temp_NxD']
    X.mult(X, target=X2)  # X2 = X*X

    # avg_X2 = np.dot(posteriors.T, X2) * inv_weights # ([KxN]*[NxD]) * [Kx1]
    avg_X2 = temp_gpu_mem['temp_KxD']
    cm.dot(posteriors.T, X2, target=avg_X2)  # [KxN]x[NxD] -> [KxD]
    avg_X2.mult_by_col(inv_weights)

    # avg_means2 = gmm.means_ ** 2
    temp_KxD_2 = temp_gpu_mem['temp_KxD_2']
    gmm.means.mult(gmm.means, target=temp_KxD_2)
    # avg_X2 += avg_means2
    avg_X2.add(temp_KxD_2)

    # avg_X_means = gmm.means_ * weighted_X_sum * inv_weights
    # [KxD]*[KxD]*[Kx1] -> [KxD]
    gmm.means.mult(weighted_X_sum, target=temp_KxD_2)
    temp_KxD_2.mult_by_col(inv_weights)
    # avg_X2 -= 2*avg_X_means
    # import pdb; pdb.set_trace()
    avg_X2.add_mult(temp_KxD_2, alpha=-2.0)

    # avg_X2 += min_covar
    avg_X2.add(min_covar)

    # return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar
    # [KxD] - 2*[KxD] + [KxD] + [1]
    return avg_X2


_covar_mstep_funcs = {
    # 'spherical': _covar_mstep_spherical,
    'diag': _covar_mstep_diag,
    # 'tied': _covar_mstep_tied,
    # 'full': _covar_mstep_full,
}
