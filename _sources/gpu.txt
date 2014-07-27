
GPU/CUDAMat Backend
*********************

.. automodule:: ggmm.gpu

Example Usage
========================
Training a GMM::

    import ggmm.gpu as ggmm

    X = some_module.load_training_data()

    # N - training examples
    # D - data dimension
    # K - number of GMM components
    N, D = X.shape 
    K = 128

    ggmm.init()
    gmm = ggmm.GMM(K,D)

    thresh = 1e-3 # convergence threshold
    n_iter = 20 # maximum number of EM iterations
    init_params = 'wmc' # initialize weights, means, and covariances

    # train GMM
    gmm.fit(X, thresh, n_iter, init_params=init_params)

    # retrieve parameters from trained GMM
    weights = gmm.get_weights()
    means = gmm.get_means()
    covars = gmm.get_covars()

    # compute posteriors of data
    posteriors = gmm.compute_posteriors(X)


Reference
================

.. autofunction:: ggmm.gpu.init
.. autofunction:: ggmm.gpu.shutdown

.. autoclass:: ggmm.gpu.GMM
    :members:
