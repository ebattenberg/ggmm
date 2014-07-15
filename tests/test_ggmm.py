import logging
import time
import os
from nose.tools import *
import numpy as np

import ggmm.gpu as ggmm
import ggmm.cpu as cgmm # for comparison tests

EPS = 1e-6

# ----------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------
script_name = os.path.basename(__file__)
log_file = os.path.splitext(script_name)[0] + '.log'
log_format = '%(asctime)s %(levelname)s (%(name)s) %(message)s'
logging.basicConfig(filename=log_file,level=logging.WARN,format=log_format)
logger = logging.getLogger(os.path.basename(__file__))

ggmm_logger = logging.getLogger('ggmm')
ggmm_logger.setLevel(logging.WARNING)



# ----------------------------------------------------------------------
# test setup
# ----------------------------------------------------------------------

def setup():
    ggmm.init() # activates cublas

def teardown():
    ggmm.shutdown() # deactivates cublas

DIMENSIONS = [
        (8,4,2),
        (100,16,32),
        (10000,128,13),
]

TIME_DIMENSIONS = [
        (1000,16,200),
        (1000,128,13),
        #(10000,128,13),
        #(100000,512,40),
        #(10000,512,40),
]
    




# ------------------------------------------
# covariance types
# ------------------------------------------
INVALID_COVAR_TYPES = ['full','tied','spherical','woah!']
        
def test_invalid_covar_type():
    for covar_type in INVALID_COVAR_TYPES:
        yield check_invalid_covar_type, covar_type

def check_invalid_covar_type(covar_type):
    assert_raises(ValueError, ggmm.GMM,1,1,covar_type)


# ------------------------------------------
# set_weights
# ------------------------------------------



def test_set_weights_wrong_dim():
    gmm = ggmm.GMM(5,10)
    assert_raises(ValueError,gmm.set_weights,np.random.randn(4))

def test_set_weights_not_normalized():
    gmm = ggmm.GMM(5,10)
    input_weights = np.random.rand(5)
    while input_weights.sum() == 1.0:
        input_weights = np.random.randn(5)
    assert_raises(ValueError,gmm.set_weights,input_weights)

def test_set_weights_negative():
    gmm = ggmm.GMM(5,10)
    input_weights = np.random.randn(5)
    while np.all(input_weights >= 0):
        input_weights = np.random.randn(5)
    assert_raises(ValueError,gmm.set_weights,input_weights)

def test_set_weights():
    gmm = ggmm.GMM(5,10)
    input_weights = np.random.rand(5).astype('float32')
    input_weights /= input_weights.sum()
    gmm.set_weights(input_weights)
    assert_true(np.all(gmm.get_weights() == input_weights))
    assert_false(gmm.get_weights() is input_weights)

# ------------------------------------------
# set_means
# ------------------------------------------

WEIGHT_INPUT = [ 
        #(K,D), weight_dim, expected_error
        ((5,10), 5, None),
        ((5,10), 4, ValueError),
        ((5,10), 6, ValueError),
]

def test_set_means_wrong_dim1():
    gmm = ggmm.GMM(5,10)
    assert_raises(ValueError,gmm.set_means,np.random.randn(4,10))

def test_set_means_wrong_dim2():
    gmm = ggmm.GMM(5,10)
    assert_raises(ValueError,gmm.set_means,np.random.randn(5,11))

def test_set_means():
    gmm = ggmm.GMM(5,10)
    input_means = np.random.randn(5,10).astype('float32')
    gmm.set_means(input_means)
    assert_true(np.all(gmm.get_means() == input_means))
    assert_false(gmm.get_means() is input_means)

# ------------------------------------------
# set_covars
# ------------------------------------------

def test_set_covars_wrong_dim1():
    gmm = ggmm.GMM(5,10)
    assert_raises(ValueError,gmm.set_covars,np.random.rand(4,10))

def test_set_covars_wrong_dim2():
    gmm = ggmm.GMM(5,10)
    assert_raises(ValueError,gmm.set_covars,np.random.rand(5,11))

def test_set_weights_negative():
    gmm = ggmm.GMM(5,10)
    input_covar = (-1) * np.random.rand(5,10)
    assert_raises(ValueError,gmm.set_covars,input_covar)

def test_set_covars():
    min_covar = 1e-3
    gmm = ggmm.GMM(5,10,min_covar=min_covar)
    input_covars = np.random.rand(5,10).astype('float32')
    input_covars[input_covars < min_covar] = min_covar
    gmm.set_covars(input_covars)
    assert_true(np.all(gmm.get_covars() == input_covars))
    assert_false(gmm.get_covars() is input_covars)

def test_set_covars_min_covar():
    min_covar = 1e-3
    gmm = ggmm.GMM(5,10,min_covar=min_covar)
    input_covars = np.random.rand(5,10).astype('float32')
    input_covars[0] = min_covar/2
    gmm.set_covars(input_covars)
    input_covars[input_covars < min_covar] = min_covar
    assert_true(np.all(gmm.get_covars() == input_covars))
    assert_false(gmm.get_covars() is input_covars)

# ------------------------------------------
# log_multivariate_normal_density (diag)
# ------------------------------------------
def test_log_multivariate_normal_density_diag():
    for N,K,D in DIMENSIONS:
        check_log_multivariate_normal_density_diag, N, K, D

def check_log_multivariate_normal_density_diag(N,K,D):
    random_state = np.random.RandomState(123)
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)


    lpr_cpu = cgmm.log_multivariate_normal_density(X,means,covars, covariance_type='diag')
    lpr_gpu = ggmm.log_multivariate_normal_density(X,means,covars, covariance_type='diag')

    max_dev = np.max(np.abs((lpr_gpu.asarray() - lpr_cpu)/(lpr_cpu+EPS)))
    assert_less(max_dev,1e-5)

# ------------------------------------------
# GMM.score_samples
# ------------------------------------------
def test_score_samples():
    for N,K,D in DIMENSIONS:
        yield check_score_samples, N, K, D

def check_score_samples(N,K,D):
    random_state = np.random.RandomState(123)
    weights = random_state.rand(K)
    weights /= weights.sum()
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_cpu.set_weights(weights)
    gmm_cpu.set_means(means)
    gmm_cpu.set_covars(covars)

    gmm_gpu = ggmm.GMM(K,D)
    gmm_gpu.set_weights(weights)
    gmm_gpu.set_means(means)
    gmm_gpu.set_covars(covars)

    logprob_cpu, posterior_cpu = gmm_cpu.score_samples(X)
    logprob_gpu, posterior_gpu = gmm_gpu.score_samples(X)

    max_logprob_dev = np.max(np.abs((logprob_gpu.asarray().flatten() - logprob_cpu)/(logprob_cpu+EPS)))
    max_posterior_dev = np.max(np.abs((posterior_gpu.asarray() - posterior_cpu)/(posterior_cpu+EPS)))

    assert_less(max_logprob_dev, 1e-4)
    assert_less(max_posterior_dev, 1e-3)

def test_time_score_samples():
    for N,K,D in TIME_DIMENSIONS:
        yield check_time_score_samples, N, K, D

def check_time_score_samples(N,K,D):
    random_state = np.random.RandomState(123)
    weights = random_state.rand(K)
    weights /= weights.sum()
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_cpu.set_weights(weights)
    gmm_cpu.set_means(means)
    gmm_cpu.set_covars(covars)

    gmm_gpu = ggmm.GMM(K,D)
    gmm_gpu.set_weights(weights)
    gmm_gpu.set_means(means)
    gmm_gpu.set_covars(covars)


    MAX_TRIALS = 10
    MAX_TIME = 1.0

    temp_gpu_mem = ggmm.TempGPUMem()
    X_gpu = ggmm.return_CUDAMatrix(X)
    gpu_trials = 0
    t0 = time.time()
    for i in xrange(MAX_TRIALS):
        logprob_gpu, posterior_gpu = gmm_gpu.score_samples(X_gpu,temp_gpu_mem)
        gpu_time = time.time() - t0
        gpu_trials += 1
        if gpu_time > MAX_TIME:
            break
    avg_gpu_time = gpu_time/gpu_trials

    t0 = time.time()
    cpu_trials = 0
    for i in xrange(MAX_TRIALS):
        logprob_cpu, posterior_cpu = gmm_cpu.score_samples(X)
        cpu_time = time.time() - t0
        cpu_trials += 1
        if cpu_time > MAX_TIME:
            break
    avg_cpu_time = cpu_time/cpu_trials

    speedup = avg_cpu_time/avg_gpu_time

    print ''
    print '------------------------------------'
    print 'N=%u, K=%u, D=%u' % (N,K,D)
    print 'avg_cpu_time (%u trials):' % cpu_trials, avg_cpu_time
    print 'avg_gpu_time (%u trials):' % gpu_trials, avg_gpu_time
    print 'speedup:', speedup
    print '------------------------------------'
    print ''
    assert_greater(speedup, 1.0)

# ------------------------------------------
# GMM._do_mstep
# ------------------------------------------


def test_do_mstep():
    for N,K,D in DIMENSIONS:
        yield check_do_mstep, N, K, D

def check_do_mstep(N,K,D):
    random_state = np.random.RandomState(456)
    weights = random_state.rand(K)
    weights /= weights.sum()
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_cpu.set_weights(weights)
    gmm_cpu.set_means(means)
    gmm_cpu.set_covars(covars)

    gmm_gpu = ggmm.GMM(K,D)
    gmm_gpu.set_weights(weights)
    gmm_gpu.set_means(means)
    gmm_gpu.set_covars(covars)

    logprob_cpu, posterior_cpu = gmm_cpu.score_samples(X)
    logprob_gpu, posterior_gpu = gmm_gpu.score_samples(X)

    update_params = 'wmc'
    min_covar = 1e-3
    gmm_cpu._do_mstep(X,posterior_cpu, update_params, min_covar)
    gmm_gpu._do_mstep(X,posterior_gpu, update_params, min_covar)

    max_weights_dev = np.max(np.abs((gmm_gpu.get_weights() - gmm_cpu.get_weights())/(gmm_cpu.get_weights()+EPS)))
    max_means_dev = np.max(np.abs((gmm_gpu.get_means() - gmm_cpu.get_means())/(gmm_cpu.get_means()+EPS)))
    max_covars_dev = np.max(np.abs((gmm_gpu.get_covars() - gmm_cpu.get_covars())/(gmm_cpu.get_covars()+EPS)))

    #precision_scale = np.sqrt(N*K*D)
    precision_scale = 1.0
    precision = 1e-2
    assert_less(max_weights_dev, precision_scale*precision)
    assert_less(max_means_dev, precision_scale*precision)
    assert_less(max_covars_dev, precision_scale*precision)

def test_time_do_mstep():
    for N,K,D in TIME_DIMENSIONS:
        yield check_time_do_mstep, N, K, D

def check_time_do_mstep(N,K,D):
    random_state = np.random.RandomState(123)
    weights = random_state.rand(K)
    weights /= weights.sum()
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_cpu.set_weights(weights)
    gmm_cpu.set_means(means)
    gmm_cpu.set_covars(covars)

    gmm_gpu = ggmm.GMM(K,D)
    gmm_gpu.set_weights(weights)
    gmm_gpu.set_means(means)
    gmm_gpu.set_covars(covars)

    X_gpu = ggmm.return_CUDAMatrix(X)
    temp_gpu_mem = ggmm.TempGPUMem()
    logprob_cpu, posterior_cpu = gmm_cpu.score_samples(X)
    logprob_gpu, posterior_gpu = gmm_gpu.score_samples(X_gpu,temp_gpu_mem)

    update_params = 'wmc'
    min_covar = 1e-3

    MAX_TRIALS = 10
    MAX_TIME = 1.0

    gpu_trials = 0
    t0 = time.time()
    for i in xrange(MAX_TRIALS):
        gmm_gpu._do_mstep(X_gpu,posterior_gpu, update_params, min_covar)
        gpu_time = time.time() - t0
        gpu_trials += 1
        if gpu_time > MAX_TIME:
            break
    avg_gpu_time = gpu_time/gpu_trials

    t0 = time.time()
    cpu_trials = 0
    for i in xrange(MAX_TRIALS):
        gmm_cpu._do_mstep(X,posterior_cpu, update_params, min_covar)
        cpu_time = time.time() - t0
        cpu_trials += 1
        if cpu_time > MAX_TIME:
            break
    avg_cpu_time = cpu_time/cpu_trials

    speedup = avg_cpu_time/avg_gpu_time

    print ''
    print '------------------------------------'
    print 'N=%u, K=%u, D=%u' % (N,K,D)
    print 'avg_cpu_time (%u trials):' % cpu_trials, avg_cpu_time
    print 'avg_gpu_time (%u trials):' % gpu_trials, avg_gpu_time
    print 'speedup:', speedup
    print '------------------------------------'
    print ''
    assert_greater(speedup, 1.0)

# ------------------------------------------
# GMM.fit
# ------------------------------------------


def test_fit():
    for N,K,D in DIMENSIONS:
        yield check_fit, N, K, D

def check_fit(N,K,D):
    random_state = np.random.RandomState(456)
    weights = random_state.rand(K)
    weights /= weights.sum()
    means = random_state.randn(K,D)
    covars = random_state.rand(K,D)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_cpu.set_weights(weights)
    gmm_cpu.set_means(means)
    gmm_cpu.set_covars(covars)

    gmm_gpu = ggmm.GMM(K,D)
    gmm_gpu.set_weights(weights)
    gmm_gpu.set_means(means)
    gmm_gpu.set_covars(covars)

    thresh = 0.0
    n_iter = 5
    init_params = ''

    gmm_cpu.fit(X, thresh, n_iter, init_params=init_params)
    gmm_gpu.fit(X, thresh, n_iter, init_params=init_params)

    max_weights_dev = np.max(np.abs((gmm_gpu.get_weights() - gmm_cpu.get_weights())/(gmm_cpu.get_weights()+EPS)))
    max_means_dev = np.max(np.abs((gmm_gpu.get_means() - gmm_cpu.get_means())/(gmm_cpu.get_means()+EPS)))
    max_covars_dev = np.max(np.abs((gmm_gpu.get_covars() - gmm_cpu.get_covars())/(gmm_cpu.get_covars()+EPS)))

    #precision_scale = np.sqrt(N*K*D)
    precision_scale = 1.0
    precision = 1e-2
    assert_less(max_weights_dev, precision_scale*precision)
    assert_less(max_means_dev, precision_scale*precision)
    assert_less(max_covars_dev, precision_scale*precision)

def test_time_fit():
    for N,K,D in TIME_DIMENSIONS:
        yield check_time_fit, N, K, D

def check_time_fit(N,K,D):
    random_state = np.random.RandomState(123)
    X = random_state.randn(N,D)

    gmm_cpu = cgmm.GMM(K,D)
    gmm_gpu = ggmm.GMM(K,D)

    thresh = 0.0
    n_iter = 5
    init_params = 'wmc'

    gmm_cpu.fit(X, thresh, n_iter, init_params=init_params)
    gmm_gpu.fit(X, thresh, n_iter, init_params=init_params)


    MAX_TRIALS = 10
    MAX_TIME = 1.0

    gpu_trials = 0
    t0 = time.time()
    for i in xrange(MAX_TRIALS):
        gmm_gpu.fit(X, thresh, n_iter, init_params=init_params)
        gpu_time = time.time() - t0
        gpu_trials += 1
        if gpu_time > MAX_TIME:
            break
    avg_gpu_time = gpu_time/gpu_trials

    t0 = time.time()
    cpu_trials = 0
    for i in xrange(MAX_TRIALS):
        gmm_cpu.fit(X, thresh, n_iter, init_params=init_params)
        cpu_time = time.time() - t0
        cpu_trials += 1
        if cpu_time > MAX_TIME:
            break
    avg_cpu_time = cpu_time/cpu_trials

    speedup = avg_cpu_time/avg_gpu_time

    print ''
    print '------------------------------------'
    print 'N=%u, K=%u, D=%u' % (N,K,D)
    print 'avg_cpu_time (%u trials):' % cpu_trials, avg_cpu_time
    print 'avg_gpu_time (%u trials):' % gpu_trials, avg_gpu_time
    print 'speedup:', speedup
    print '------------------------------------'
    print ''
    assert_greater(speedup, 1.0)

def test_correct_fit():
    K,D = 2,2
    gmm = ggmm.GMM(K,D)

    random_state = np.random.RandomState(seed=123)

    N = 1000
    true_means = np.array([[5,3],[-5,-3]]).astype('float32')
    true_covars = np.array([[1,1],[1,1]]).astype('float32')
    true_weights = np.array([0.5,0.5]).astype('float32')
    sample_component = np.where(random_state.multinomial(1,true_weights,size=N))[1]
    sample_mean = true_means[sample_component]
    sample_noise = np.sqrt(true_covars[sample_component])*random_state.randn(N,D).astype('float32')
    samples = sample_mean + sample_noise


    gmm.fit(samples,n_init=3,random_state=random_state)

    # match learned components to true components
    if (np.sum(np.abs(true_means - gmm.get_means())) 
            > np.sum(np.abs(true_means - gmm.get_means()[::-1]))):
        true_means = true_means[::-1]
        true_covars = true_covars[::-1]
        true_weights = true_weights[::-1]

    assert_less(np.max(np.abs(gmm.get_means() - true_means)), 0.1)

