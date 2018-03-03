ggmm
====

Python module to train GMMs using CUDA (via CUDAMat)

### Contents
[Dependencies](#dependencies)  
[Installation](#installation)  
[Example usage](#example-usage)  
[Documentation](#documentation)  

### Dependencies

* Not Windows (only tested on Linux and Mac)
* CUDA 6.0+ (only tested with 6.0)
* numpy
* CUDAMat, avaiable here: https://github.com/cudamat/cudamat.git
* future: http://python-future.org/index.html
* nose (optional, for running tests)

### Installation

Clone ggmm and CUDAMat in local install path:
```bash
cd ${INSTALL_PATH}
git clone https://github.com/ebattenberg/ggmm.git
git clone https://github.com/cudamat/cudamat.git
```

Compile and install CUDAMat:
```bash
cd ${INSTALL_PATH}/cudamat
sudo python setup.py install
```
Run CUDAMat tests (optional, requires nose):
```bash
cd ${INSTALL_PATH}/cudamat
nosetests
```
Run ggmm tests (optional, requires nose):
```bash
cd ${INSTALL_PATH}/ggmm
nosetests
```
Install ggmm:
```bash
cd ${INSTALL_PATH}/ggmm
sudo pip install .
```

### Example Usage

```python
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
```

### Documentation
Documentation available [here](http://ebattenberg.github.io/ggmm)
