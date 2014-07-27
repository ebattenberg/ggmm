ggmm
====

Python module to train GMMs using CUDA (via CUDAMat)

####Dependencies

* Not Windows (not tested)
* CUDA 6.0+ (only tested with 6.0)
* numpy
* my fork of CUDAMat avaiable here: https://github.com/ebattenberg/cudamat
* nose (for running tests)

####Installation

Clone ggmm and CUDAMat fork in local install path:
```bash
cd ${INSTALL_PATH}
git clone https://github.com/ebattenberg/ggmm.git
git clone https://github.com/ebattenberg/cudamat.git
```
Update paths (add these to .bashrc if desired):
```bash
CUDA_BIN=/usr/local/cuda/bin # your CUDA binary path may be different
CUDA_LIB=/usr/local/cuda/lib64 # your CUDA library path may be different
export PATH=${CUDA_BIN}:$PATH
export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH

# Add ggmm and CUDAMat to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:${INSTALL_PATH}
```
Compile CUDAMat:
```bash
cd ${INSTALL_PATH}/cudamat
make
```
Run CUDAMat tests:
```
cd ${INSTALL_PATH}/cudamat
nosetests
```
Run ggmm tests (requires nose):
```bash
cd ${INSTALL_PATH}/ggmm
nosetests
```

####Example Usage

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

thresh = 1e-3
n_iter = 20

gmm.fit(X, thresh, n_iter)

weights = gmm.get_weights()
means = gmm.get_means()
covars = gmm.get_covars()
```
