ggmm
====

Python module to train GMMs using CUDA (via CUDAMat)

Requires my fork of CUDAMat avaiable here: https://github.com/ebattenberg/cudamat

Example usage:

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
