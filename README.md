# Implicit-Sparse-Regularization
Code for [Implicit Sparse Regularization: The Impact of Depth and Early Stopping](https://arxiv.org/abs/2108.05574)
# How to use
The convergence for different choices of depth N is shown on both simulated data and real data (MNIST). We also studied the effects of depth N on initialization scale and early stopping window. The connection to incremental learning and kernel regime is also empirically shown. To get the numerical results, simply run
```
chmod +x run.sh
./run.sh
```
# References
* Jiangyuan Li, Thanh V. Nguyen, Chinmay Hegde, Raymond K. W. Wong. (2021) "Implicit Sparse Regularization: The Impact of Depth and Early Stopping". Conference on Neural Information Processing Systems (NeurIPS).
