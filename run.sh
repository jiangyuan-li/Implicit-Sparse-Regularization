#!/bin/bash

set -x

python src/run_con_init.py --name convergence --n 500 --p 3000 --k 5
python src/run_con_init.py --name initialization_effects --n 500 --p 3000 --k 5
python src/run_early_stopping.py --name coordinate --n 100 --p 200 --k 5 --replicates 30
python src/run_early_stopping.py --name l2_error --n 100 --p 200 --k 5 --replicates 30
python src/run_incremental_learning.py --name incremental_learning --n 500 --p 3000 --k 4
python src/run_kernel_regime.py --name kernel_regime --n 500 --p 100 --replicates 30
python src/run_mnist.py --name mnist --n 392