#!/usr/bin/env python

import argparse
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glmnet
from model import Dataset, Simulation
from plotting import kernel_regime_plot

def main():
    parser = argparse.ArgumentParser(
        description="Illustration of kernel regime for different N")
    parser.add_argument("--name", default="kernel_regime", type=str, help="name of task")
    parser.add_argument("--n", default=500,type=int,help="number of measurements")
    parser.add_argument("--p", default=100, type=int, help="number of feature dimensions")
    parser.add_argument("--replicates", default=30, type=int, help="number of replicates")
 
    args = parser.parse_args()
    print("Running with following command line arguments: {}".
        format(args))
    
    output_name = "./outputs/"+args.name+".pkl"
    n = args.n
    p = args.p
    noise_std = 25
    np.random.seed(42)
    w_star = np.random.normal(0,1,p).reshape(-1,1)

    N_vec = [2,3,4]
    w0 = 1e3
    lr = 1e-7
    epochs_vec = [5000,300,100]
    if_positive = False
    replicates = args.replicates

    if not os.path.exists(output_name):
        sims_list = [[],[],[]]
        ridge_list = []
        for i in range(replicates):
            print(f"running: {i+1} out of {replicates}")
            seed = i*1000
            params_data = {'n': n, 'p': p, 'k': -1,
                    'noise_std': noise_std,
                    'beta': w_star,
                    "random_state": seed,
                    }
            dataset = Dataset(**params_data)
            for j in range(3):
                N = N_vec[j]
                epochs = epochs_vec[j]
                params = {'dataset': dataset, 
                'N': N,
                'w0': w0,
                'lr': lr,
                'epochs': epochs,
                'if_positive': if_positive}
                sim = Simulation(**params)
                sim.train()
                sims_list[j].append(sim)
            ridge = glmnet.linear.ElasticNet(alpha=0,standardize=False,n_splits=5, n_lambda=1000, random_state=42)
            ridge.fit(dataset.X.numpy(), dataset.y.numpy().flatten())
            ridge_error = np.log2(((ridge.coef_-w_star.flatten())**2).sum())
            ridge_list.append(ridge_error)
        pickle.dump([sims_list, ridge_list], open(output_name,"wb"))
    else:
        print('Found cache in outputs folder, skip running.')
        sims_list, ridge_list = pickle.load(open(output_name,"rb"))
    
    fig_name = "./figs/"+args.name+".pdf"
    fig, axes = plt.subplots(1,3)
    fig.set_size_inches(16, 4)
    kernel_regime_plot(sims_list[0], ridge_list, axes[0])
    kernel_regime_plot(sims_list[1], ridge_list, axes[1])
    kernel_regime_plot(sims_list[2], ridge_list, axes[2])
    fig.text(0.525,0.00,r'Number of iterations $t$', ha='center')
    fig.tight_layout()
    fig.savefig(fig_name)
    print('Done.')

if __name__ == "__main__":
    main()