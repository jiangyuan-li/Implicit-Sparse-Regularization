#!/usr/bin/env python

import argparse
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from model import Dataset, Simulation
from plotting import incremental_plot

def main():
    parser = argparse.ArgumentParser(
        description="Illustration of the incremental learning effects for different N")
    parser.add_argument("--name", default="incremental_learning", type=str, help="name of task")
    parser.add_argument("--n", default=500,type=int,help="number of measurements")
    parser.add_argument("--p", default=3000, type=int, help="number of feature dimensions")
    parser.add_argument("--k", default=4, type=int, help="sparsity level")

    args = parser.parse_args()
    print("Running with following command line arguments: {}".
        format(args))
    
    n = args.n
    p = args.p
    k = args.k
    w_star = np.zeros((p,1))
    w_star[:k, 0] = [1,2,3,4]
    noise_std = .5
    seed = 42
    params_data = {'n': n, 'p': p, 'k': k,
            'noise_std': noise_std,
            'beta': w_star,
            "random_state": seed,
            }
    dataset = Dataset(**params_data)

    output_name = "./outputs/"+args.name+".pkl"

    N_vec = [2,3,5]
    w0 = 1e-4
    lr = 1e-3
    epochs_vec = [2500, 6000, 15000]
    if_positive = False
    num = len(N_vec)
    if not os.path.exists(output_name):
        sims_vec = []
        for i in range(num):
            print(f"running: {i+1} out of {num}")
            N = N_vec[i]
            epochs = epochs_vec[i]
            params = {'dataset': dataset, 
            'N': N,
            'w0': w0,
            'lr': lr,
            'epochs': epochs,
            'if_positive': if_positive}
            sim = Simulation(**params)
            sim.train()
            sims_vec.append(sim)
        pickle.dump(sims_vec, open(output_name,"wb"))
    else:
        print('Found cache in outputs folder, skip running.')
        sims_vec = pickle.load(open(output_name,"rb"))
    
    fig_name = "./figs/"+args.name+".pdf"
    fig, axes = plt.subplots(1,3)
    fig.set_size_inches(16, 4)
    incremental_plot(sims_vec[0], axes[0])
    incremental_plot(sims_vec[1], axes[1])
    incremental_plot(sims_vec[2], axes[2])
    fig.text(0.525,0.00,r'Number of iterations $t$', ha='center')
    fig.tight_layout()
    fig.savefig(fig_name)
    print('Done.')

if __name__ == "__main__":
    main()