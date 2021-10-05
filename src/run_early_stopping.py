#!/usr/bin/env python

import argparse
import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from model import Dataset, Simulation
from plotting import coordinate_plot_with_error_bound, log2_error_plot


def main():
    parser = argparse.ArgumentParser(
        description="Illustration of effects of different N on early stopping")
    parser.add_argument("--name", default="cooridnate", type=str, help="name of task")
    parser.add_argument("--n", default=100,type=int,help="number of measurements")
    parser.add_argument("--p", default=200, type=int, help="number of feature dimensions")
    parser.add_argument("--k", default=5, type=int, help="sparsity level")
    parser.add_argument("--replicates", default=30, type=int, help="number of replicates")

    args = parser.parse_args()
    print("Running with following command line arguments: {}".
        format(args))
    
    n = args.n
    p = args.p
    k = args.k
    w_star = np.zeros((p,1))
    w_star[:k, 0] = np.ones(k)
    noise_std = .5
    seed = 42
    params_data = {'n': n, 'p': p, 'k': k,
            'noise_std': noise_std,
            'beta': w_star,
            "random_state": seed,
            }
    dataset = Dataset(**params_data)

    output_name = "./outputs/early_stopping_"+args.name+".pkl"
    fig_name = "./figs/early_stopping_"+args.name+".pdf"
    if_positive = False

    if args.name == 'coordinate':
        N_vec = [2,3,4]
        w0_vec = [.005**N for N in N_vec]
        lr = 1e-2
        epochs_vec = [3000, 50000, 3200000]
        num = len(N_vec)


        num = len(N_vec)
        if not os.path.exists(output_name):
            sims_vec = []
            for i in range(num):
                print(f"running: {i+1} out of {num}")
                N = N_vec[i]
                w0 = w0_vec[i]
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
        
        fig, axes = plt.subplots(1,3)
        fig.set_size_inches(16, 4)
        coordinate_plot_with_error_bound(sims_vec[0], axes[0])
        coordinate_plot_with_error_bound(sims_vec[1], axes[1])
        coordinate_plot_with_error_bound(sims_vec[2], axes[2])
        fig.text(0.525,0.00,r'Number of iterations $t$', ha='center')
        fig.tight_layout()
        fig.savefig(fig_name)

    elif args.name == "l2_error":
        N_vec = [2,3,4]
        w0 = 1e-5
        lr = 1e-2
        epochs_vec = [3000, 10000, 25000]
        replicates = args.replicates
        if not os.path.exists(output_name):
            sims_list = [[],[],[]]
            for i in range(replicates):
                print(f"running: {i+1} out of {replicates}")
                seed = i*1000
                params_data = {'n': n, 'p': p, 'k': k,
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
            pickle.dump(sims_list, open(output_name,"wb"))
        else:
            print('Found cache in outputs folder, skip running.')
            sims_list = pickle.load(open(output_name,"rb"))

        fig, axes = plt.subplots(1,3)
        fig.set_size_inches(16, 4)
        log2_error_plot(sims_list[0], axes[0])
        log2_error_plot(sims_list[1], axes[1])
        log2_error_plot(sims_list[2], axes[2])
        fig.text(0.525,0.00,r'Number of iterations $t$', ha='center')
        fig.tight_layout()
        fig.savefig(fig_name)

    else:
        sys.exit("Task name must be coordinate or l2_error, please double check or use other files.")
    print('Done.')

if __name__ == "__main__":
    main()