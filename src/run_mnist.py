#!/usr/bin/env python

import argparse
import os, sys
import pickle
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import ExpMNIST
from utils import get_measurement

def main():
    parser = argparse.ArgumentParser(
        description="Illustration of convergence of different N on MNIST")
    parser.add_argument("--name", default="mnist", type=str, help="name of task")
    parser.add_argument("--n", default=392,type=int,help="number of measurements")

    args = parser.parse_args()
    print("Running with following command line arguments: {}".
        format(args))
    
    mnist = datasets.MNIST(
    root = "./data",
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    )

    n = args.n
    N_vec = [2,3,5]
    w0 = 1e-3
    lr_vec = [5e-3/N**2 for N in N_vec]
    epochs = 10000

    output_name = "./outputs/"+args.name+".pkl"
    fig_name = "./figs/"+args.name+".pdf"
    if not os.path.exists(output_name):
        sims_list = [[],[],[]]
        for i in range(3):
            one_pt = mnist.data[i]
            y, X = get_measurement(one_pt, n=n)
            sims_list[i].append(one_pt)
            for j in range(3):
                print(f"running: ({i+1},{j+1}) out of {3,3}")
                N = N_vec[j]
                lr = lr_vec[j]
                sim = ExpMNIST(N=N, w0=w0, lr=lr, epochs=epochs, 
                X=X, y=y, one_pt=one_pt)
                sim.train()
                im = 255*sim.model.get_params().detach().reshape(28,28)
                sims_list[i].append(im)
        pickle.dump(sims_list, open(output_name,"wb"))
    else:
        print('Found cache in outputs folder, skip running.')
        sims_list = pickle.load(open(output_name,"rb"))
    title_list = ["Original", "N=2", "N=3", "N=5"]
    fig, axes = plt.subplots(3,4)
    fig.set_size_inches(16, 12)
    for i in range(3):
        for j in range(4):
            axes[i,j].imshow(sims_list[i][j], cmap="gray")
            axes[i,j].axis('off')
            if i==0:
                _ = axes[i,j].set_title(title_list[j], fontsize=24)
    fig.tight_layout()
    fig.savefig(fig_name)
    print("Done.")

if __name__ == "__main__":
    main()