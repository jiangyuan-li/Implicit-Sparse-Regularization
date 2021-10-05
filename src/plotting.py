import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsmath}'})
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'legend.frameon': False})

def coordinate_plot(Sim, ax, length=None):
    signal = Sim.signal
    noise_max = Sim.noise_max
    k = Sim.k
    N = Sim.N
    w0 = Sim.w0

    if length is not None:
        epochs = length
        signal = signal[:epochs, :]
        noise_max = noise_max[:epochs]
    else:
        epochs = len(noise_max)

    epochs_vec = [*range(epochs)]

    for j in range(k):
        line_signal, = ax.plot(epochs_vec, signal[:, j],
                               c='C0', label='signal')
    ax.hlines(1.0, 0, epochs, colors='C2', linestyle='dashed')

    line_noise, = ax.plot(epochs_vec, noise_max, c='red',
                          linestyle='dashed', label='noise')

    ax.legend(
        (line_signal, line_noise),
        (r'$i \in S$', r'$||\mathbf{e}_t||_{\infty}$'),
        loc='upper right')
    ax.set_ylabel(r'$w_{t,i}$')
    ax.set_title(f'N={Sim.N}')
    ax.set_ylim(0, 1.5)


def incremental_plot(Sim, ax, length=None):
    signal = Sim.signal
    noise_max = Sim.noise_max
    k = Sim.k
    N = Sim.N
    w0 = Sim.w0

    if length is not None:
        epochs = length
        signal = signal[:epochs, :]
        noise_max = noise_max[:epochs]
    else:
        epochs = len(noise_max)

    epochs_vec = [*range(epochs)]

    for j in range(k):
        line_signal, = ax.plot(epochs_vec, signal[:, j],
                               c='C0', label='signal')
    ax.hlines(1.0, 0, epochs, colors='C2', linestyle='dashed')
    ax.hlines(2.0, 0, epochs, colors='C2', linestyle='dashed')
    ax.hlines(3.0, 0, epochs, colors='C2', linestyle='dashed')
    ax.hlines(4.0, 0, epochs, colors='C2', linestyle='dashed')

    line_noise, = ax.plot(epochs_vec, noise_max, c='red',
                          linestyle='dashed', label='noise')

    ax.legend(
        (line_signal, line_noise),
        (r'$i \in S$', r'$||\mathbf{e}_t||_{\infty}$'),
        loc='upper right')
    ax.set_ylabel(r'$w_{t,i}$')
    ax.set_title(f'N={Sim.N}')
    ax.set_ylim(0, 5.5)


def coordinate_plot_with_error_bound(Sim, ax, length=None):
    signal = Sim.signal
    noise_max = Sim.noise_max
    k = Sim.k
    N = Sim.N
    w0 = Sim.w0

    if length is not None:
        epochs = length
        signal = signal[:epochs, :]
        noise_max = noise_max[:epochs]
    else:
        epochs = len(noise_max)

    epochs_vec = [*range(epochs)]



    for j in range(k):
        line_signal, = ax.plot(epochs_vec, signal[:, j],
                               c='C0', label='signal')
    ax.hlines(1.0, 0, epochs, colors='C2', linestyle='dashed')

    line_noise, = ax.plot(epochs_vec, noise_max, c='red',
                          linestyle='dashed', label='noise')
    line_error_bound = ax.hlines(w0**(1./4), 0, epochs, colors='orange',
                                 linestyle='dashed')
    start_idx = 0
    for j in epochs_vec:
        if max(abs(signal[j, :] - 1)) < .1:
            start_idx = j
            break

    ax.vlines(start_idx, 0, 1.5, colors='black',
              linestyle='dashed')

    stop_idx = epochs
    for j in epochs_vec:
        if noise_max[j] > w0**(1./4.):
            stop_idx = j
            break
    ax.vlines(stop_idx, 0, 1.5, colors='black',
              linestyle='dashed')

    ax.legend(
        (line_signal, line_noise, line_error_bound),
        (r'$i \in S$', r'$||\mathbf{e}_t||_{\infty}$', 'error bound'),
        loc='upper right')
    ax.set_ylabel(r'$w_{t,i}$')
    ax.set_title(f'N={Sim.N}, window size={stop_idx-start_idx}')
    ax.set_ylim(0, 1.6)

def log2_error_plot(Sim_list, ax=None, length=None):
    errs = [np.log2(x.l2_squared_errors) for x in Sim_list]
    errs = np.vstack(errs)
    
    if length is not None:
        epochs = length
        errs = errs[:,:length]
    else:
        epochs = errs.shape[1]
        
    means = np.median(errs, axis=0)
    lower_percentile = np.percentile(errs, 25, axis=0)
    upper_percentile = np.percentile(errs, 75, axis=0)
    
    bars = np.stack((means - lower_percentile, upper_percentile - means))

        
    epochs_vec = [*range(epochs)]
    line_width = 2
    
    idx = np.where(means < min(means)+.5)[0]

    start_idx = min(idx)
    end_idx = max(idx)
    ax.vlines(start_idx, -8, 2, colors='black', 
                             linestyle='dashed')

    ax.vlines(end_idx, -8, 2, colors='black', 
                             linestyle='dashed')
            
    ax.plot(epochs_vec, means)
    ax.fill_between(epochs_vec, lower_percentile, upper_percentile, alpha=.15)

    ax.set_ylabel(r'$\log_{2} ||\mathbf{w}_{t} - \mathbf{w}^{\star}||_{2}^{2}$')
    ax.set_title(f'N={Sim_list[0].N}, window size={end_idx-start_idx}')

def kernel_regime_plot(Sim_list, ridge_list, ax, length=None):

    errs = [np.log2(x.l2_squared_errors) for x in Sim_list]
    errs = np.vstack(errs)
    
    w_star = Sim_list[0].beta.numpy()

    if length is not None:
        epochs = length
        errs = errs[:, :length]
    else:
        epochs = errs.shape[1]

    means = np.median(errs, axis=0)
    lower_percentile = np.percentile(errs, 25, axis=0)
    upper_percentile = np.percentile(errs, 75, axis=0)

    epochs_vec = [*range(epochs)]

    ax.plot(epochs_vec, means)
    ax.fill_between(epochs_vec, lower_percentile, upper_percentile, alpha=.15)

    ridge_means = np.median(ridge_list)
    line_ridge = ax.hlines(ridge_means, 0, epochs, colors='orange',
                           linestyle='dashed')
    ridge_lower = np.percentile(ridge_list, 25)
    ridge_upper = np.percentile(ridge_list, 75)
    ax.fill_between(epochs_vec, ridge_lower, ridge_upper, alpha=.15)

    ls_list = [np.log2(((sim.beta_ls-w_star.flatten())**2).sum())
               for sim in Sim_list]
    ls_means = np.median(ls_list)
    ls_lower = np.percentile(ls_list, 25)
    ls_upper = np.percentile(ls_list, 75)
    ax.fill_between(epochs_vec, ls_lower, ls_upper, alpha=.15)

    line_ls = ax.hlines(ls_means, 0, epochs, colors='red',
                        linestyle='dashed')

    ax.set_ylabel(
        r'$\log_{2} ||\mathbf{w}_{t} - \mathbf{w}^{\star}||_{2}^{2}$')
    ax.set_title(f'N={Sim_list[0].N}')
    ax.legend(
        (line_ridge, line_ls),
        ('ridge', 'least square'),
        loc='upper right')
    ax.set_ylim(5.5, 8)
