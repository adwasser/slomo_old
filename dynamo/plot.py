"""Nefarious plotting"""

import dill as pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from emcee.autocorr import function as autocorr_function
from emcee.autocorr import integrated_time
from corner import corner

label_map = {"r_s": r"$r_s$", "rho_s": r"$\rho_s$", "gamma": r"$\gamma$",
             "upsilon": r"$\Upsilon_*$", "beta_s": r"$\beta_s$",
             "beta_b": r"$\beta_b$", "beta_r": r"$\beta_r$", "dist": r"$D$",
             "phi_b": r"$\phi_b$", "alpha_s": r"$\alpha_*$",
             "I0_s": r"$\Sigma_{0, *}$", "Re_s": r"$R_{\mathrm{eff}, *}$",
             "n_s": r"$n_*$"}

def plotstyle():
    mpl.rc("figure", figsize=(12, 8))
    mpl.rc("font", size=12, family="serif")
    mpl.rc("xtick", direction="in")
    mpl.rc("ytick", direction="out")
    mpl.rc("errorbar", capsize=3)
    mpl.rc("savefig", bbox="tight")
    mpl.rc("axes.formatter", limits=(-3, 3))
    mpl.rc("hist", bins="auto")
plotstyle()

def read_chain(filename):
    """Shape is (nwalkers, niterations, ndim)"""
    flatchain = np.loadtxt(filename)
    # remove walker label
    walker_labels = flatchain[:,0].flatten()
    nwalkers = int(walker_labels[-1]) + 1
    flatchain = flatchain[:,1:]
    return flatchain.reshape((nwalkers, -1, flatchain.shape[-1]))

def read_model(filename):
    return pickle.load(open(filename, 'rb'))

def walker_plot(model, chain, skip_step=100):
    """Does the walker choose the path or does the path choose the walker?"""
    if isinstance(chain, str):
        chain = read_chain(chain)
        
    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = [label_map[name] for name in model.params.names]
    
    # take every n-th iteration
    n = skip_step
    steps = n * np.arange(1, niterations/n + 1)

    fig, axarr = plt.subplots(ndim, sharex=True, figsize=(4, 1 * ndim))    
    for i in range(ndim):
        walkers = chain[:, :, i]
        for j in range(nwalkers):
            samples = chain[j, ::n, i]
            axarr[i].plot(steps, samples, alpha=0.3)
        axarr[i].set_ylabel(labels[i])
        axarr[i].set_yticks([walkers.min(), (walkers.min() + walkers.max()) / 2, walkers.max()])
        axarr[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    axarr[-1].set_xlabel('Step number')

    return fig, axarr


def autocorr_plot(model, chain, skip_step=100, **kwargs):
    """0 is good, 1 is bad"""
    if isinstance(chain, str):
        chain = read_chain(chain)

    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = [label_map[name] for name in model.params.names]
    flatchain = chain.reshape((-1, ndim))
    nsamples = flatchain.shape[0]
    
    acorr = autocorr_function(flatchain)
    # lower integrated autocorrelation times are better

    try:
        acorr_times = integrated_time(flatchain, **kwargs)
    except:
        acorr_times = np.zeros(ndim)
        
    n = skip_step
    steps = n * np.arange(1, flatchain.shape[0]/n + 1)
    fig, axarr = plt.subplots(ndim, sharex=True, figsize=(4, 1 * ndim))
    label_str = r'$\sqrt{\tau_\mathrm{int} / n} = $'
    for i in range(ndim):
        # rough estimate of uncertainty fraction on mean value
        unc = np.sqrt(acorr_times[i] / nsamples)
        axarr[i].plot(steps, acorr[::n, i], alpha=0.5)
        axarr[i].text(0.95, 0.90, label_str + '{:.2e}'.format(unc),
                      transform=axarr[i].transAxes, horizontalalignment='right',
                      verticalalignment='top')
        axarr[i].set_ylabel(labels[i])
        axarr[i].set_yticks([0, 0.5, 1])
    axarr[-1].set_xlabel('Iterations')
    axarr[-1].set_xticks([steps.min(), (steps.min() + steps.max()) / 2, steps.max()])
    axarr[-1].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0e'))
    return fig, axarr

    
def corner_plot(model, chain, burn_fraction=0.5):
    """Make an enormous corner plot, rejecting the burn-in iterations at the start."""
    if isinstance(chain, str):
        chain = read_chain(chain)

    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = [label_map[name] for name in model.params.names]

    # convert dist from kpc to Mpc    
    try:
        idx = model.params.names.index("dist")
        chain[:, :, idx] = 1e-3 * chain[:, :, idx]
    except ValueError as e:
        pass
    
    keep = round(niterations * burn_fraction)
    samples = chain[:, keep:, :].reshape((-1, ndim))
        
    kwargs = {'labels': labels, 'quantiles': [.16, .5, .84],
              'hist_kwargs': {'lw': 2}, 'use_math_text': True,
              'show_titles': True, 'title_kwargs': {'fontsize': 16},
              'plot_datapoints': True, 'fill_contours': True,
              'plot_density': True,
              'contourf_kwargs': {'cmap': 'BuPu', 'colors': None}}
    fig = corner(samples, **kwargs)
    return fig


def mass_plot(model, chain, burn_fraction=0.5, rmax=100):
    pass


def data_plot(model, chain):
    pass


    
