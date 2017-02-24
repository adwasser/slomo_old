"""Nefarious plotting"""

import dill as pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from emcee.autocorr import function as autocorr_function
from emcee.autocorr import integrated_time
from corner import corner

from .utils import get_params, radians_per_arcsec
from . import mass

label_map = {"r_s": r"$r_s$", "rho_s": r"$\rho_s$", "gamma": r"$\gamma$",
             "upsilon": r"$\Upsilon_*$", "beta_s": r"$\beta_s$",
             "beta_b": r"$\beta_b$", "beta_r": r"$\beta_r$", "dist": r"$D$",
             "phi_b": r"$\phi_b$", "alpha_s": r"$\alpha_*$",
             "I0_s": r"$\Sigma_{0, *}$", "Re_s": r"$R_{\mathrm{eff}, *}$",
             "n_s": r"$n_*$", "r_a": r"$r_a$"}

def plotstyle():
    mpl.rc("figure", figsize=(12, 8))
    mpl.rc("font", size=12, family="serif")
    mpl.rc("xtick", direction="in")
    mpl.rc("ytick", direction="in")
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

    
def corner_plot(model, chain, burn_fraction=0.5, **kwargs):
    """Make an enormous corner plot, rejecting the burn-in iterations at the start."""
    if isinstance(chain, str):
        chain = read_chain(chain)

    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = [label_map[name] for name in model.params.names]
    
    keep = round(niterations * burn_fraction)
    samples = chain[:, keep:, :].reshape((-1, ndim))
    # convert dist from kpc to Mpc    
    try:
        idx = model.params.names.index("dist")
        samples[:, idx] = 1e-3 * samples[:, idx]
    except ValueError as e:
        pass
        
    kwargs.update({'labels': labels, 'quantiles': [.16, .5, .84],
                   'hist_kwargs': {'lw': 2}, 'use_math_text': True,
                   'show_titles': True, 'title_kwargs': {'fontsize': 16},
                   'plot_datapoints': True, 'fill_contours': True,
                   'plot_density': True,
                   'contourf_kwargs': {'cmap': 'BuPu', 'colors': None}})
    fig = corner(samples, **kwargs)
    return fig


def mass_plot(model, chain, burn_fraction=0.5,
              rmax=500, nsamples=10000, size=50,
              dm_model=mass.M_gNFW, st_model=mass.M_sersic):
    """rmax: maximum radius in kpc, nsamples: number of posterior samples"""
    if isinstance(chain, str):
        chain = read_chain(chain)

    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    keep = round(niterations * burn_fraction)
    samples = chain[:, keep:, :].reshape((-1, ndim))
    idx = np.random.choice(np.arange(samples.shape[0]), nsamples)
    
    radii = np.logspace(0, np.log10(rmax), size)    
    dm_profiles = np.zeros((nsamples, size))
    st_profiles = np.zeros((nsamples, size))
    for i, param_values in enumerate(samples[idx]):
        param_map = model.params.mapping(param_values)
        kwargs = {**param_map, **model.constants}
        dm_profiles[i] = dm_model(radii, **kwargs)
        st_profiles[i] = st_model(radii, **kwargs)
        
    M_tot = dm_profiles + st_profiles
    M_dm_low, M_dm_med, M_dm_high = np.percentile(dm_profiles, [16, 50, 84], axis=0)
    M_star_low, M_star_med, M_star_high = np.percentile(st_profiles, [16, 50, 84], axis=0)
    M_tot_low, M_tot_med, M_tot_high = np.percentile(M_tot, [16, 50, 84], axis=0)
    r = radii
    
    fig, ax = plt.subplots()
    ax.plot(r, M_star_med, 'c-.', label='Stars')
    ax.fill_between(r, M_star_low, M_star_high, facecolor='c', alpha=0.3)
    ax.plot(r, M_dm_med, 'm--', label='DM')
    ax.fill_between(r, M_dm_low, M_dm_high, facecolor='m', alpha=0.3)
    ax.plot(r, M_tot_med, 'k-', label='Total')
    ax.fill_between(r, M_tot_low, M_tot_high, facecolor='k', alpha=0.3)
    ax.set_xlim(np.min(r), np.max(r))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel(r'Enclosed mass [M$_\odot$]')
    ax.legend(loc='best')

    return fig, ax
    
def data_plot(model, chain):
    pass


    