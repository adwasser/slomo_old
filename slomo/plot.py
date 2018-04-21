"""Nefarious plotting"""
from itertools import cycle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from emcee.autocorr import function as autocorr_function
from emcee.autocorr import integrated_time
from corner import corner

from .utils import radians_per_arcsec, G
from . import io

__all__ = [
    "walker_plot",
    "autocorr_plot",
    "corner_plot",
    "component_plot"
]


label_map = {
    "r_s": r"$\log_{10} r_s$",
    "rho_s": r"$\log_{10} \rho_s$",
    "gamma": r"$\gamma$",
    "upsilon": r"$\Upsilon_*$",
    "beta_s": r"$\tilde{\beta}_*$",
    "beta_b": r"$\tilde{\beta}_b$",
    "beta_r": r"$\tilde{\beta}_r$",
    "dist": r"$D$",
    "phi_b": r"$\phi_b$",
    "I0_s": r"$\log_{10} \Sigma_{0, *}$",
    "Re_s": r"$\log_{10} R_{\mathrm{eff}, *}$",
    "n_s": r"$n_*$",
    "r_a": r"$r_a$",
    "M200": r"$\log_{10} M_{200}$",
    "c200": r"$c_{200}$",
    "alpha_stars": r"$\alpha_*$",
    "alpha_gc": r"$\alpha_\mathrm{gc}$",
    "alpha_mass": r"$\alpha_m$",
    "alpha_ms": r"$\alpha_\mathrm{ms}$",
    "alpha_ls": r"$\alpha_\mathrm{ls}$",
    "alpha_sp": r"$\alpha_\mathrm{sp}$",
    "M_bh": r"$\log_{10}$ M$_\mathrm{bh}$",
    "gamma_tot": r"$\gamma_\mathrm{tot}$",
    "rho0": r"$\rho_0$",
    "vsys": r"$v_\mathrm{sys}$"
}

corner_kwargs = {
    'quantiles': [.16, .5, .84],
    'hist_kwargs': {
        'lw': 2
    },
    'use_math_text': True,
    'show_titles': True,
    'title_kwargs': {
        'fontsize': 16
    },
    'plot_datapoints': True,
    'fill_contours': True,
    'plot_density': True,
    'contourf_kwargs': {
        'cmap': 'BuPu',
        'colors': None
    }
}


def walker_plot(outfile, skip_step=100):
    """Does the walker choose the path or does the path choose the walker?
    
    Parameters
    ----------
    outfile : str
        hdf5 file name
    skip_step : int, optional
        number of steps to skip between iterations to save on compute time
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axarr : ndarray
        2d array of matplotlib.axes._subplots.AxesSubplot instances
    """
    chain = io.read_dataset(outfile, "chain")
    model = io.read_model(outfile)

    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = []
    for i, name in enumerate(model.params.names):
        if name in label_map:
            labels.append(label_map[name])
        else:
            labels.append(name)

    # take every n-th iteration
    n = skip_step
    steps = n * np.arange(1, niterations / n + 1)

    ncols = int(np.sqrt(ndim))
    nrows = int(np.ceil(ndim / ncols))
    fig, axarr = plt.subplots(
        nrows, ncols, sharex="col", figsize=(4.8 * ncols, 2.4 * nrows))
    fig.tight_layout(w_pad=1.5, h_pad=0.5)
    for i in range(ndim):
        col = i % ncols
        row = int(np.floor((i - col) / ncols))
        walkers = chain[:, :, i]
        try:
            ax = axarr[row][col]
        except TypeError:
            # only one row
            ax = axarr[row if nrows > ncols else col]
        for j in range(nwalkers):
            samples = chain[j, ::n, i]
            ax.plot(steps, samples, alpha=0.3)
        ax.annotate(
            labels[i],
            xy=(0.1, 0.8),
            xycoords="axes fraction",
            bbox={"fc": "w",
                  "ec": "k",
                  "pad": 4.0,
                  "alpha": 0.5})
        ax.set_yticks([
            walkers.min(), (walkers.min() + walkers.max()) / 2,
            walkers.max()
        ])
        ax.yaxis.set_major_formatter(
            mpl.ticker.FormatStrFormatter("%.1f"))
    for col in range(ncols):
        try:
            axarr[-1][col].set_xlabel('Step number')
        except TypeError:
            axarr[-1].set_xlabel('Step number')
    return fig, axarr


def autocorr_plot(outfile, skip_step=100, **kwargs):
    """Autocorrelation plots.
    0 is good, 1 is bad
    extra keyword arguments get passed into emcee.autocorr.intergrated_time
    
    Parameters
    ----------
    outfile : str
        hdf5 file name
    skip_step : int, optional
        number of steps to skip to thin the flattened chain

    Returns
    -------
    fig : matplotlib.figure.Figure
    axarr : ndarray
        2d array of matplotlib.axes._subplots.AxesSubplot instances
    """
    chain = io.read_dataset(outfile, "chain")
    model = io.read_model(outfile)
    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = []
    for i, name in enumerate(model.params.names):
        if name in label_map:
            labels.append(label_map[name])
        else:
            labels.append(name)
    # lower integrated autocorrelation times are better
    flatchain = chain.reshape((-1, ndim))
    nsamples = flatchain.shape[0]
    acorr = autocorr_function(flatchain)
    try:
        acorr_times = integrated_time(flatchain, **kwargs)
    except:
        acorr_times = np.zeros(ndim)
    n = skip_step
    steps = n * np.arange(1, flatchain.shape[0] / n + 1)

    ncols = int(np.sqrt(ndim))
    nrows = int(np.ceil(ndim / ncols))
    label_str = r'$\sqrt{\tau_\mathrm{int} / n} = $'
    fig, axarr = plt.subplots(
        nrows,
        ncols,
        sharex="col",
        sharey="row",
        figsize=(4.8 * ncols, 2.4 * nrows))
    # fig.tight_layout()
    for i in range(ndim):
        col = i % ncols
        row = int(np.floor((i - col) / ncols))
        unc = np.sqrt(acorr_times[i] / nsamples)
        axarr[row][col].plot(steps, acorr[::n, i], alpha=0.5)
        axarr[row][col].annotate(
            labels[i],
            xy=(0.1, 0.7),
            xycoords="axes fraction",
            bbox={"fc": "w",
                  "ec": "k",
                  "pad": 4.0,
                  "alpha": 0.5})
        axarr[row][col].annotate(
            label_str + '{:.1e}'.format(unc),
            xy=(0.5, 0.7),
            xycoords="axes fraction",
            fontsize=10,
            bbox={"fc": "w",
                  "ec": "k",
                  "pad": 4.0,
                  "alpha": 0.5})
    for col in range(ncols):
        axarr[-1][col].set_xlabel('Iterations')
    return fig, axarr


def corner_plot(outfile, burn_fraction=0.5, **kwargs):
    """Make an enormous corner plot, rejecting the burn-in iterations at the 
    start.

    You should at very least visually examine the walker plots to determine
    how much of the chain should be rejected.

    Parameters
    ----------
    outfile : str
        hdf5 file name
    burn_fraction : float, optional
        fraction of the chain to reject

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    chain = io.read_dataset(outfile, "chain")
    model = io.read_model(outfile)
    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    labels = []
    for i, name in enumerate(model.params.names):
        if name in label_map:
            labels.append(label_map[name])
        else:
            labels.append(name)
    keep = round(niterations * burn_fraction)
    samples = chain[:, keep:, :].reshape((-1, ndim))
    kwargs.update(corner_kwargs)
    kwargs.update(labels=labels)
    fig = corner(samples, **kwargs)
    return fig


def component_plot(outfile,
                   burn_fraction=0.75,
                   rmin=1,
                   rmax=1000,
                   nsamples=10000,
                   size=50,
                   **fig_kwargs):
    """Plot mass components and total mass.
    
    Parameters
    ----------
    outfile : str
        path to hdf5 file
    burn_fraction : float 
        in (0, 1), fraction of chain to discard
    rmin : float, optional
        arcsec, minimum radius
    rmax : float, optional
        arcsec, maximum radius
    nsamples : int, optional
        number of subsamples of chain
    size : int, optional
        number of radial points on grid
    fig_kwargs
        keywords passed to pyplot.subplots
    
     Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple
        2-tuple of matplotlib.axes._subplots.AxesSubplot instances
    """
    chain = io.read_dataset(outfile, "chain")
    model = io.read_model(outfile)
    mass_model = model.mass_model
    nwalkers, niterations, ndim = chain.shape
    assert ndim == len(model.params)
    keep = round(niterations * burn_fraction)
    samples = chain[:, keep:, :].reshape((-1, ndim))

    # construct profiles for each mass component
    idx = np.random.choice(np.arange(samples.shape[0]), nsamples)
    radii = np.logspace(np.log10(rmin), np.log10(rmax), size)
    profiles = np.zeros((len(mass_model), nsamples, size))
    distances = np.zeros(nsamples)
    for i, param_values in enumerate(samples[idx]):
        param_map = model.params.mapping(param_values)
        kwargs = {**param_map, **model.constants}
        distances[i] = kwargs['dist']
        for j, (name, mass_component) in enumerate(mass_model.items()):
            profiles[j][i] = mass_component(radii, **kwargs)

    color = cycle(["C" + str(i) for i in range(6)])
    style = cycle(["--", "-.", ":"])
    label_map = {"dm": "DM", "st": "Stars", "bh": "BH"}

    fig, (ax0, ax1) = plt.subplots(2, sharex=True, **fig_kwargs)

    for i, (name, mass_component) in enumerate(mass_model.items()):
        M_low, M_med, M_high = np.percentile(profiles[i], [16, 50, 84], axis=0)
        c = next(color)
        s = next(style)
        try:
            label = label_map[name]
        except KeyError:
            label = name
        kpc_per_arcsec = distances[i] * radians_per_arcsec
        kpc = kpc_per_arcsec * radii
        vc_med = np.sqrt(G * M_med / kpc)
        vc_low = np.sqrt(G * M_low / kpc)
        vc_high = np.sqrt(G * M_high / kpc)
        ax0.plot(radii, vc_med, c + s, label=label)
        ax0.fill_between(radii, vc_low, vc_high, facecolor=c, alpha=0.3)
        ax1.plot(radii, M_med, c + s)
        ax1.fill_between(radii, M_low, M_high, facecolor=c, alpha=0.3)
    # total mass profile
    M_low, M_med, M_high = np.percentile(
        np.sum(profiles, axis=0), q=[16, 50, 84], axis=0)
    kpc_per_arcsec = np.median(distances) * radians_per_arcsec
    kpc = kpc_per_arcsec * radii
    vc_med = np.sqrt(G * M_med / kpc)
    vc_low = np.sqrt(G * M_low / kpc)
    vc_high = np.sqrt(G * M_high / kpc)
    ax0.plot(radii, vc_med, 'k-', label="Total")
    ax0.fill_between(radii, vc_low, vc_high, facecolor='k', alpha=0.3)
    ax1.plot(radii, M_med, 'k-')
    ax1.fill_between(radii, M_low, M_high, facecolor='k', alpha=0.3)

    ax0.legend(loc="best")
    ax0.set_xlim(radii.min(), radii.max())
    ax0.set_xscale('log')
    ax0.set_ylabel(r'$v_\mathrm{circ}$  [km s$^{-1}$]')
    ax1.set_xlim(radii.min(), radii.max())
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$M(<R)$  [M$_\odot$]')
    ax1.set_xlabel('R  [arcsec]')
    return fig, (ax0, ax1)

