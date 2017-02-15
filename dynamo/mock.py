"""Mock data set generation.  Fake it till ya make it."""

import os
import numpy as np

from .surface_density import I_sersic
from .volume_density import L_sersic_tot

def mock(model, outdir, prefix):
    """Create a mock data set"""

    mass_model = model.mass_model
    kwargs = model.construct_kwargs(model.params._values)
    saveto = os.path.join(outdir, prefix)
    
    for mm in model.measurements:
        radii = mm.radii
        if mm.likelihood.__name__ == "lnlike_continuous":
            dsigma = mm.observables["dsigma"]
            sigma = mm.tracers[0](radii, mass_model, kwargs)
            sigma += dsigma * np.random.randn(dsigma.size)
            x = np.array([radii, sigma, dsigma]).T
            np.savetxt(saveto + "_" + mm.tracers[0].name + ".txt",
                       x, header="R sigma dsigma")
        elif mm.likelihood.__name__ == "lnlike_discrete":
            dv = mm.observables['dv']
            sigma = mm.tracers[0](radii, mass_model, kwargs)
            variances = dv ** 2 + sigma ** 2
            v = np.sqrt(variances) * np.random.randn(dv.size)
            x = np.array([radii, v, dv]).T
            np.savetxt(saveto + "_" + mm.tracers[0].name + ".txt",
                       x, header="R v dv")
        elif mm.likelihood.__name__ == "lnlike_gmm":
            dv = mm.observables['dv']
            dc = mm.observables['dc']
            phi_b = kwargs['phi_b']
            mu_color_b = kwargs['mu_color_b']
            mu_color_r = kwargs['mu_color_r']            
            sigma_color_b = kwargs['sigma_color_b']
            sigma_color_r = kwargs['sigma_color_r']
            I0_b = kwargs['I0_b']
            I0_r = kwargs['I0_r']
            Re_b = kwargs['Re_b']
            Re_r = kwargs['Re_r']
            n_b = kwargs['n_b']
            n_r = kwargs['n_r']
            sigma_b = np.sqrt(mm.tracers[0](radii, mass_model, kwargs) ** 2 +
                              (dv * np.random.randn(dv.size)) ** 2)
            sigma_r = np.sqrt(mm.tracers[1](radii, mass_model, kwargs) ** 2 +
                              (dv * np.random.randn(dv.size)) ** 2)            
            v = np.zeros(dv.shape)
            c = np.zeros(dc.shape)
            for i, R in enumerate(radii):
                # Bayes to the rescue!
                # p(blue | R) propto p(R | blue) * p(blue)
                p_blue = phi_b * (2 * np.pi * R * I_sersic(R, I0_b, Re_b, n_b)
                                  / L_sersic_tot(I0_b, Re_b, n_b))
                if np.random.rand() < p_blue:
                    v[i] = sigma_b[i] * np.random.randn()
                    c[i] = np.sqrt(sigma_color_b ** 2 + dc[i] ** 2) * \
                           np.random.randn() + mu_color_b
                else:
                    # in reds
                    v[i] = sigma_r[i] * np.random.randn()
                    c[i] = np.sqrt(sigma_color_r ** 2 + dc[i] ** 2) * \
                           np.random.randn() + mu_color_r
            x = np.array([radii, v, dv, c, dc]).T
            np.savetxt(saveto + "_gmm.txt", x, header="R v dv c dc") 
        else:
            raise ValueError(mm.likelihood.__name__ + " not recognized.")
        
