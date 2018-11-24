"""
Subclasses of colossus.halo.HaloDensityProfile
"""
import numpy as np
from scipy import optimize

from colossus.cosmology import cosmology
from colossus.cosmology.cosmology import setCosmology, getCurrent
from colossus.halo.concentration import concentration
from colossus.halo.profile_base import HaloDensityProfile
from colossus.halo.profile_nfw import NFWProfile
from colossus.halo import mass_so
cosmo = setCosmology('planck15')


class SolitonNFWProfile(HaloDensityProfile):
    """
    The SolitonNFWProfile consists of an inner soliton core, described by a scale 
    density and radius, transitioning to an NFW profile with its own scale density
    and radius.
    
    The NFW profile region can be described either with the fundamental parameters
    ``rhos`` and ``rs``, or by specifying a halo mass ``M``, concentration ``c``, 
    and virial mass definition, ``mdef``, along with a redshift, ``z``.
    
    The soliton region has the fundamental parameters `rhosol` and `rsol`.  The 
    transition radius, ``repsilon``, where the profile transitions from the soliton
    model to the NFW model is calculated on instantiation by requiring continuity,
    and it is stored as self.opt['repsilon'].
    
    Parameters
    ----------
    rhos: float
        The NFW scale density in physical :math:`M_{\odot} h^2 / {\rm kpc}^3`.
    rs: float
        The NFW scale radius in physical kpc/h.
    rhosol: float
        The soliton scale density in physical :math:`M_{\odot} h^2 / {\rm kpc}^3`.
    rsol: float
        The soliton scale radius in physical kpc/h.    
    M: float
        A spherical overdensity mass in :math:`M_{\odot}/h` corresponding to the mass
        definition ``mdef`` at redshift ``z``. 
    c: float
        The concentration, :math:`c = R / r_{\rm s}`, corresponding to the given halo mass and
        mass definition.
    z: float
        Redshift
    mdef: str
        The mass definition in which ``M`` and ``c`` are given. See :doc:`halo_mass` for details.
    m22: float
        mass of ultra-light axion in 1e-22 eV
    solition_scaling: bool, optional
        If true, use the core mass-halo mass scaling from Schive+2014/Robles+2018
    """
    
    def __init__(self, rhos=None, rs=None, rhosol=None, rsol=None, 
                 M=None, c=None, z=None, mdef=None, m22=None,
                 soliton_scaling=True, **kwargs):
        self.par_names = ['rhos', 'rs', 'rhosol', 'rsol']
        self.opt_names = ['repsilon']
        HaloDensityProfile.__init__(self, **kwargs)
        
        if rhos is not None and rs is not None:
            self.par['rhos'] = rhos
            self.par['rs'] = rs
        elif M is not None and c is not None and mdef is not None and z is not None:
            rhos, rs = NFWProfile.fundamentalParameters(M, c, z, mdef)
            self.par['rhos'] = rhos
            self.par['rs'] = rs
        else:
            msg = ('An NFW profile must be defined either using rhos and rs, or M, '
                   'c, mdef, and z.')
            raise ValueError(msg)

        if rhosol is not None and rsol is not None:
            self.par['rhosol'] = rhosol
            self.par['rsol'] = rsol
            self.opt['repsilon'] = self._matching_radius()
        elif rsol is not None and m22 is not None:
            self.par['rsol'] = rsol
            self.par['rhosol'] = SolitonNFWProfile._rhosol_from_m22(m22, rsol)
            self.opt['repsilon'] = self._matching_radius()
        elif m22 is not None and soliton_scaling and z is not None:
            rhosol, rsol = SolitonNFWProfile.fundamentalParameters(rhos, rs, z, m22,
                                                                   soliton_scaling=soliton_scaling)
            self.par['rhosol'] = rhosol
            self.par['rsol'] = rsol
            self.opt['repsilon'] = self._matching_radius()
        else:
            msg = ('A soliton profile must be defined using either rhosol and'
                   ' rsol, rsol and m22, or m22 and z.')
            raise ValueError(msg)
        

    @classmethod
    def fundamentalParameters(cls, rhos, rs, z, m22, soliton_scaling=True):
        """Calculate the scale density/radius of the soliton core.

        Parameters
        ----------
        rhos: float
            The NFW scale density in physical :math:`M_{\odot} h^2 / {\rm kpc}^3`.
        rs: float
            The NFW scale radius in physical kpc/h.
		z: float
			Redshift
        m22: float
            axion mass in 1e-22 eV
        solition_scaling: bool, optional
            If true, use the core mass-halo mass scaling from Schive+2014/Robles+2018

        Returns
        -------
        rhosol: float
            The soliton scale density in physical :math:`M_{\odot} h^2 / {\rm kpc}^3`.
        rsol: float
            The soliton scale radius in physical kpc/h.    
        """
        nfw_profile = NFWProfile(rhos=rhos, rs=rs)
        # convert virial mass to non-h-scaled
        Mvir = nfw_profile.MDelta(z, mdef='vir') / cosmo.h
        
        # relations with non-h-scaled units, need to convert later to h-scaled
        rsol = 3.315 * 1.6 * (Mvir / 1e9)**(-1/3.) * m22**-1
        alpha = 0.230 # from Marsh & Pop 2015
        rhoc = cosmo.rho_c(z) * cosmo.h**2
        rhosol = rhoc * (cosmo.h / 0.7)**-2 * (5e4 / alpha**4) * rsol**-4 * m22**-2

        # give back h-scalings
        rsol = rsol * cosmo.h
        rhosol = rhosol / cosmo.h**2
        return rhosol, rsol

    
    @staticmethod
    def rho_sol(rhosol, x):
        """Calculate the density profile as a function of r / rsol.
        
        Parameters
        ----------
        rhosol: float
            The soliton scale density in physical :math:`M_{\odot} h^2 / {\rm kpc}^3`.
        x: float or array_like
            radius as a fraction of the soliton scale radius
        
        Returns
        -------
        density: array_like
            Density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`.
            Has the same dimensions as ``x``. 
        """
        return rhosol * (1 + x**2)**(-8)

    
    def _matching_radius(self):
        rhos = self.par['rhos']
        rs = self.par['rs']
        rhosol = self.par['rhosol']
        rsol = self.par['rsol']
        f_nfw = lambda r: NFWProfile.rho(rhos, r / rs)
        f_sol = lambda r: SolitonNFWProfile.rho_sol(rhosol, r / rsol)
        metric = lambda r: ((f_sol(r) - f_nfw(r)) / f_sol(r))**2
        repsilon = optimize.fminbound(metric, 1e-3, 1e3, xtol=1e-9)
        return repsilon

    @staticmethod
    def _m22_from_sol(rhosol, rsol, z, alpha=0.23):
        """Calculate the axion mass from the soliton scale parameters.
        See Marsh & Pop 2015, equation 8.
        Axion mass has units of 1e-22 eV"""
        # convert from h-scaled units to "real" units
        rsol = rsol / cosmo.h
        # rhosol is normalized by rho_c
        delta_sol = rhosol / cosmo.rho_c(z)
        m22 = np.sqrt(delta_sol**-1 * rsol**-4 * (cosmo.h / 0.7)**2 * 5e4 * alpha**-4)
        return m22
    
    
    @classmethod
    def _rhosol_from_m22(cls, m22, rsol, z=0):
        """Calculate the scale density from m22 and the scale radius"""
        alpha = 0.230
        # convert rsol to physical kpc
        rsol = rsol / cosmo.h
        delta_sol = (5.0e4 / alpha**4) * (cosmo.h / 0.7)**-2 * m22**-2 * rsol**-4
        return delta_sol * cosmo.rho_c(z=z)

    
    def densityInner(self, r):
        """
        Parameters
        ----------
        r: array_like
            Radius in physical kpc/h; can be a number or a numpy array.
            
        Returns
        -------
        density: array_like
            Density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`.
            Has the same dimensions as ``r``.        
        """
        rhos = self.par['rhos']
        rs = self.par['rs']
        rhosol = self.par['rhosol']
        rsol = self.par['rsol']
        repsilon = self.opt['repsilon']
        
        rho_soliton = SolitonNFWProfile.rho_sol(rhosol, r / rsol)
        rho_nfw = NFWProfile.rho(rhos, r / rs)
        return np.where(r < repsilon, rho_soliton, rho_nfw)

    def get_m22(self, z=0):
        return _m22_from_sol(self.par['rhosol'], self.par['rsol'], z)
