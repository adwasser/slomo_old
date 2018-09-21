"""
Subclasses of colossus.halo.HaloDensityProfile
"""

from scipy import stats, special, optimize, integrate

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
    epsilon: float
        The matching parameter, equal to the ratio :math:`\rho_{\rm sol} / \rho(r_\epsilon)` 
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
    ma: float
        mass of ultra-light axion
    """
    
    def __init__(self, rhos=None, rs=None, epsilon=None, rsol=None, 
                 M=None, c=None, z=None, mdef=None, ma=None, **kwargs):
        self.par_names = ['rhos', 'rs', 'epsilon', 'rsol']
        self.opt_names = ['repsilon', 'rhosol']
        HaloDensityProfile.__init__(self, **kwargs)
        
        if rhos is not None and rs is not None:
            self.par['rhos'] = rhos
            self.par['rs'] = rs
        elif M is not None and c is not None and mdef is not None and z is not None:
            self.par['rhos'], self.par['rs'] = NFWProfile.fundamentalParameters(M, c, z, mdef)      
        else:
            msg = ('An NFW profile must be defined either using rhos and rs, or M, '
                   'c, mdef, and z.')
            raise ValueError(msg)

        if epsilon is not None and rsol is not None:
            self.par['epsilon'] = epsilon
            self.par['rsol'] = rsol
            rhosol, repsilon = self._get_opts(epsilon, rsol)
            self.opt['rhosol'] = rhosol
            self.opt['repsilon'] = repsilon
        elif epsilon is not None and ma is not None and z is not None:
            self.par['epsilon'] = epsilon
            self._set_pars_from_ma(ma, z=z)
        else:
            msg = ('A soliton profile must be defined using either epsilon and'
                   ' rsol or epsilon, ma, and z.')
            raise ValueError(msg)
        
        
    def _get_opts(self, epsilon, rsol):
        """Calculate the soliton scale density from the matching paramter and
        scale radius.
        
        Parameters
        ----------
        epsilon: float
            The matching parameter, equal to the ratio :math:`\rho_{\rm sol} / \rho(r_\epsilon)` 
        rsol: float
            The soliton scale radius in physical kpc/h.
        
        Returns
        -------
        rhosol: float
        repsilon: float
        """
        rs = self.par['rs']
        rhos = self.par['rhos']
        repsilon = rsol * (epsilon**(-0.125) - 1)**0.5
        rhoepsilon = NFWProfile.rho(rhos, repsilon / rs)
        return rhoepsilon / epsilon, repsilon
    
    
    def _set_pars_from_ma(self, ma, z):
        """Sets the correct soliton scale parameters from the axion mass."""
        rhos = self.par['rhos']
        rs = self.par['rs']
        epsilon = self.par['epsilon']
        f = lambda rsol: (ma - self._ma_from_sol(self._get_opts(epsilon, rsol)[0], 
                                                 rsol, z))**2
        rsol = optimize.fminbound(f, 1e-3, 1e3)
        rhosol, repsilon = self._get_opts(epsilon, rsol)
        self.par['rsol'] = rsol
        self.opt['rhosol'] = rhosol
        self.opt['repsilon'] = repsilon
        
    
    def _ma_from_sol(self, rhosol, rsol, z, alpha=0.23):
        """Calculate the axion mass from the soliton scale parameters.
        See Marsh & Pop 2015, equation 8.
        Axion mass has units of 1e-22 eV"""
        # convert from h-scaled units to "real" units
        rsol = rsol / cosmo.h
        # rhosol is normalized by rho_c
        rho_c = cosmo.rho_c(z)
        delta_sol = rhosol / rho_c
        ma = np.sqrt(delta_sol**-1 * rsol**-4 * (cosmo.h / 0.7)**2 * 5e4 * alpha**-4)
        return ma
    
    
    def rho_sol(self, rhosol, x):
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
        rhosol = self.opt['rhosol']
        rsol = self.par['rsol']
        repsilon = self.opt['repsilon']
        
        rho_soliton = self.rho_sol(rhosol, r / rsol)
        rho_nfw = NFWProfile.rho(rhos, r / rs)
        return np.where(r < repsilon, rho_soliton, rho_nfw)

    def get_ma(self, z):
        return self._ma_from_sol(self.opt['rhosol'], self.par['rsol'], z)
