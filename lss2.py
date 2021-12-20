#!/usr/bin/python3
# file   : lss2.py 
# author : ms3

r"""

Large-Scale Structure
=====================

A light-weight module for computations related to large scale structure formation in cosmology. This contains

1. Transfer function fits (:class:`Transfer`).
2. Mass-function fits (:class:`MassFunction`).
3. Large scale linear bias functions (:class:`Bias`)
4. An object for computations related to large-scale structure formation, :class:`CosmoStructure`.

This module is intented to be a *single file module*, for almost all computations related to the large-scale structure formation in the universe. The main tool is the :class:`CosmoStructure` object. It can be used for computing the linear power spectra, variance or the halo mass function under some specified cosmology.

.. version:: 2.0

"""

__version__ = '2.0'

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from collections import namedtuple

# ============================== Useful constants ==================================
RHO_CRIT0 = 1.878347245530161e-26 # critical density in h^2 kg / m^3
DELTA_C   = 1.6864701998411453    # critical density for spherical collapse
SIGMA_SB  = 5.670374419184E-08    # stefan's constant in kg/sec^3/K^4
C_SI      = 299792458.0           # speed of light in m/sec
G_SI      = 6.67428e-11           # newtonian gravitational constant
RHO_CRIT0_ASTRO = 2.77536627E+11  # critical density in h^2 Msun / Mpc^3

# ========================== Units and conversions =================================
class Units:
    r""" Some useful units """
    au   = 1.49597870700e+11     # 1 astronomical unit (au)
    pc   = 3.085677581491367e+16 # 1 parsec (pc)
    kpc  = 3.085677581491367e+19 # 1 kpc
    Mpc  = 3.085677581491367e+22 # 1 Mpc
    Msun = 1.98842e+30           # mass of sun in kg 
    yr   = 31558149.8            # year (sidereal) in sec

# ======================= Transfer function definitions ============================
class Transfer:
    r""" 
    Definitions of various transfer functions. Available functions are listed in the `available` dictionary. Use its keys to select a specific form. Alternatively, use the `.` operator to access the specific function, like `Transfer.modelEisenstein98` etc. For example, 

    
    >>> Transfer.available.keys() # available keys
    dict_keys(['eisenstein98', 'eisenstein98_zb', 'eisenstein98_mdm', 'sugiyama95'])
    >>> f = Transfer.available['eisenstein98'].f # select a function
    >>> callable(f)
    True
    >>> f == Transfer.modelEisenstein98
    True

    Notes
    -----
    Transfer functions available are Eisenstein-Hu with and without baryon oscillations [1]_, with mixed dark-matter [2]_ and the BBKS with Sugiyama correction [3]_.

    References
    ----------
    .. [1] Daniel J. Eisenstein and Wayne Hu. Baryonic Features in the Matter Transfer Function, `arXive:astro-ph/9709112v1, <http://arXiv.org/abs/astro-ph/9709112v1>`_, 1997.
    .. [2] Daniel J. Eisenstein and Wayne Hu. Power Spectra for Cold Dark Matter and its Variants, `arXive:astro-ph/9710252v1, <http://arXiv.org/abs/astro-ph/9710252v1>`_, 1997.
    .. [3] A. Meiksin, matrin White and J. A. Peacock. Baryonic signatures in large-scale structure, Mon. Not. R. Astron. Soc. 304, 851-864, 1999.
    """
    Data      = namedtuple("Data", ("f", "z_dep", "model"), defaults = (False, "cdm"))
    available = {}

    def modelEisenstein98(k: float, h: float, Om0: float, Ob0: float, Tcmb0: float) -> float:
        r""" 
        Matter transfer function given by Eisentein and Hu (1998), with baryon oscillations.

        Parameters
        ----------
        k:  array_like
            Wavenumbers in Mpc/h
        h: float
            Hubble parameter in 100 km/sec/Mpc at present
        Om0: float
            Normalized matter density at present
        Ob0: float
            Normalized baryon density at present
        Tcmb0: float
            Present CMB temperature in Kelvin

        Returns
        -------
        T: array_like
            Value of transfer function. Has the same shape as `k`

        Examples
        --------
        
        >>> Transfer.modelEisenstein98(k = [1.e-4, 1., 1.e+4], Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, h = 0.7)
        array([9.99906939e-01, 4.59061096e-03, 2.18246719e-10])

        """
        k     = np.asarray(k) * h # convert wavenumber from h/Mpc to 1/Mpc

        # setting cosmological parameters
        h2          = h * h
        Omh2, Obh2  = Om0 * h2, Ob0 * h2
        theta       = Tcmb0 / 2.7 # cmb temperature in units of 2.7 K

        wt_b  = Ob0 / Om0 # fraction of baryons
        wt_c  = 1 - wt_b  # fraction of cold dark matter
        
        # redshift at equality : eqn. 2 (corrected)
        zp1_eq = (2.50e+04)*Omh2 / theta**4

        # wavenumber at equality : eqn. 3
        k_eq = (7.46e-02)*Omh2 / theta**2

        # redshift at drag epoch : eqn 4
        c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
        c2  = 0.238*Omh2**0.223
        z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

        # baryon - photon momentum density ratio : eqn. 5
        R_const = 31.5*(Obh2 / theta**4) * 1000
        R_eq    = R_const / zp1_eq     # ... at equality epoch
        R_d     = R_const / (1 + z_d)  # ... at drag epoch

        # sound horizon : eqn. 6
        s = (2/3/k_eq)*np.sqrt(6/R_eq)*np.log((np.sqrt(1 + R_d) + np.sqrt(R_eq + R_d)) / (1 + np.sqrt(R_eq)))

        # silk scale : eqn. 7
        k_silk = 1.6*(Obh2**0.52)*(Omh2**0.73)*(1 + (10.4*Omh2)**(-0.95))
        
        q = k/(13.41*k_eq)  # eqn. 10
        x = k*s             # new variable

        # eqn. 11
        a1      = (1 + (32.1*Omh2)**(-0.532))*(46.9*Omh2)**0.670
        a2      = (1 + (45.0*Omh2)**(-0.582))*(12.0*Omh2)**0.424
        alpha_c = (a1**(-wt_b)) * (a2**(-wt_b**3))

        # eqn. 12
        b1      = 0.944 / (1 + (458.0*Omh2)**(-0.708))
        b2      = (0.395*Omh2)**(-0.0266)
        beta_c  = 1 / (1 + b1*(wt_c**b2 - 1))

        # eqn. 18
        f = 1 / (1 + (x/5.4)**4)

        # eqn. 19 and 20
        l_beta     = np.log(np.e + 1.8*beta_c*q)

        c_no_alpha = 14.2           + 386.0 / (1 + 69.9*q**1.08)
        t_no_alpha = l_beta / (l_beta + c_no_alpha*q**2)

        c_alpha    = 14.2 / alpha_c + 386.0 / (1 + 69.9*q**1.08)
        t_alpha    = l_beta / (l_beta + c_alpha*q**2)

        # cold-dark matter part : eqn. 17
        tc = f*t_no_alpha + (1 - f)*t_alpha

        # eqn. 15
        y   = zp1_eq / (1 + z_d)
        y1  = np.sqrt(1 + y)
        Gy  = y*( -6*y1 + (2 + 3*y) * np.log((y1 + 1) / (y1 - 1)) )

        # eqn. 14
        alpha_b = 2.07*(k_eq*s)*Gy*(1 + R_d)**(-3/4)

        # eqn. 24
        beta_b  = 0.5 + wt_b + (3 - 2*wt_b)*np.sqrt((17.2*Omh2)**2 + 1)

        # eqn. 23
        beta_node = 8.41*Omh2**0.435

        # eqn. 22
        s_tilde   = s / (1 + (beta_node / x)**3)**(1/3)
        x_tilde   = k*s_tilde

        # eqn. 19 and 20 again
        l_no_beta = np.log(np.e + 1.8*q)
        t_nothing = l_no_beta / (l_no_beta + c_no_alpha*q**2)

        # baryonic part : eqn. 21
        j0 = np.sin(x_tilde) / x_tilde # zero order spherical bessel
        tb = (t_nothing / (1 + (x / 5.2)**2) + ( alpha_b / (1 + (beta_b / x)**3) ) * np.exp(-(k / k_silk)**1.4)) * j0
    
        return wt_b * tb + wt_c * tc # full transfer function : eqn. 16

    def modelEisenstein98_zeroBaryon(k: float, h: float, Om0: float, Ob0: float, Tcmb0: float):
        r""" 
        Matter transfer function given by Eisentein and Hu (1998), with without baryon oscillations.

        Parameters
        ----------
        k:  array_like
            Wavenumbers in Mpc/h
        h: float
            Hubble parameter in 100 km/sec/Mpc at present
        Om0: float
            Normalized matter density at present
        Ob0: float
            Normalized baryon density at present
        Tcmb0: float
            Present CMB temperature in Kelvin

        Returns
        -------
        T: array_like
            Value of transfer function. Has the same shape as `k`

        Examples
        --------
        
        >>> Transfer.modelEisenstein98_zeroBaryon(k = [1.e-4, 1., 1.e+4], Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, h = 0.7)
        array([9.99899458e-01, 4.55592401e-03, 2.14698256e-10])

        """
        k = np.asarray(k)

        # setting cosmological parameters
        h2    = h * h
        Omh2  = Om0 * h2
        Obh2  = Ob0 * h2
        theta = Tcmb0 / 2.7 # cmb temperature in units of 2.7 K
        wt_b  = Ob0 / Om0   # fraction of baryons

        # sound horizon : eqn. 26
        s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))

        # eqn. 31
        alpha_gamma = 1 - 0.328*np.log(431*Omh2)*wt_b + 0.38*np.log(22.3*Omh2)*wt_b**2

        # eqn. 30
        gamma_eff   = Om0*h*(alpha_gamma + (1 - alpha_gamma)/(1 + (0.43*k*s)**4))

        # eqn. 28
        q   = k*(theta*theta/ gamma_eff)

        # eqn. 29
        l_0 = np.log(2*np.e + 1.8*q)
        c_0 = 14.2 + 731.0 / (1 + 62.5*q)
        T_0 = l_0 / (l_0 + c_0*q**2)
        return T_0

    def modelEisenstein98_mixedDarkMatter(k: float, z: float, h: float, Om0: float, Ob0: float, Onu0: float, Ode0: float, Nnu: float, Tcmb0: float, _zisDz: bool = False) -> float:
        r""" 
        Matter transfer function given by Eisentein and Hu (1998), with mixed dark-matter.

        NOTE: **This is not tested.**

        Parameters
        ----------
        k:  array_like
            Wavenumbers in Mpc/h
        z: float
            Redshift or growth at that time, based on the value of `_zisDz` argument.
        h: float
            Hubble parameter in 100 km/sec/Mpc at present
        Om0: float
            Normalized matter density at present
        Ob0: float
            Normalized baryon density at present
        Onu0: float
            Normalized massive neutrino (hot dark-matter) density at present
        Ode0: float
            Normalized dark energy density at present
        Nnu: float
            Number of massive nuetrinos.
        Tcmb0: float
            Present CMB temperature in Kelvin
        _zisDz: bool, optional
            If true, this takes the value of `z` as **growth**, :math:`D(z)`. Otherwise take it as redshift. This is used internaly for using growth factor definitions consistently. It is False by default.

        Returns
        -------
        T: array_like
            Value of transfer function. Has the same shape as `k`

        Examples
        --------
        TODO

        """
        k   = np.asarray(k) * h # convert wavenumber from h/Mpc to 1/Mpc

        # setting cosmological parameters
        h2    = h * h
        Omh2  = Om0  * h2
        Obh2  = Ob0  * h2
        theta = Tcmb0 / 2.7 # cmb temperature in units of 2.7 K

        fb  = Ob0  / Om0    # fraction of baryons
        fnu = Onu0 / Om0    # fraction of nuetrinos (hot dark-matter)
        fc  = 1 - fb - fnu  # fraction of cold dark matter
        fcb = fc  + fb
        fnb = fnu + fb
        
        # redshift at equality : eqn. 1 (corrected)
        zp1_eq = (2.50e+04)*Omh2 / theta**4

        # redshift at drag epoch : eqn 2
        c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
        c2  = 0.238*Omh2**0.223
        z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

        yd  = zp1_eq / (1 + z_d) # eqn 3

        # sound horizon : eqn. 4
        s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))

        q = k * theta**2 / Omh2 # eqn 5

        # growth factor :
        if _zisDz: 
            D1 = zp1_eq * z # normalization D1 = (z_eq + 1) / (z + 1) for early times
        else:
            zp1  = z + 1
            g2z  = Om0 * zp1**3 + (1 - Om0 - Ode0) * zp1**2 + Ode0 # eqn. 9
            Omz  = Om0 * zp1**3 / g2z # eqn. 10
            Odez = Ode0 / g2z
            D1   = (zp1_eq / zp1) * (2.5 * Omz) / (Omz**(4/7) - Odez + (1 + Omz / 2.) * (1 + Odez / 70.))

        # growth factor in presence of free-streaming :
        pc  = 0.25 * (5 - np.sqrt(1 + 24 * fc )) # eqn. 11
        pcb = 0.25 * (5 - np.sqrt(1 + 24 * fcb)) 

        yfs = 17.2 * fnu * (1 + 0.488 / fnu**(7/6)) * (Nnu * q / fnu)**2 # eqn. 14

        __x   = D1 / (1 + yfs)
        Dcb   = (1                + __x**0.7)**(pcb / 0.7) * D1**(1 - pcb) # eqn. 12
        Dcbnu = (fcb**(0.7 / pcb) + __x**0.7)**(pcb / 0.7) * D1**(1 - pcb) # eqn. 13

        # small-scale suppression : eqn. 15
        alpha  = (fc / fcb) * (5 - 2 *(pc + pcb)) / (5 - 4 * pcb)
        alpha *= (1 - 0.533 * fnb + 0.126 * fnb**3) / (1 - 0.193 * np.sqrt(fnu * Nnu) + 0.169 * fnu * Nnu**0.2)
        alpha *= (1 + yd)**(pcb - pc)
        alpha *= (1 + 0.5 * (pc - pcb) * (1 + 1 / (3 - 4 * pc) / (7 - 4 * pcb)) / (1 + yd))

        Gamma_eff = Omh2 * (np.sqrt(alpha) + (1 - np.sqrt(alpha)) / (1 + (0.43 * k * s)**4)) # eqn. 16
        qeff      = k * theta**2 / Gamma_eff

        # transfer function T_sup :
        beta_c = (1 - 0.949 * fnb)**(-1) # eqn. 21
        L      = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha) * qeff) # eqn. 19
        C      = 14.4 + 325 / (1 + 60.5 * qeff**1.08) # eqn. 20
        Tk_sup = L / (L + C * qeff**2) # eqn. 18

        # master function :
        qnu       = 3.92 * q * np.sqrt(Nnu / fnu) # eqn. 23
        Bk        = 1 + (1.24 * fnu**0.64 * Nnu**(0.3 + 0.6 * fnu)) / (qnu**(-1.6) + qnu**0.8) # eqn. 22
        Tk_master = Tk_sup * Bk # eqn. 24  

        # transfer function - cdm + baryon : eqn. 6
        Tkcb = Tk_master * Dcb / D1

        # # transfer function - cdm + baryon + neutrino : eqn. 7
        # Tkcbnu = Tk_master * Dcbnu / D1

        # # transfer function - neutrino : eqn. 26
        # Tknu = (Tkcbnu - fcb * Tkcb) / fnu
        return Tkcb

    def modelSugiyama95(k, h: float, Om0: float, Ob0: float, Tcmb0: float = None) -> float:
        r""" 
        Matter transfer function given by Bardeen et al.(1986), with correction given by Sugiyama(1995). 

        Parameters
        ----------
        k:  array_like
            Wavenumbers in Mpc/h
        h: float
            Hubble parameter in 100 km/sec/Mpc at present
        Om0: float
            Normalized matter density at present
        Ob0: float
            Normalized baryon density at present
        Tcmb0: float
            Present CMB temperature in Kelvin (not used)

        Returns
        -------
        T: array_like
            Value of transfer function. Has the same shape as `k`

        Examples
        --------
        
        >>> Transfer.modelSugiyama95(k = [1.e-4, 1., 1.e+4], Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, h = 0.7)
        array([9.98671614e-01, 4.65026996e-03, 2.03316613e-10])

        """
        k = np.asarray(k)

        # eqn. 4
        gamma = Om0 * h
        q     = k / gamma * np.exp(Ob0 + np.sqrt(2*h) * Ob0 / Om0) 

        # transfer function : eqn. 3
        T = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

        # for smaller `k`, transfer function returns `0` instead one. fixing it :
        if k.shape == (): # scalar `k`
            if k < 1.0e-09:
                T = 1.0
        else: # array `k`
            T[k < 1.0e-09] = 1.0
        return T

    available["eisenstein98"]     = Data(f = modelEisenstein98)
    available["eisenstein98_zb"]  = Data(f = modelEisenstein98_zeroBaryon)
    available["eisenstein98_mdm"] = Data(f = modelEisenstein98_mixedDarkMatter, z_dep = True, model = "mdm")
    available["sugiyama95"]       = Data(f = modelSugiyama95)

# ======================== Fitting function definitions ============================
class MassFunction:
    r""" 
    Definitions of various halo-mass function fits. Available functions are listed in the `available` dictionary. Use its keys to select a specific form. Alternatively, use the `dot` operator to access the specific function, like `MassFunction.modelTinker08` etc. For example, 

    
    >>> MassFunction.available.keys() # available keys
    dict_keys(['press74', 'sheth01', 'tinker08'])
    >>> f = MassFunction.available['tinker08'].f # select a function
    >>> callable(f)
    True
    >>> f == MassFunction.modelTinker08
    True

    Notes
    -----
    Available massfunction models are the Press-Schechter [1]_, Sheth et al [2]_ and Tinker et al [3]_.

    References
    ----------
    .. [1] Houjun Mo, Frank van den Bosch, Simon White. Galaxy Formation and Evolution, Cambridge University Press, 2010.
    .. [2] Zarija Lukic et al. The Halo Mass Function: High-Redshift Evolution and Universality, `arXive:astro-ph/0702360v2, <http://arXiv.org/abs/astro-ph/0702360v2>`_, 14 January 2008.
    .. [3] Jeremy Tinker et. al. Toward a halo mass function for precision cosmology: The limits of universality, `arXive:astro-ph/0803.2706v1, <http://arXiv.org/abs/0803.2706v1>`_, 2008
    
    """
    Data      = namedtuple("Data", ("f", "mdef", "z_dep"), defaults = (False,))
    available = {}

    def modelPress74(sigma: float):
        r"""
        Fitting function found by Press and Schechter in 1974. 

        Parameters
        ----------
        sigma : array_like
            Mass variance

        Returns
        -------
        f : array_like
            Value of the fitting function

        Examples
        --------
        
        >>> MassFunction.modelPress74([0.7, 0.8, 0.9, 1.0])
        array([0.10553581, 0.18231359, 0.25834461, 0.32457309])
        
        """
        nu = DELTA_C / np.asarray(sigma)
        f  = np.sqrt(2 / np.pi) * nu * np.exp(-0.5 * nu**2)
        return f

    def modelSheth01(sigma: float):
        r"""
        Fitting function found by Sheth et. al. in 2001.

        Parameters
        ----------
        sigma : array_like
            Mass variance

        Returns
        -------
        f : array_like
            Value of the fitting function

        Examples
        --------
        
        >>> MassFunction.modelSheth01([0.7, 0.8, 0.9, 1.0])
        array([0.1107284 , 0.16189185, 0.2061881 , 0.24155146])
        
        """
        A = 0.3222
        a = 0.707
        p = 0.3
        nu = DELTA_C / np.asarray(sigma)
        f = A * np.sqrt(2*a / np.pi) * nu * np.exp(-0.5 * a * nu**2) * (1 + (1/a/nu**2)**p)
        return f

    def modelTinker08(sigma: float, Delta : float, z : float):
        r""" 
        Fitting function found by J. Tinker et. al. in 2008[1]. 

        Parameters
        ----------
        sigma: array_like
            Sigma values (square root of variance)
        Delta: float
            Overdensity with respect to mean density of the background.
        z: float
            redshift

        Returns
        -------
        f: array_like
            Values of fit. Has the same size as `sigma`.

        Examples
        --------
        
        >>> MassFunction.modelTinker08([0.7, 0.8, 0.9, 1.0], 200, 0.)
        array([0.12734265, 0.19005856, 0.24294619, 0.28320827])

        """
        sigma  = np.asarray(sigma)
        if Delta < 200 or Delta > 3200:
            raise ValueError('`Delta` value is out of bound. must be within 200 and 3200.')

        # find interpolated values from 0-redshift parameter table : table 2
        A  = CubicSpline(
                            [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],
                        )(Delta)
        a  = CubicSpline(
                            [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                            [1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66 ],
                        )(Delta)
        b  = CubicSpline(
                            [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                            [2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41 ],
                        )(Delta)
        c  = CubicSpline(
                            [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                            [1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44 ],
                        )(Delta) # `c` is z-independant!
        # print("A = ", A, ", a = ", a, ", b = ", b, ", c = ", c)

        # redshift evolution of parameters : 
        zp1   = 1 + z
        A     = A / zp1**0.14 # equation 5
        a     = a / zp1**0.06 # equation 6  
        alpha = 10.0**(-(0.75 / np.log10(Delta / 75))**1.2) # equation 8    
        b     = b / zp1**alpha # equation 7 
        
        # equation 3
        f = A * (1 + (b / sigma)**a) * np.exp(-c / sigma**2)
        return f

    available["press74"]  = Data(f = modelPress74,  mdef = ['fof'])
    available["sheth01"]  = Data(f = modelSheth01,  mdef = ['fof'])
    available["tinker08"] = Data(f = modelTinker08, mdef = ['so'], z_dep = True)

# ============================ Halo bias definitions ===============================
class Bias:
    r"""
    Definitions of various large scale linear halo bias functions. Available functions are listed in the `available` dictionary. Use its keys to select a specific form. Alternatively, use the `dot` operator to access the specific function, like `Bias.modelTinker10` etc. 

    Examples
    --------
    
    >>> Bias.available.keys() # available keys
    dict_keys(['cole89', 'sheth01', 'tinker10'])
    >>> f = Bias.available['tinker10'].f # select a function
    >>> callable(f)
    True
    >>> f == Bias.modelTinker10
    True

    Notes
    ----- 
    Available models are Tinker et al, Cole et al and Sheth et al (for references, see [1]_ [2]_ and [3]_).

    References
    ----------
    .. [1] Jeremy L. Tinker et al. The Large Scale Bias of Dark Matter Halos: Numerical Calibration and Model Tests. arXive:astro-ph/1001.3162v2, 2010.
    .. [2] H. J. Mo and Y. P. Jing and S. D. M. White. High-order correlations of peaks and haloes: a step towards understanding galaxy biasing. Mon. Not. R. Astron. Soc. 284,189-201, 1997.
    .. [3] Shaun Cole and Nick Kaiser. Biased clustering in the cold dark matter cosmogony, Mon. Not. R. astr. Soc., 237, 1127-1146, 1989.
    

    .. versionadded:: 1.1
    """
    Data      = namedtuple("Data", ("f", "mdef", "z_dep"), defaults = (False,))
    available = {}

    def modelCole89(nu: float):
        r"""
        Bias function given by Cole & Kaiser (1989) and Mo & White (1996).

        Parameters
        ----------
        nu: array_like
            Peak height, defined as :math:`\delta_{\rm c} / \sigma`.

        Returns
        -------
        b: array_like
            Value of bias function.

        Examples
        --------
        
        >>> Bias.modelCole89([1., 1.5, 2., 2.5])
        array([1.        , 1.74119306, 2.77886333, 4.11301083])

        """
        nu = np.asarray(nu)
        return 1. + (nu**2 - 1.) / DELTA_C

    def modelSheth01(nu: float):
        r"""
        Bias function given by Sheth et al. (2001).

        Parameters
        ----------
        nu: array_like
            Peak height, defined as :math:`\delta_{\rm c} / \sigma`.

        Returns
        -------
        b: array_like
            Value of bias function.

        Examples
        --------
        
        >>> Bias.modelSheth01([1., 1.5, 2., 2.5])
        array([1.07578897, 1.66258103, 2.47024337, 3.49037188])

        """
        nu = np.asarray(nu)
        a  = 0.707
        b  = 0.5
        c  = 0.6
        sqrt_a = np.sqrt(a)
        anu2   = a * nu**2
        return 1. + 1. / sqrt_a / DELTA_C * (sqrt_a * anu2 + sqrt_a * b * anu2**(1-c) - anu2**c / (anu2**c + b * (1-c) * (1-c/2.)))

    def modelTinker10(nu: float, Delta: float, z: float):
        r"""
        Bias function given by Tinker et al. (2010).

        Parameters
        ----------
        nu: array_like
            Peak height, defined as :math:`\delta_{\rm c} / \sigma`.
        Delta: float
            Overdensity with respect to mean density of the background.
        z: float
            redshift

        Returns
        -------
        b: array_like
            Value of bias function.

        Examples
        --------
        
        >>> Bias.modelTinker10([1., 1.5, 2., 2.5], 200, 0.)
        array([0.96550128, 1.54189048, 2.41182243, 3.60186066])

        """
        nu = np.asarray(nu)
        y  = np.log10(Delta)
        A  = 1. + 0.24 * y * np.exp(-(4./y)**4)
        a  = 0.44 * y - 0.88
        B  = 0.183
        b  = 1.5
        C  = 0.019 + 0.107 * y + 0.19 * np.exp(-(4./y)**4)
        c  = 2.4
        return 1. - A * nu**a / (nu**a + DELTA_C**a) + B * nu**b + C * nu**c

    available["cole89"]   = Data(f = modelCole89,   mdef = ['fof'],)
    available["sheth01"]  = Data(f = modelSheth01,  mdef = ['fof'],)
    available["tinker10"] = Data(f = modelTinker10, mdef = ['so'],  z_dep = True)

# ========================== `CosmoStructure` Object ===============================
class CosmoStructure:
    r""" 
    An object for storing a radiationless :math:`\Lambda`-MDM or :math:`\Lambda`-CDM cosmology and do calculations related to large-scale structure formation. It can be used for computing power spectrum and halo-mass functions. For the details of equations used, refer any books on cosmology and structure formation (eg., *Galaxy Formation and Evolution* by Mo and White).
    
    Parameters
    ----------    
    flat: bool, optional
        Specify the space geometry. By default, a flat cosmology is used.
    heavy_nu: bool, optional
        Specify if the model has heavy neutrinos (i.e., mixed dark-matter model). Default is True (cold dark-matter only).
    Om0: float
        Normalized matter density at present.
    Ob0: float
        Normalized baryon density at present.
    Ode0: float
        Normalized dark-energy density. It is a **required** parameter for non-flat cosmologies.
    sigma8: float
        Variance at the scale of 8 Mpc/h at present.
    n: float
        Tilt of the power spectrum or spectral index.
    h: float
        Hubble parameter in 100 km/sec/Mpc at present.
    Tcmb0: float, optional
        Present CMB temperature in Kelvin. Default is 2.725 K.
    psmodel: str, optional
        Power spectrum model. Allowed values are `eisenstein98` (for Eisenstein and Hu with BAO, default), `eisenstein98_zb` (EH without BAO) and `sugiyama95` (or `bbks` for BBKS with Sugiyama correction).
    hmfmodel: str, optional
        Halo mass-function model. Allowed values are `press74` (for fit by Press and Schechter, 1974), `sheth01` (Sheth et al, 2001) and `tinker08` (Tinker et al, 2008).
    biasmodel: str, optional
        Linear halo-bias model. Allowed values `cole89` (Cole & Kaiser, 1989), `sheth01` (Sheth et al, 2001) and `tinker10` (Tinker et al, 2010).
    gfmodel: str, optional
        Growth function model. Allowed values are `exact` (default, use the integral form) and `carroll92` (use the fit by Carroll et al, 1992).


    Examples
    --------
    To create a :class:`CosmoStructure` with defaults (flat cosmology with only cold dark-matter),
    
    >>> cs = CosmoStructure(Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, n = 1., h = 0.7)
    >>> cs
    <CosmoStructure flat = True, Om0 = 0.3, Ob0 = 0.05, Ode0 = 0.7, sigma8 = 0.8, ns = 1, h = 0.7, Tcmb0 = 2.73 K, Mnu = 0 eV/c2; power = 'eisenstein98', hmf = 'tinker08', bias = 'tinker10'>

    

    If not specified `cs` in the deocumentation of methods defined here correspond to this object.

    Notes
    -----
    1. All the parameters are keyword arguments. Some has default values, while others are required.

    2. This use :meth:`scipy.integrate.simps` for k space integrations and :meth:`scipy.integrate.quad` for time integration. Integration settings can be updated by calling :meth:`quadOptions` with new values of `n` - number of points, `kmin` and `kmax` - minimum and maximum :math:`k` limits. 

    """

    __slots__ = ('psmodel', 'hmfmodel', 'biasmodel', 'gfmodel', 'flat', 'heavy_nu', 
                 'Om0',     'Ob0',      'Ode0',      'Ok0',     'Onu0', 'Nnu',     
                 'Mnu',     'Tcmb0',    'n',         'sigma8',  'h',    'POWER_NORM',     
                 '_qopt',   '_temp',)

    def __init__(self, *, flat: bool = True, heavy_nu: bool = False, Om0: float = ..., Ob0: float = ..., Ode0: float = ..., sigma8: float = ..., n: float = ..., h: float = ..., Nnu: int = ..., Mnu: float = ..., Tcmb0: float = 2.725, psmodel: str = 'eisenstein98', hmfmodel: str = 'tinker08', biasmodel: str = 'tinker10', gfmodel: str = 'exact'):
        # checking input ...
        psmodel = "sugiyama95" if psmodel == "bbks" else psmodel
        if psmodel not in Transfer.available.keys():
            raise KeyError(f"unsupported power spectrum model {psmodel}")
        self.psmodel    = psmodel

        if hmfmodel not in MassFunction.available.keys():
            raise KeyError(f"unsupported mass-function model {hmfmodel}")
        self.hmfmodel   = hmfmodel
        
        if biasmodel not in Bias.available.keys():
            raise KeyError(f"unsupported bias function model {biasmodel}")
        self.biasmodel  = biasmodel

        if gfmodel not in ['exact', 'carroll92']:
            raise KeyError(f"unsupported growth function model {gfmodel}")
        self.gfmodel = gfmodel

        if Om0 == Ellipsis:
            raise ValueError("`Om0` is a required argument")
        elif Om0 < 0.: 
            raise ValueError("`Om0` must be positive")
        self.Om0 = Om0

        if Ob0 == Ellipsis:
            raise ValueError("`Ob0` is a required argument")
        elif Ob0 < 0. or Ob0 > Om0: 
            raise ValueError("`Ob0` must be positive and less than or equal to `Om0`")
        self.Ob0 = Ob0

        if sigma8 == Ellipsis:
            raise ValueError("`sigma8` is a required argument")
        elif sigma8 < 0.: 
            raise ValueError("`sigma8` must be positive")
        self.sigma8 = sigma8

        if n == Ellipsis:
            raise ValueError("`n` is a required argument")
        self.n = n

        if h == Ellipsis:
            raise ValueError("`h` is a required argument")
        elif h < 0.:
            raise ValueError("`h` must be positive")
        self.h = h
        
        if Tcmb0 < 0.:
            raise ValueError("`Tcmb0` must be positive")
        self.Tcmb0 = Tcmb0

        if heavy_nu:
            if Nnu == Ellipsis:
                raise ValueError("`Nnu` is a required argument if mixed dark-matter is present")
            elif Nnu < 0.:
                raise ValueError("`Nnu` must be positive")
            self.Nnu = Nnu

            if Mnu == Ellipsis:
                raise ValueError("`Mnu` is a required argument if mixed dark-matter is present")
            elif Mnu < 0.:
                raise ValueError("`Mnu` must be positive")
            self.Mnu = Mnu

            Onu0 = Mnu / 91.5 / h**2
            if (Onu0 + self.Ob0) > self.Om0:
                raise ValueError("baryon + neutrino content cannot exceed total matter content")
            self.Onu0 = Onu0
        else:
            self.Nnu, self.Mnu, self.Onu0 = 0., 0., 0.

        if flat:
            Ode0 = 1. -  Om0
            if Ode0 < 0.:
                raise ValueError("`Ode0` must be positive for a flat cosmology (adjust `Om0`)")
            self.Ode0, self.Ok0 = Ode0, 0.
        else:
            if Ode0 == Ellipsis:
                raise ValueError("`Ode0` is a required argument if non-flat cosmology is used")
            elif Ode0 < 0.:
                raise ValueError("`Ode0` must be positive")
            
            self.Ok0  = 1. - Om0 - Ode0
            self.Ode0 = Ode0

        self.flat       = flat
        self.heavy_nu   = heavy_nu
        self.POWER_NORM = 1.
        
        self.quadOptions(n = 5001, kmin = 1e-08, kmax = 1e+08) # with normalization

    def __repr__(self) -> str:
        return f"<CosmoStructure flat = {self.flat}, Om0 = {self.Om0:.3g}, Ob0 = {self.Ob0:.3g}, Ode0 = {self.Ode0:.3g}, sigma8 = {self.sigma8:.3g}, ns = {self.n:.3g}, h = {self.h:.3g}, Tcmb0 = {self.Tcmb0:.3g} K, Mnu = {self.Mnu:.3g} eV/c2; power = '{self.psmodel}', hmf = '{self.hmfmodel}', bias = '{self.biasmodel}'>"

    def quadOptions(self, n: int, kmin: float, kmax: float, renormalize: bool = True):
        r"""
        Set the k-space integration options. By default, integration range is chosen as :math:`[10^{-8}, 10^8]` with 5001 function evaluations.

        Parameters
        ----------
        n: int
            Number of function calls to make (or, size of k-space grid)
        kmin: float
            Lower integration limit
        kmax: float
            Upper integration limit
        renormalize: bool, optional
            If True, it will renormalize the power spectrum with given settings.
        """
        if not isinstance(n, int):
            raise TypeError("`n` sholud be an integer")
        if kmin < 0. or kmax < 0.:
            raise ValueError("both `kmin` and `kmax` should be positive")
        if kmin > kmax:
            print("Warning: `kmax` is smaller than `kmin`, they will be interchanged.")
            kmin, kmax = kmax, kmin
        lnk_min = np.log(kmin)
        lnk_max = np.log(kmax)
        dlnk    = (lnk_max - lnk_min) / (n-1)
        self._qopt = {'n': n, 'min': lnk_min, 'max': lnk_max, 'dlnk': dlnk}

        if renormalize:
            self.normalize()
        return
    
    def _quadSetup(self):
        """ k-space integration settings: nodes and weight """
        lnk = np.linspace(self._qopt['min'], self._qopt['max'], self._qopt['n'])
        return np.exp(lnk), self._qopt['dlnk']
    
    # ======================== Useful cosmology functions ==========================
    def Ez(self, z: float):
        r""" 
        Cosmology function :math:`E(z)` defined as

        .. math::
            E(z) = \sqrt{\Omega_{\rm m} (z + 1)^3 + \Omega_{\rm k} (z + 1)^2 + \Omega_{\rm de}} 

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            Value of :math:`E(z)`. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.Ez([0., 1., 2.])
        array([1.        , 1.76068169, 2.96647939])

        
        """
        zp1 = 1 + np.asarray(z)
        return np.sqrt(self.Om0 * zp1**3 + self.Ok0 * zp1**2 + self.Ode0)

    def rho_m(self, z: float):
        r""" 
        Matter density in units of :math:`kg / m^3` 

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            Density. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.rho_m([0., 1., 2.]) # in kg/m3
        array([2.76117045e-27, 2.20893636e-26, 7.45516022e-26])

        
        """
        return self.Om0 * RHO_CRIT0 * self.h**2 * (1 + np.asarray(z))**3

    def rho_de(self, z: float):
        r""" 
        Dark-energy density in units of :math:`kg / m^3` 

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            Density. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.rho_de([0., 1., 2.]) # in kg/m3
        array([6.44273105e-27, 6.44273105e-27, 6.44273105e-27])

        
        """
        return self.Ode0 * RHO_CRIT0 * self.h**2 * np.ones_like(z)

    def Omz(self, z: float):
        r"""
        Normalized matter density at redshift `z`.

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            Density. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.Omz([0., 1., 2.])
        array([0.3       , 0.77419355, 0.92045455])

                
        """
        return self.Om0 * (np.asarray(z) + 1)**3 / (self.Ez(z))**2

    def Odez(self, z: float):
        r"""
        Normalized matter density at redshift `z`.

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            Density. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.Odez([0., 1., 2.])
        array([0.7       , 0.22580645, 0.07954545])

                
        """
        return self.Ode0 / (self.Ez(z))**2
    
    def criticalDensity(self, z: float):
        r""" 
        Critical density in units of :math:`kg / m^3`.

        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        retval: array_like
            critical density. Has the same shape as `z`.

        Examples
        --------
        
        >>> cs.criticalDensity([0., 1., 2.]) # in kg/m3
        array([9.20390150e-27, 2.85320947e-26, 8.09943332e-26])

        
        """
        return RHO_CRIT0 * self.h**2 * self.Ez(z)**2
    
    def lagrangianR(self, m: float):
        r""" 
        Comoving lagrangian radius in Mpc/h.

        Parameters
        ----------
        m: array_like
            Mass in Msun/h

        Returns
        -------
        retval: array_like
            Lagrangian radius. Has the same shape as `m`.

        Examples
        --------
        
        >>> cs.lagrangianR([1.e+6, 1.e+9, 1.e+12]) # in Mpc/h
        array([0.0142066 , 0.14206603, 1.42066031])

        
        """
        rho0 = self.rho_m(0.) / (self.h**2 * Units.Msun / Units.Mpc**3)
        return (3. * np.asarray(m) / 4. / np.pi / rho0)**(1./3)
    
    def lagrangianM(self, r: float):
        r""" 
        Comoving mass corresponding to comoving lagrangian radius.

        Parameters
        ----------
        r: array_like
            Lagrangian radius in Mpch/

        Returns
        -------
        retval: array_like
            Mass in Msun/h. Has the same shape as `r`.

        Examples
        --------
        
        >>> m = cs.lagrangianM([0.01, 0.1, 1.]) # in Msun/h
        >>> m
        array([3.4876208e+05, 3.4876208e+08, 3.4876208e+11])

        

        To test the results, let's find the radius:
        
        >>> cs.lagrangianR(m) # same as input `r`
        array([0.01, 0.1 , 1.  ])

        

        .. versionadded:: 1.1
        """
        rho0 = self.rho_m(0.) / (self.h**2 * Units.Msun / Units.Mpc**3)
        return 4 * np.pi / 3 * np.asarray(r)**3 * rho0

    # ======================= Linear growth factor =================================    
    def _growthExact(self, z: float):
        """ Un-normalized linear growth factor (exact) """
        def _growthIntegrand(a: float):
            a = np.asarray(a)
            return a**1.5 / (self.Om0 + self.Ok0 * a + self.Ode0 * a**3)**1.5

        def _growthExact_scalar(z: float):
            # integration in `a` variable ...
            integral, _ = integrate.quad(_growthIntegrand, 0., 1. /(1. + z))
            return integral
        
        if np.isscalar(z):
            integral = _growthExact_scalar(z)
        else:
            integral = np.vectorize(_growthExact_scalar)(z)
        return integral * self.Ez(z) * self.Om0 * 2.5

    def _growthCarroll92(self, z: float):
        """ Un-normalized linear growth factor (fit by Carroll et al, 1992) """
        zp1 = 1 + np.asarray(z)
        Om  = self.Omz(z)
        Ode = self.Odez(z)
        return 2.5 / zp1 * Om * (Om**(4./7.) - Ode + (1 + Om / 2.) * (1 + Ode / 70.))**(-1)
    
    def _unn_growth(self, z: float):
        """ growth function (un-normalized) """
        if self.gfmodel == 'exact':
            return self._growthExact(z)
        elif self.gfmodel == 'carroll92':
            return self._growthCarroll92(z)
        return

    def growth(self, z: float, ignore_norm: bool = False):
        r""" 
        Returns the normalized linear growth factor. This is **not designed for vector input**. Growth factor is calculated either by using the fit by Carroll et al (1992) or by evaluating the integral

        .. math::
            D_{+}(z) \propto H(z) \int_z^\infty {\rm d}z' \frac{1 + z'}{E^3(z')} 

        Parameters
        ----------
        z: float
            Redshift
        ignore_norm: bool
            Ignore the normalization if set true.

        Returns
        -------
        retval: float
            Linear growth factor. 

        Examples
        ----------------
        
        >>> cs.growth(1.)
        0.6118057532973097

        
        """
        Dplus = self._unn_growth(z)
        if ignore_norm:
            return Dplus
        return Dplus / self._unn_growth(0.)

    # ========================= Power spectrum =====================================    
    def transfer(self, k: float, z: float = 0.):
        r""" 
        Evaluate the current model of transfer function, using the current cosmology.

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc.
        z: float, optional
            Redshift. Used only when the transfer function need it.
        _zisDz: bool, optional
            If True, take the `z` parameter as growth, else as redshift. Default is False.

        Returns
        -------
        retval: array_like
            Transfer function. Has the same shape as `k`.

        Examples
        --------
        
        >>> cs.transfer([1.e-4, 1., 1.e+4])
        array([9.99906939e-01, 4.59061096e-03, 2.18246719e-10])

        
        """
        fdata = Transfer.available[self.psmodel]
        args  = {'k': k, 'h': self.h, 'Om0': self.Om0, 'Ob0': self.Ob0, 'Tcmb0': self.Tcmb0}
        if fdata.z_dep:
            args['z'] = z
        if fdata.model == 'mdm':
            args['Onu0'] = self.Onu0
            args['Ode0'] = self.Ode0
            args['Nnu']  = self.Nnu

            # find (un-normalised) growth function at z
            args['z']      = self._unn_growth(z)
            args['_zisDz'] = True
        return fdata.f(**args)
    
    def filt(self, x: float, deriv: bool = False):
        r""" Top-hat filter """
        if deriv:
            return 3. * ((x**2 - 3.) * np.sin(x) + 3. * x * np.cos(x)) / x**4
        return (np.sin(x) - x * np.cos(x)) * 3. / x**3 
    
    def _unn0_matterPowerSpectrum(self, k: float, z: float = 0.):
        r""" un-normalised matter power spectrum """
        k = np.asarray(k)
        return k**self.n * self.transfer(k, z)**2

    def _unn0_correlation(self, r: float, z: float = 0.):
        r""" un-normalised 2-point correlation function """
        k, dlnk = self._quadSetup()

        # integration in log(k) variable ...
        kr       = np.outer(r, k)
        integ    = k**3 * self._unn0_matterPowerSpectrum(k, z) * np.sin(kr) / kr
        integral = integrate.simps(integ, dx = dlnk, axis = -1) # k in columns

        xi = integral / 2. / np.pi**2
        if np.isscalar(r):
            return xi[0]
        return xi

    def _unn0_variance(self, r: float, z: float = 0.):
        r""" un-normalised variance at z = 0 """
        k, dlnk = self._quadSetup()
        
        # integration in log(k) variable ...
        kr       =  np.outer(r, k)
        integ    = k**3 * self._unn0_matterPowerSpectrum(k, z) * self.filt(kr)**2
        integral = integrate.simps(integ, dx = dlnk, axis = -1) # k in columns

        # def _varianceIntegrand(lnk: float):
        #     k = np.exp(lnk)
        #     return k**3 * self._unn0_matterPowerSpectrum(k) * self.filt(k * r)**2

        # integral, _ = integrate.quad(_varianceIntegrand, lnk_min, lnk_max)

        sigma2 = integral / 2. / np.pi**2
        if np.isscalar(r):
            return sigma2[0]
        return sigma2
    
    def _unn0_varianceDeriv(self, r: float, z: float = 0.):
        r""" un-normalised derivative of variance """
        k, dlnk = self._quadSetup()
        
        # integration in log(k) variable ...
        kr       = np.outer(r, k)
        integ    = k**4 * self._unn0_matterPowerSpectrum(k, z) * self.filt(kr) * self.filt(kr, deriv = True)
        integral = integrate.simps(integ, dx = dlnk, axis = -1) / np.pi**2 # k in columns

        # def _varianceIntegrand(lnk: float):
        #     k  = np.exp(lnk)
        #     kr = k * r
        #     return k**4 * self._unn0_matterPowerSpectrum(k) * self.filt(kr) * self.filt(kr, deriv = True)
        
        # integral, _ = integrate.quad(_varianceIntegrand, lnk_min, lnk_max) / np.pi**2
        
        if np.isscalar(r):
            return integral[0]
        return integral
    
    def normalize(self):
        r""" Normalise the power spectrum. """
        _var8       = self._unn0_variance(8.)
        _var8_model = self.sigma8**2
        self.POWER_NORM = _var8_model / _var8
        return
    
    def matterPowerSpectrum(self, k: float, z: float = 0., ignore_norm: bool = False):
        r""" 
        Get the (normalised) linear matter power spectrum at redshift :math:`z`. 

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc.
        z: float
            Redshift
        ignore_norm: bool
            Ignore the normalization and return the un-normalized value.

        Returns
        -------
        retval: array_like
            Power spectrum. Has the same shape as `k`.

        Examples
        --------
        
        >>> cs.matterPowerSpectrum([1.e-4, 1., 1.e+4], z = 0.)
        array([3.23064206e+02, 6.80942839e+01, 1.53909393e-09])
        >>> cs.matterPowerSpectrum([1.e-4, 1., 1.e+4], z = 1.)
        array([1.20924961e+02, 2.54881181e+01, 5.76092522e-10])

        
        """
        unn_pk = self._unn0_matterPowerSpectrum(k, z) * self.growth(z)**2
        if ignore_norm:
            return unn_pk
        return unn_pk * self.POWER_NORM
    
    def matterPowerSpectrumNorm(self):
        """ Return the value of power spectrum norm """
        return self.POWER_NORM

    def correlation(self, r: float, z: float = 0., ignore_norm: bool = False):
        r"""
        Get the (normalised) 2-point correlation function ar scale :math:`r`. This is the Fourier transform of the matter power spectrum and is found by evaluating the integral

        .. math::
            \xi(r) = \frac{1}{2\pi^2} \int_0^\infty P(k) \frac{\sin (kr)}{kr} k^2 {\rm d}k

        Parameters
        ----------
        r: array_like
            Radius in Mpc/h
        z: float
            Redshift
        ignore_norm: bool
            If true, give the un-normalized values.

        Returns
        -------
        retval: array_like
            Value of 2-pt. correlation function. Has the same shape as `r`.

        Examples
        --------
        
        >>> cs.correlation([0.01, 0.1, 1.], z = 0.)
        array([79.9002375 , 27.03930878,  5.39579643])
        >>> cs.correlation([0.01, 0.1, 1.], z = 1.)
        array([29.90716065, 10.12098308,  2.01968049])

        

        .. versionadded:: 1.1
        """
        y = self._unn0_correlation(r, z) * self.growth(z)**2
        if ignore_norm:
            return y
        return y * self.POWER_NORM

    def _variance(self, r: float, z: float = 0., ignore_norm: bool = False, deriv: bool = False):
        """ variance """
        if deriv:
            y = self._unn0_varianceDeriv(r, z) * self.growth(z)**2
        else:
            y = self._unn0_variance(r, z) * self.growth(z)**2
        if ignore_norm:
            return y
        return y * self.POWER_NORM

    def variance(self, r: float, z: float = 0., ignore_norm: bool = False, deriv: bool = False):
        r""" 
        Get the (normalised) value of variance of the density fluctuations, smoothed at a scale of :math:`r` using a top-hat filter. It is found by evaluating the integral

        .. math::
            \sigma^2(r) = \frac{1}{2 \pi^2} \int_0^\infty P(k) W(kr) k^2 {\rm d}k

        where :math:`W(x) = 3 (\sin x - x \cos x) / x^2` is the spherical top-hat filter in Fourier space.

        Parameters
        ----------
        r: array_like
            Radius of spherical mass (scale) in Mpc/h.
        z: float
            Redshift
        ignore_norm: bool
            If true, give the un-normalized values.
        deriv: bool
            If true, give the first derivative w.r.to the radius.

        Returns
        -------
        retval: array_like
            Variance. Has the same shape as `r`.

        Examples
        --------
        
        >>> cs.variance([0.01, 0.1, 1.], z = 1.)
        array([30.92459974, 10.67772887,  2.2518911 ])
        >>> cs.variance([0.01, 0.1, 1.], z = 1., deriv = True) 
        array([-1214.43822506,   -58.31250423,    -1.88953923])

        
        """
        # if np.isscalar(r):
        #     return self._variance(r, z, ignore_norm, deriv)
        # x = np.empty_like(r)
        # for i in range(x.shape[0]):
        #     x[i] = self._variance(r[i], z, ignore_norm, deriv)
        # return x
        return self._variance(r, z, ignore_norm, deriv)
    
    def sigma(self, r: float, z: float = 0., ignore_norm: bool = False, deriv: bool = False):
        r""" 
        Get the square root of variance (std. deviation) or its logarithmic derivative w. r. to radius (i.e., :math:`{\rm d}\log \sigma/{\rm d}\log r`).

        Parameters
        ----------
        r: array_like
            Radius of spherical mass (scale) in Mpc/h.
        z: float
            Redshift.
        ignore_norm: bool
            If true, give the un-normalized value of :math:`sigma`. It is not used for derivative.
        deriv: bool
            If true, give the first derivative w.r.to the radius.

        Returns
        -------
        retval: array_like
            Value of :math:`\sigma`. Has the same shape as `r`.

        Examples
        --------
        
        >>> cs.sigma([0.01, 0.1, 1.], z = 1.) # sigma
        array([5.5609891 , 3.26767943, 1.50063023])
        >>> cs.sigma([0.01, 0.1, 1.], z = 1., deriv = True) # dln(sigma)/dln(r)
        array([-0.19635472, -0.27305668, -0.41954498])

        
        """
        r = np.asarray(r)
        if deriv: # dlnsdlnr
            var  = self.variance(r, z, ignore_norm = True, deriv = False)
            dvar = self.variance(r, z, ignore_norm = True, deriv = True)
            return r * dvar / var / 2.
        return np.sqrt(self.variance(r, z, ignore_norm, False))
        
    def dlnsdlnm(self, r: float, z: float = 0.):
        r""" 
        Get the logarithmic derivative of :math:`\sigma` w.r.to mass. This is 1/3 times the derivative w.r.to radius.

        Parameters
        ----------
        r: array_like
            Radius in Mpc/h.
        z: float, optional
            Redshift

        Returns
        -------
        y: array_like
            Values of derivative. Has the same size as `m`.

        Examples
        --------
        
        >>> cs.dlnsdlnm([0.01, 0.1, 1.])
        array([-0.06545157, -0.09101889, -0.13984833])
        
        
        """
        # var  = self.variance(r, z = 0., ignore_norm = True)
        # dvar = self.variance(r, z = 0., ignore_norm = True, deriv = True)
        # return np.asarray(r) * dvar / var / 3.
        return self.sigma(r, deriv = True) / 3.
    
    def radius(self, sigma: float, z: float = 0., rmin: float = 1.e-3, rmax: float = 1.e+4, reltol: float = 1.e-8):
        r"""
        Get the radius from the sigma values. This solves for the radius using Brent's method (:meth:`scipy.optimize.brentq`).

        Parameters
        ----------
        sigma: array_like
            Mass variance
        z: float
            Redshift
        rmin: float, optional
            Lower bracketing limit of r. Default is 1e-3.
        rmax: float, optional
            Upper bracketing limit of r. Default is 1e+4.
        reltol: float, optional
            Relative tolerance for the result. Default is 1e-8.

        Returns
        -------
        r: array_like
            Radius in Mpc/h

        Examples
        --------
        
        >>> r = cs.radius([2., 1., 0.1, 0.01], z = 0.)
        >>> r
        array([  1.58825172,   5.71225866,  70.47066068, 330.48096125])

        

        To check the result, 
        
        >>> s = cs.sigma(r)
        >>> s
        array([2.  , 1.  , 0.1 , 0.01])
        >>> np.allclose(s, [2., 1., 0.1, 0.01])
        True

        

        .. versionadded:: 1.1
        """
        def _radius(s):
            return brentq(lambda r: self.sigma(r, z) - s, rmin, rmax, rtol = reltol)
        if np.isscalar(sigma):
            return _radius(sigma)
        return np.vectorize(_radius)(sigma)

    # ============================ Mass function ===================================
    def parseMassDefn(self, mdef: str, z: float):
        r""" 
        This convert the mass definition to overdensity values.

        Parameters
        ----------
        mdef: str
            one of the SO mass definitions. `vir`, `*m` and `*c` are only choises and others will raise exception.
        z: float
            Redshift
        
        Returns
        -------
        retval: float
            Overdensity in :math:`h^2 Msun/Mpc^3`.

        Examples
        --------
        
        >>> for mdef in ['200m', '500c', 'vir']:
        ...     print(cs.parseMassDefn(mdef, z = 0.))   # in kg/m3
        ... 
        5.522340901858673e-25
        4.601950751548894e-24
        9.295940518128765e-25

        
        """
        if mdef == "vir": # value given by Bryan & Norman 1998
            d = self.Omz(z) - 1.
            return int(18 * np.pi**2 + 82 * d - 39 * d**2) * self.criticalDensity(z)
        delta, ref = mdef[:-1], mdef[-1]
        if ref == 'm':
            return int(delta) * self.rho_m(z)
        elif ref == 'c':
            return int(delta) * self.criticalDensity(z)
        raise NotImplementedError(f'definition `{mdef}` not implemented')
    
    def fit(self, sigma: float, mdef: str = ..., z: float = ...):
        r""" 
        Get the fitting function.

        Parameters
        ----------
        sigma: array_like
            Sigma values (square root of variance)
        mdef: str, optional
            Mass definition corresponding to the halo finder. This function is defined only for Fof and spherical overdensity (SO) mass definitions (i.e., `*m`, `*c` and `vir`). 
        z: float, optional
            Redshift.
        
        Note: Not all these parameters are needed for all fitting functions.

        Returns
        -------
        f: array_like
            Values of fit. Has the same size as `sigma`.

        Examples
        --------
        
        >>> s = np.linspace(0., 10., 21)
        >>> cs.hmfmodel
        'tinker08'
        >>> cs.fit([0.1, 1., 10.], mdef = '200m', z = 0.)
        array([4.62093863e-51, 2.83208269e-01, 2.08742594e-01])

                
        """
        fdata = MassFunction.available[self.hmfmodel]
        args  = {'sigma': sigma}

        if mdef == Ellipsis: # use default
            mdef = '200m' if 'so' in fdata.mdef else fdata.mdef[0]
        elif ('so' if (mdef == 'vir' or mdef[-1] in "mc") else mdef) not in fdata.mdef:
            raise ValueError(f"cannot use mdef `{mdef}` for `{self.hmfmodel}` fit")

        if mdef != 'fof':
            args['Delta'] = round(self.parseMassDefn(mdef, z) / self.rho_m(z))
        if fdata.z_dep:
            if z == Ellipsis:
                raise ValueError(f"redshift must be given for `{self.hmfmodel}` fit")
            args['z'] = z
        return fdata.f(**args)
    
    def massFunction(self, m: float, mdef: str = '200m', z: float = 0., form: str = 'dndm'):
        r""" 
        Get the halo mass-function with current settings. It is given by in terms of a fitting (or, multiplicity) function :math:`f(\sigma)` as

        .. math::
            \frac{{\rm d}n}{{\rm d}M} {\rm d}M = \frac{\rho_{\rm m}}{M^2} f(\sigma) \left\vert \frac{{\rm d} \ln \sigma}{{\rm d} \ln M} \right\vert {\rm d}M

        Parameters
        ----------
        m: array_like
            Mass of halo
        mdef: str, optional
            Mass definition corresponding to the halo finder. This function is defined only for spherical overdensity (SO) mass definitions, i.e., `*m`, `*c` and `vir`. 
        z: float, optional
            Redshift. Default is 0.
        form: str, optional
            Format of output. Available values are `dndm` (which give :math:`\frac{{\rm d}n}{{\rm d}M}`, default), `dndlnm` (for :math:`\frac{{\rm d}n}{{\rm d}\log M}`) and `fsigma` (for :math:`f(\sigma)`).

        Returns
        -------
        retval: float
            Values of mass-function in specified format. Has the same size as `m`.

        Examples
        --------
        
        >>> cs.massFunction([1.e+6, 1.e+9, 1.e+12], mdef = '200m', z = 0., form = 'fsigma')
        array([0.21462692, 0.2465193 , 0.33287869])
        >>> cs.massFunction([1.e+6, 1.e+9, 1.e+12], mdef = '200m', z = 0., form = 'dndm')
        array([1.22326921e-03, 1.98075984e-09, 4.18069529e-15])
        >>> cs.massFunction([1.e+6, 1.e+9, 1.e+12], mdef = '200m', z = 0., form = 'dndlnm')
        array([1.22326921e+03, 1.98075984e+00, 4.18069529e-03])

        
        
        """
        m = np.asarray(m)
        r   = self.lagrangianR(m)
        s   = self.sigma(r, z)
        fs  = self.fit(s, mdef, z)
        if form == 'fsigma':
            return fs
        else:
            rho  = self.rho_m(z) / (self.h**2 * Units.Msun / Units.Mpc**3)
            dndm = fs * np.abs(self.dlnsdlnm(r, z)) * rho / m**2
            if form == 'dndm':
                return dndm
            elif form == 'dndlnm':
                return dndm * m
        raise ValueError(f"unknown format `{form}`")
        
    # ========================== Bias and 2-halo terms =============================
    def peakHeight(self, m: float, z: float = 0.):
        r"""
        Get the peak height corresponding to mass :math:`m`. It is given by :math:`\nu = \delta_{\rm c} / \sigma(M, z)`.

        Parameters
        ----------
        m: array_like
            Mass in Msun/h
        z: float
            Redshift

        Returns
        -------
        retval: array_like
            Values of peak height. Has the same size as `m`.

        Examples
        --------
        
        >>> cs.peakHeight([1.e+06, 1.e+09, 1.e+12], z = 0.)
        array([0.1990946 , 0.34851773, 0.80123727])

        

        .. versionadded:: 1.1
        """
        r = self.lagrangianR(m)
        return DELTA_C / self.sigma(r, z, ignore_norm = False, deriv = False)

    def bias(self, m: float, z: float = 0., mdef: str = ...):
        r"""
        Get the (large scale) linear halo bias function.

        Parameters
        ----------
        m: array_like
            Mass in Msun/h
        mdef: str, optional
            Mass definition corresponding to the halo finder. This function is defined only for spherical overdensity (SO) mass definitions, i.e., `*m`, `*c` and `vir`. 
        z: float, optional
            Redshift. Default is 0.

         Returns
        -------
        retval: array_like
            Values of bias function. Has the same size as `m`.

        Examples
        --------
        
        >>> cs.biasmodel
        'tinker10'
        >>> cs.bias([1.e+06, 1.e+09, 1.e+12], z = 0., mdef = '200m')
        array([0.59202591, 0.61077932, 0.81167364])

        

        .. versionadded:: 1.1
        """
        nu = self.peakHeight(m, z)

        fdata = Bias.available[self.biasmodel]
        args  = {'nu': nu}

        if mdef == Ellipsis: # use default
            mdef = '200m' if 'so' in fdata.mdef else fdata.mdef[0]
        elif ('so' if (mdef == 'vir' or mdef[-1] in "mc") else mdef) not in fdata.mdef:
            raise ValueError(f"cannot use mdef `{mdef}` for `{self.biasmodel}` model")

        if mdef != 'fof':
            args['Delta'] = round(self.parseMassDefn(mdef, z) / self.rho_m(z))
        if fdata.z_dep:
            args['z'] = z
        return fdata.f(**args)


if __name__ == "__main__":
    # ================================ Doctest =====================================
    import doctest
    cs = CosmoStructure(Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, n = 1., h = 0.7)
    doctest.testmod()