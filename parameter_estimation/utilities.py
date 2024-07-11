import os
import numpy as np
import jax.numpy as jnp

MTsun_SI = 4.925490947641266978197229498498379006e-6

#################### Directories Management ####################

def mkdir(path):
    """
    To create a directory if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


#################### GW Functions ####################

def Mc_q_to_m1m2(Mc, q):
    """
    Transforming the chirp mass Mc and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    Mc : Float
            Chirp mass.
    q : Float
            Mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    eta = q / (1 + q) ** 2
    M_tot = Mc / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2

def rotate_y(angle, x, y, z):
    """
    Rotate the vector (x, y, z) about y-axis
    """
    x_new = x * jnp.cos(angle) + z * jnp.sin(angle)
    z_new = - (x * jnp.sin(angle)) + z * jnp.cos(angle)
    return x_new, y, z_new


def rotate_z(angle, x, y, z):
    """
    Rotate the vector (x, y, z) about z-axis
    """
    x_new = x * jnp.cos(angle) - y * jnp.sin(angle)
    y_new = x * jnp.sin(angle) + y * jnp.cos(angle)
    return x_new, y_new, z
    

def spin_to_spin(
    thetaJN, 
    phiJL, 
    theta1, 
    theta2, 
    phi12, 
    chi1, 
    chi2, 
    M_c, 
    eta,
    fRef, 
    phiRef
):
  
    LNhx = 0.
    LNhy = 0.
    LNhz = 1.

    s1hatx = jnp.sin(theta1)*jnp.cos(phiRef)
    s1haty = jnp.sin(theta1)*jnp.sin(phiRef)
    s1hatz = jnp.cos(theta1)
    s2hatx = jnp.sin(theta2) * jnp.cos(phi12+phiRef)
    s2haty = jnp.sin(theta2) * jnp.sin(phi12+phiRef)
    s2hatz = jnp.cos(theta2)
  
    temp = (1 / eta / 2 - 1)
    q = temp - (temp ** 2 - 1) ** 0.5
    m1, m2 = Mc_q_to_m1m2(M_c, q)
    v0 = jnp.cbrt((m1+m2) * MTsun_SI * jnp.pi * fRef)
  
    Lmag = ((m1+m2)*(m1+m2)*eta/v0) * (1.0 + v0*v0*(1.5 + eta/6.0))
    s1x = m1 * m1 * chi1 * s1hatx
    s1y = m1 * m1 * chi1 * s1haty
    s1z = m1 * m1 * chi1 * s1hatz
    s2x = m2 * m2 * chi2 * s2hatx
    s2y = m2 * m2 * chi2 * s2haty
    s2z = m2 * m2 * chi2 * s2hatz
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag + s1z + s2z
  

    Jnorm = jnp.sqrt( Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    theta0 = jnp.arccos(Jhatz)
    phi0 = jnp.arctan2(Jhaty, Jhatx)

    s1hatx, s1haty, s1hatz = rotate_z(-phi0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phi0, s2hatx, s2haty, s2hatz)
  
    LNhx, LNhy, LNhz = rotate_y(-theta0, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = rotate_y(-theta0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-theta0, s2hatx, s2haty, s2hatz)
    
    LNhx, LNhy, LNhz = rotate_z(phiJL - jnp.pi, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = rotate_z(phiJL - jnp.pi, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(phiJL - jnp.pi, s2hatx, s2haty, s2hatz)

    Nx=0.0
    Ny=jnp.sin(thetaJN)
    Nz=jnp.cos(thetaJN)
    iota=jnp.arccos(Nx*LNhx+Ny*LNhy+Nz*LNhz)
  
    thetaLJ = jnp.arccos(LNhz)
    phiL = jnp.arctan2(LNhy, LNhx)
  
    s1hatx, s1haty, s1hatz = rotate_z(-phiL, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phiL, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_z(-phiL, Nx, Ny, Nz)
    
    s1hatx, s1haty, s1hatz = rotate_y(-thetaLJ, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-thetaLJ, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_y(-thetaLJ, Nx, Ny, Nz)

    phiN = jnp.arctan2(Ny, Nx)
    s1hatx, s1haty, s1hatz = rotate_z(jnp.pi/2.-phiN-phiRef, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(jnp.pi/2.-phiN-phiRef, s2hatx, s2haty, s2hatz)

    S1x = s1hatx*chi1
    S1y = s1haty*chi1
    S1z = s1hatz*chi1
    S2x = s2hatx*chi2
    S2y = s2haty*chi2
    S2z = s2hatz*chi2
    
    return iota, S1x, S1y, S1z, S2x, S2y, S2z


#################### Miscellaneous Functions ####################
    
def KLdivergence(x, y):
    """
    Compute the Kullback-Leibler divergence between two samples.
    Parameters
    ----------
    x : 1D np array
        Samples from distribution P, which typically represents the true
        distribution.
    y : 1D np rray
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    """ 
    bins = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)), 100) 
    # The size of array must be much greater than the number of bins
    
    prob_x = np.histogram(x, bins=bins)[0]/x.shape
    prob_y = np.histogram(y, bins=bins)[0]/y.shape
       
    return np.sum(np.where((prob_y != 0) & (prob_x != 0), prob_y * (np.log(prob_y) - np.log(prob_x)), 0))


def JSdivergence(P, Q):
    """
    Compute the Jensen-Shannon divergence between two samples.
    Parameters
    ----------
    P : 1D np array
        Samples from distribution P
    Q : 1D np array
        Samples from distribution Q
    Returns
    -------
    out : float
        The estimated Jensen-Shannon divergence D(P||Q).
    """
    M=(P+Q)/2 #sum the two distributions then get average

    kl_p_q = KLdivergence(P, M)
    kl_q_p = KLdivergence(Q, M)

    js = (kl_p_q+kl_q_p)/2
    return js
