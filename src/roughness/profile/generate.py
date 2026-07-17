# roughness/profile/generate.py

import numpy as np
from scipy.special import gamma


def psd_noise_contrib(freq, sigma_noise):
    """ Contribution of sigma_noise in a (discrete) PSD """
    ds = ds_from_freq(freq)
    return sigma_noise ** 2 * ds / (2. * np.pi)

def ds_from_freq(freq):
    """ Return step size 'ds' from 'freq' issued from np.fft.rfftfreq(N, d=ds) """
    N = len(freq)
    if N % 2 == 0:
        return 1 / ((2 * (N + 1) + 1) * (freq[1] - freq[0]))
    else:
        return 1 / (2 * (N + 1) * (freq[1] - freq[0]))

def psd_lorentzian(freq, sigma, xi, alpha, sigma_noise=0, ndim=None):
    ndim = 1
    kn = freq * 2 * np.pi
    psd0 = sigma ** 2 * xi * (1 / np.sqrt(np.pi)) * gamma(ndim * alpha + 0.5) / gamma(ndim * alpha)
    psd = psd0 / (1.0 + (kn * xi) ** 2) ** (ndim * alpha + 0.5)

    if sigma_noise > 0:
        psd += psd_noise_contrib(freq, sigma_noise)
    
    return psd

def psd_lorentzian_2(freq, sigma, xi, alpha, sigma_noise=0, ndim=None):
    """
    Return the second formulation of lorentzian PSD

    Parameters
    ----------
    freq: np.array
        array of sampled frequencies
    sigma, xi, alpha: floats
        PSD model parameters
    sigma_noise: float, optional
        Standard deviation associated with the noise
    """
    # TODO: the need to introduce ndim into the formula needs to be studied in more detail
    ndim = 1
    kn = freq
    psd0 = sigma ** 2 * xi * (1 / np.sqrt(np.pi)) * gamma(ndim * alpha + 0.5) / gamma(ndim * alpha)
    psd = psd0 / (1.0 + (kn * xi) ** 2) ** (ndim * alpha + 0.5)

    if sigma_noise > 0:
        psd += psd_noise_contrib(freq, sigma_noise)

    return psd

def acf(u, sigma, xi, alpha, sigma_noise=0.):
    """
    Auto-Covariance Function (ACF) related to the Sinha & al expression with u = m.dy.
    """
    return sigma ** 2 * np.exp(-(np.abs(u / xi) ** (2 * alpha))) + (sigma_noise ** 2) * (u == 0)


def psd_azarnouche(freq, sigma, xi, alpha, sigma_noise=0., ds=1.0):
    """
    PSD evaluation from the Azer-Nouche expression related to the discrete Auto-correlation Function

    Parameters
    ----------
    freq: np.array
        array of sampled frequencies
    sigma, xi, alpha: floats
        PSD model parameters
    sigma_noise: float, optional
        Standard deviation associated with the noise
    ds: float, optional
        space sampling
    """
    N = len(freq)
    kn = 2.0 * np.pi * freq

    Pn = acf(0.0, sigma, xi, alpha) * N  # P0 contribution
    for m in range(1, N):
        u = m * ds
        Pn += (2.0 * acf(u, sigma, xi, alpha) * np.cos(kn * u) * (N - m))
    Pn *= ds / (2.0 * np.pi * N)

    if sigma_noise > 0:
        Pn += psd_noise_contrib(freq, sigma_noise)

    return Pn

# ---- original generator, unmodified except: seed param, no squeeze ----
def synthetic_line_vectorised(N, dx=1.,
                               psd_model=None,
                               sigma=1, xi=10, alpha=0.5, sigma_noise=0., mu=0.,
                               N_realizations=1, seed=None):
    

    if psd_model is None:
        psd_model = psd_lorentzian
    elif isinstance(psd_model, str):
        if psd_model == "azarnouche":
            psd_model = psd_azarnouche
        elif psd_model == "lorentzian 2":
            psd_model = psd_lorentzian_2
        else:
            raise ValueError(f"Unknown PSD model: {psd_model}")
    

    
    freq = np.fft.rfftfreq(N, d=dx)

    kwargs = {"ds": dx} if psd_model in (psd_azarnouche, psd_lorentzian_2) else {}
    psd = psd_model(freq, sigma, xi, alpha, **kwargs)

    ampli = np.sqrt(psd * 2 * np.pi * N / dx)

    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N_realizations, psd.shape[0]))

    inv_z = ampli * (np.cos(phi) + 1j * np.sin(phi))

    inv_z[:, 0] = ampli[0]
    if N % 2 == 0:
        inv_z[:, -1] = ampli[-1]

    z = np.fft.irfft(inv_z, n=N, axis=1)
    z -= (np.mean(z, axis=1, keepdims=True) - mu)

    if sigma_noise > 0:
        z += rng.normal(0, sigma_noise, size=z.shape)

    return z  # always (N_realizations, N)


# ---- new wrapper: single entry point for cdsaxs/cdsem/tests ----
def generate_profiles(N, dx, sigma, xi, alpha, N_realizations, seed,
                       sigma_noise=0., mu=0., psd_model=psd_lorentzian, save_file=None):
    """
    Shared entry point. Returns shape (N_realizations, N) always.
    """
    profiles = synthetic_line_vectorised(
        N=N, dx=dx, psd_model=psd_model,
        sigma=sigma, xi=xi, alpha=alpha,
        sigma_noise=sigma_noise, mu=mu,
        N_realizations=N_realizations, seed=seed
    )

    if save_file:
        #concatenate N dx sigma xi and alpha in the file name which is save_file string
        file_name = f"{save_file}_N{N}_dx{dx}_sigma{sigma}_xi{xi}_alpha{alpha}.npy"
        np.save(file_name, profiles)

    return profiles


def validate_profile_psd(z, dx, sigma, xi, alpha, sigma_noise=0., psd_model=psd_lorentzian):
    """
    Re-extract PSD from generated profile(s) z, compare to target model.
    z: shape (N_realizations, N) or (N,)
    Returns freq, psd_measured (ensemble-averaged), psd_target
    """
    if z.ndim == 1:
        z = z[None, :]

    N = z.shape[1]
    freq = np.fft.rfftfreq(N, d=dx)

    Z = np.fft.rfft(z, axis=1)
    psd_measured = (np.abs(Z) ** 2).mean(axis=0) * dx / (2 * np.pi * N)

    psd_target = psd_model(freq, sigma, xi, alpha, sigma_noise=sigma_noise)

    return freq, psd_measured, psd_target