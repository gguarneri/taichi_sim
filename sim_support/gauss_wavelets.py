from sim_support import *
import numpy as np
from scipy.signal import gausspulse


def gaussian_pulse(*args, **kwargs):
    return gausspulse(*args, **kwargs)


def gaussian_first_dev_pulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
               retenv=False):
    """
    Return the first derivative of a Gaussian modulated sinusoid:

        ``-t*exp(-a*(t)^2) exp(1j*2*pi*fc*t).``

    If `retquad` is True, then return the real and imaginary parts
    (in-phase and quadrature).
    If `retenv` is True, then return the envelope (unmodulated signal).
    Otherwise, return the real part of the modulated sinusoid.

    Parameters
    ----------
    t : ndarray or the string 'cutoff'
        Input array.
    fc : float, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.
    tpr : float, optional
        If `t` is 'cutoff', then the function returns the cutoff
        time for when the pulse amplitude falls below `tpr` (in dB).
        Default is -60.
    retquad : bool, optional
        If True, return the quadrature (imaginary) as well as the real part
        of the signal.  Default is False.
    retenv : bool, optional
        If True, return the envelope of the signal.  Default is False.

    Returns
    -------
    yI : ndarray
        Real part of signal.  Always returned.
    yQ : ndarray
        Imaginary part of signal.  Only returned if `retquad` is True.
    yenv : ndarray
        Envelope of signal.  Only returned if `retenv` is True.

    Examples
    --------
    Plot real component, imaginary component, and envelope for a 5 Hz pulse,
    sampled at 100 Hz for 2 seconds:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 2 * 100, endpoint=False)
    >>> i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
    >>> plt.plot(t, i, t, q, t, e, '--')

    """
    if fc < 0:
        raise ValueError(f"Center frequency (fc={fc:.2f}) must be >=0.")
    if bw <= 0:
        raise ValueError(f"Fractional bandwidth (bw={bw:.2f}) must be > 0.")
    if bwr >= 0:
        raise ValueError(f"Reference level for bandwidth (bwr={bwr:.2f}) "
                         "must be < 0 dB")

    ref = pow(10.0, bwr / 20.0)
    # fdel = fc*bw/2:  g(fdel) = ref --- solve this for a
    #
    # pi^2/a * fc^2 * bw^2 /4=-log(ref)
    a = -(PI * fc * bw) ** 2 / (4.0 * np.log(ref))

    if isinstance(t, str):
        if t == 'cutoff':  # compute cut_off point
            #  Solve exp(-a tc**2) = tref  for tc
            #   tc = sqrt(-log(tref) / a) where tref = 10^(tpr/20)
            if tpr >= 0:
                raise ValueError("Reference level for time cutoff must "
                                 "be < 0 dB")
            tref = pow(10.0, tpr / 20.0)
            return np.sqrt(-np.log(tref) / a)
        else:
            raise ValueError("If `t` is a string, it must be 'cutoff'")

    ygauss = np.exp(-a * t * t)
    yenv = - 2 * ygauss * (a * t)
    yI = - 2 * ygauss * (a * t * np.cos(2 * PI * fc * t) + PI * fc * np.sin(2 * PI * fc * t))
    yQ = yenv * np.sin(2 * PI * fc * t)
    if not retquad and not retenv:
        return yI
    if not retquad and retenv:
        return yI, yenv
    if retquad and not retenv:
        return yI, yQ
    if retquad and retenv:
        return yI, yQ, yenv

    
def gaussian_second_dev_pulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
               retenv=False):
    """
    Return the second derivative of a Gaussian (Ricker) modulated sinusoid:

        ``(1.0 - 2.0*a*(t-t0)^2)*exp(-a*(t-t0)^2) exp(1j*2*pi*fc*t).``

    If `retquad` is True, then return the real and imaginary parts
    (in-phase and quadrature).
    If `retenv` is True, then return the envelope (unmodulated signal).
    Otherwise, return the real part of the modulated sinusoid.

    Parameters
    ----------
    t : ndarray or the string 'cutoff'
        Input array.
    fc : float, optional
        Center frequency (e.g. Hz).  Default is 1000.
    bw : float, optional
        Fractional bandwidth in frequency domain of pulse (e.g. Hz).
        Default is 0.5.
    bwr : float, optional
        Reference level at which fractional bandwidth is calculated (dB).
        Default is -6.
    tpr : float, optional
        If `t` is 'cutoff', then the function returns the cutoff
        time for when the pulse amplitude falls below `tpr` (in dB).
        Default is -60.
    retquad : bool, optional
        If True, return the quadrature (imaginary) as well as the real part
        of the signal.  Default is False.
    retenv : bool, optional
        If True, return the envelope of the signal.  Default is False.

    Returns
    -------
    yI : ndarray
        Real part of signal.  Always returned.
    yQ : ndarray
        Imaginary part of signal.  Only returned if `retquad` is True.
    yenv : ndarray
        Envelope of signal.  Only returned if `retenv` is True.

    Examples
    --------
    Plot real component, imaginary component, and envelope for a 5 Hz pulse,
    sampled at 100 Hz for 2 seconds:

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 2 * 100, endpoint=False)
    >>> i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
    >>> plt.plot(t, i, t, q, t, e, '--')

    """
    if fc < 0:
        raise ValueError(f"Center frequency (fc={fc:.2f}) must be >=0.")
    if bw <= 0:
        raise ValueError(f"Fractional bandwidth (bw={bw:.2f}) must be > 0.")
    if bwr >= 0:
        raise ValueError(f"Reference level for bandwidth (bwr={bwr:.2f}) "
                         "must be < 0 dB")

    ref = pow(10.0, bwr / 20.0)
    # fdel = fc*bw/2:  g(fdel) = ref --- solve this for a
    #
    # pi^2/a * fc^2 * bw^2 /4=-log(ref)
    a = -(PI * fc * bw) ** 2 / (4.0 * np.log(ref))

    if isinstance(t, str):
        if t == 'cutoff':  # compute cut_off point
            #  Solve exp(-a tc**2) = tref  for tc
            #   tc = sqrt(-log(tref) / a) where tref = 10^(tpr/20)
            if tpr >= 0:
                raise ValueError("Reference level for time cutoff must "
                                 "be < 0 dB")
            tref = pow(10.0, tpr / 20.0)
            return np.sqrt(-np.log(tref) / a)
        else:
            raise ValueError("If `t` is a string, it must be 'cutoff'")
        
    ygauss = np.exp(-a * t * t)
    yenv = 2.0 * a * ygauss * (2.0 * a * t**2 - 1.0)
    yI = ygauss * ((2.0 * a * (2.0 * a * t**2 - 1.0) - 4.0 * (PI * fc)**2) * np.cos(2 * PI * fc * t) +
                   8.0 * PI * a * fc * t * np.sin(2 * PI * fc * t))
    yQ = yenv * np.sin(2 * PI * fc * t)
    if not retquad and not retenv:
        return yI
    if not retquad and retenv:
        return yI, yenv
    if retquad and not retenv:
        return yI, yQ
    if retquad and retenv:
        return yI, yQ, yenv