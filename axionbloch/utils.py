"""General-purpose utilities for the axionbloch package.

Contents
--------
- Debugging helpers: :func:`check`
- Simple polynomial and lineshape functions: :func:`poly1`, :func:`poly2`,
  :func:`Lorentzian`, :func:`dualLorentzian`, :func:`tribLorentzian`
- Curve-fit estimators: :func:`estimateLorzfit`, :func:`estimatedualLorzfit`
- Unit-aware base class: :class:`PhysicalObject`
- Misc: :func:`giveDateAndTime`, :func:`sci_fmt`, :func:`save_phys_quantity`
"""

import inspect  # for check()
import os
import pickle
import re  # for check()
import sys
import time
import warnings
from functools import partial
from typing import Sequence

import h5py  # TODO make h5py an optional dependency
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from axionbloch.dependency import *


def getDateAndTime() -> str:
    """Return the current date and time as a compact string ``'YYYYMMDD_HHMMSS'``."""
    timestr = Time.now().datetime.strftime("%Y%m%d_%H%M%S")
    return timestr


def check(arg):
    """
    Print information of input arg

    Example
    ------
    import numpy as np

    a = np.zeros((2, 4))

    check(a)

    a+=1

    check(a)

    check(len(a))

    TERMINAL OUTPUT:

    casper-gradient-code\\testofcheckpoint.py @45 a : ndarray(array([[0., 0., 0., 0.], [0., 0., 0., 0.]])) [shape=(2, 4)]

    casper-gradient-code\\testofcheckpoint.py @47 a : ndarray(array([[1., 1., 1., 1.], [1., 1., 1., 1.]])) [shape=(2, 4)]

    casper-gradient-code\\testofcheckpoint.py @48 len(a) : int(2)

    casper-gradient-code\\testofcheckpoint.py @49 a.shape : tuple((2, 4)) [len=2]


    Copyright info:
    ------
    Adopted from https://gist.github.com/HaleTom/125f0c0b0a1fb4fbf4311e6aa763844b

    Author: Tom Hale

    Original comment: Print the line and filename, function call, the class, str representation and some other info
                    Inspired by https://stackoverflow.com/a/8856387/5353461



    """
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = "".join([line.strip() for line in context])
        m = re.search("check\\s*\\((.+?)\\)$", caller_lines)
        if m:
            caller_lines = m.group(1)
            position = (
                str(callerframeinfo.filename) + " line " + str(callerframeinfo.lineno)
            )

            # Add additional info such as array shape or string length
            additional = ""
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "" + caller_lines + " : ", end="")
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("check: couldn't find caller context")
    finally:
        del frame
        del callerframeinfo


def poly1(x, C0, C1):
    """Evaluate the linear polynomial ``C0 + C1*x``."""
    return C0 + C1 * x


def poly2(x, C0, C1, C2):
    """Evaluate the quadratic polynomial ``C0 + C1*x + C2*x^2``."""
    return C0 + C1 * x + C2 * x**2


def Lorentzian(x, center, FWHM, area: float = 1.0, offset: float = 0.0):
    """
    Return the value of the Lorentzian function
        offset + 0.5*FWHM*area / (np.pi * ( (x-center)**2 + (0.5*FWHM)**2 )      )

                           FWHM A
        offset + ───────────────────────
                  2π ((x-c)^2+(FWHM/2)^2 )

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    FWHM : scalar
        full width of half maximum (FWHM) / linewidth of the Lorentzian peak
    area : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    Null

    """
    return offset + 0.5 * np.abs(FWHM) * area / (
        np.pi * ((x - center) ** 2 + (0.5 * FWHM) ** 2)
    )


def Lorentzian_0edge(x, center, FWHM, area: float = 1.0, offset: float = 0.0):
    """
    Return the value of the Lorentzian function with values = 0 at edges
        offset + 0.5*FWHM*area / (np.pi * ( (x-center)**2 + (0.5*FWHM)**2 )      )

                           FWHM A
        offset + ───────────────────────
                  2π ((x-c)^2+(FWHM/2)^2 )

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    FWHM : scalar
        full width of half maximum (FWHM) / linewidth of the Lorentzian peak
    area : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    Null

    """
    return np.hamming(len(x)) * (
        offset
        + 0.5 * np.abs(FWHM) * area / (np.pi * ((x - center) ** 2 + (0.5 * FWHM) ** 2))
    )


def estimateLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    # check the length of datax and datay
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    datay_smoothed = datay.copy()
    if smooth:
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smoothed[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        for i in range(smoothlevel):
            datay_smoothed[i] = datay_smoothed[smoothlevel]
            datay_smoothed[-i] = datay_smoothed[-smoothlevel]
        if verbose:
            plt.figure()
            plt.plot(datax, datay, label="no smoothing")
            plt.plot(datax, datay_smoothed, label="after smoothing")
            plt.legend()
            plt.grid()
            plt.show()
    centerindex = np.argmax(datay_smoothed)
    center = datax[np.argmax(datay_smoothed)]
    # if centerindex<5 or abs(len(datay)-centerindex)<5:
    #     #print('signal peak is too close to the edge of range')
    #     raise ValueError('signal peak is too close to the edge of range')
    # amp = np.amax(datay)
    amp = datay[np.argmax(datay_smoothed)]
    amp_smoothed = np.amax(datay_smoothed)

    HMindexleft = 0
    HMindexright = len(datay) - 1
    HMindex = np.flatnonzero(
        (
            (datay_smoothed[1:] > amp_smoothed / 2.0)
            & (datay_smoothed[:-1] < amp_smoothed / 2.0)
        )
        | (
            (datay_smoothed[1:] < amp_smoothed / 2.0)
            & (datay_smoothed[:-1] > amp_smoothed / 2.0)
        )
    )
    # if verbose:
    #     print('HMindex.shape ', HMindex.shape)
    #     print('centerindex ', centerindex)
    #     print('HMindex \n', HMindex)
    for index in HMindex:
        if index <= centerindex and HMindexleft <= index:
            HMindexleft = index
        if index >= centerindex and index <= HMindexright:
            HMindexright = index
    if HMindexleft >= 1:
        HMindexleft -= 1
    if HMindexright <= (len(datax) - 2):
        HMindexright += 1
    gamma = np.average(
        [
            abs(datax[HMindexright] - datax[HMindexleft]),
            2 * abs(datax[centerindex] - datax[HMindexleft]),
            2 * abs(datax[centerindex] - datax[HMindexright]),
        ]
    )

    area = amp * np.pi * gamma / 2.0

    offset = 0
    if verbose:
        print(
            "estimateLorzfit [center, gamma, area, offset] ",
            [center, gamma, area, offset],
        )
    return [center, gamma, area, offset]


def dualLorentzian(
    x,
    center0,  # 0
    FWHM0,  # 1
    area0,  # 2
    center1,  # 3
    FWHM1,  # 4
    area1,  # 5
    offset,  # 6
):
    """
    Return the value of the dual-Lorentzian function

                        Γ0 A0                       Γ1 A1
        offset + ─────────────────────── + ───────────────────────
                  2(π(x-c0)^2+(Γ0/2)^2)     2(π(x-c1)^2+(Γ1/2)^2)

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the positions of two Lorentzian peaks
    gamma0 and gamma1 : scalar
        linewidth / full width of half maximum (FWHM) of Lorentzian peaks
    area0 and area1 : scalar
        areas under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve

    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    Null

    """
    return (
        offset
        + 0.5 * abs(FWHM0) * area0 / (np.pi * ((x - center0) ** 2 + (0.5 * FWHM0) ** 2))
        + 0.5 * abs(FWHM1) * area1 / (np.pi * ((x - center1) ** 2 + (0.5 * FWHM1) ** 2))
    )


def estimatedualLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for dual-Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    7 estimated parameters of the dual-Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    paras0 = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras1 = estimateLorzfit(
        datax=datax,
        datay=datay - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    return [
        paras0[0],
        paras0[1],
        paras0[2],
        paras1[0],
        paras1[1],
        paras1[2],
        paras0[3] + paras1[3],
    ]


def tribLorentzian(
    x,
    center0,  # 0
    gamma0,  # 1
    area0,  # 2
    center1,  # 3
    gamma1,  # 4
    area1,  # 5
    center2,  # 6
    gamma2,  # 7
    area2,  # 8
    offset,  # 9
):
    """
    Return the value of the trible-Lorentzian function

                        Γ0 A0                       Γ1 A1
        offset + ─────────────────────── + ───────────────────────
                  2(π(x-c0)^2+(Γ0/2)^2)     2(π(x-c1)^2+(Γ1/2)^2)

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the positions of two Lorentzian peaks
    gamma0 and gamma1 : scalar
        linewidth / full width of half maximum (FWHM) of Lorentzian peaks
    area0 and area1 : scalar
        areas under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve

    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    Null

    """
    return (
        offset
        + 0.5 * gamma0 * area0 / (np.pi * ((x - center0) ** 2 + (0.5 * gamma0) ** 2))
        + 0.5 * gamma1 * area1 / (np.pi * ((x - center1) ** 2 + (0.5 * gamma1) ** 2))
        + 0.5 * gamma2 * area2 / (np.pi * ((x - center2) ** 2 + (0.5 * gamma2) ** 2))
    )


def estimatetribLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for dual-Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    7 estimated parameters of the dual-Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    paras0 = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras1 = estimateLorzfit(
        datax=datax,
        datay=datay - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras2 = estimateLorzfit(
        datax=datax,
        datay=datay
        - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3])
        - Lorentzian(datax, paras1[0], paras1[1], paras1[2], paras1[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    return [
        paras0[0],
        paras0[1],
        paras0[2],
        paras1[0],
        paras1[1],
        paras1[2],
        paras2[0],
        paras2[1],
        paras2[2],
        paras0[3] + paras1[3],
    ]


def Gaussian(x, center, sigma, area, offset):
    """
    Return the value of the Gaussian function

                           area                 1  (x-center)^2
        offset + ─────────────────────── exp(- ─── ──────────────)
                    sigma * sqrt(2 Pi)          2     sigma^2

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    sigma : scalar
        variance of Gaussian function. FWHM = 2.35482 sigma, FWTM = 4.29193 sigma
    area : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    https://en.wikipedia.org/wiki/Gaussian_function

    """
    return offset + area / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * (x - center) ** 2 / sigma**2
    )


def estimateGaussfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    fitparas = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    fitparas[1] /= 2.35482
    return fitparas


def dualGaussian(
    x,
    center0,  # 0
    sigma0,  # 1
    area0,  # 2
    center1,  # 3
    sigma1,  # 4
    area1,  # 5
    offset,  # 6
):
    """
    Return the value of the Gaussian function

                           area0                 1 (x-center0)^2              area1                 1  (x-center1)^2
        offset + ─────────────────────── exp(- ─── ──────────────) + ─────────────────────── exp(- ─── ──────────────)
                    sigma0 * sqrt(2 Pi)          2    sigma0^2         sigma1 * sqrt(2 Pi)          2    sigma1^2

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the position of the Lorentzian peak
    sigma0 and sigma1 : scalar
        variance of Gaussian function. FWHM = 2.35482 sigma, FWTM = 4.29193 sigma
    area0 and area1 : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
    https://en.wikipedia.org/wiki/Gaussian_function

    """
    return (
        offset
        + area0
        / (sigma0 * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * (x - center0) ** 2 / sigma0**2)
        + +area1
        / (sigma1 * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * (x - center1) ** 2 / sigma1**2)
    )


def estimatedualGaussFit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    fitparas = estimatedualLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    fitparas[1] /= 2.35482
    fitparas[4] /= 2.35482
    return fitparas


def PolyEven(x, C0, C2, C4, C6, C8, C10, center, verbose=False):
    """
    Return the value of the polynomial
        C0 + C2 * (x-center)^2 + C4 * (x-center)^4 + C6 * (x-center)^6 + C8 * (x-center)^8 + C10 * (x-center)^10

    Parameters
    ----------
    x : scalar or array_like
        argument of the polynomial


    C0, C2, C4, C6, C8  : scalar

    verbose : bool
        the option for displaying assistive information

    Returns
    -------
    the value of the polynomial : scalar or array_like

    Examples
    --------
    >>>

    References
    ----------
    Null

    """
    return (
        C0
        + C2 * (x - center) ** 2
        + C4 * (x - center) ** 4
        + C6 * (x - center) ** 6
        + C8 * (x - center) ** 8
        + C10 * (x - center) ** 10
    )


def estimatePolyEvenfit(datax=None, datay=None, verbose=False):
    """
    Return the estimated parameters as the initial guess for even-oder polynomial curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    verbose : bool
        choose True to display processing information


    Returns
    -------
    the value of the even-oder polynomial function : ndarray or scalar

    Examples
    --------
    >>>

    References
    ----------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    center = datax[len(datax) // 2]
    C0 = datay[center]
    C2 = (datay[-1] - C0) / (datax[-1] - datax[center]) ** 2
    C4 = 0
    C6 = 0
    C8 = 0
    C10 = 0
    return [C0, C2, C4, C6, C8, C10, center]


def ExpCos(
    t=None,
    Amp=None,
    T2=None,
    nu=None,
    phi0=None,
    offset=None,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    return (
        Amp * np.exp(-(t - t[0]) / T2) * np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
        + offset
    )


def estimateExpCos(
    t=None,
    s=None,  # signal
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpCos1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offset = 0
    return [Amp, T2, nu, phi0, offset]


def ExpCosiSin(
    t=None,
    Amp=None,
    T2=None,
    nu=None,
    phi0=None,
    offsetx=None,
    offsety=None,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    return (
        Amp
        * np.exp(-(t - t[0]) / T2)
        * (
            np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
            + 1j * np.sin(2 * np.pi * nu * (t - t[0]) + phi0)
        )
        + offsetx
        + 1j * offsety
    )


def ExpCosiSinResidual(
    params,
    t,
    s,
    # Amp,
    # T2,
    # nu,
    # phi0,
    # offsetx,
    # offsety,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    Amp, T2, nu, phi0, offsetx, offsety = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    return np.abs(
        s
        - Amp
        * np.exp(-(t - t[0]) / T2)
        * (
            np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
            + 1j * np.sin(2 * np.pi * nu * (t - t[0]) + phi0)
        )
        - offsetx
        - 1j * offsety
    )


def estimateExpCosiSin(
    t=None,
    s=None,
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpCos1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offsetx = 0
    offsety = 0
    return [Amp, T2, nu, phi0, offsetx, offsety]


def dualExpCos(
    t=None,
    Amp_0=None,
    T2_0=None,
    nu_0=None,
    phi0_0=None,
    Amp_1=None,
    T2_1=None,
    nu_1=None,
    phi0_1=None,
    offset=None,
    verbose=False,
):
    """
    Two Exponentially decay cos waves

    """
    return (
        Amp_0
        * np.exp(-(t - t[0]) / T2_0)
        * np.cos(2 * np.pi * nu_0 * (t - t[0]) + phi0_0)
        + Amp_1
        * np.exp(-(t - t[0]) / T2_1)
        * np.cos(2 * np.pi * nu_1 * (t - t[0]) + phi0_1)
        + offset
    )


def estimatedualExpCos(
    t=None,
    s=None,
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpSin1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offset = 0
    return [Amp / 2, T2, nu, phi0, Amp / 2, T2, abs(nu - 3), phi0, offset]


def tribExpCos(
    t=None,
    Amp_0=None,
    T2_0=None,
    nu_0=None,
    phi0_0=None,
    Amp_1=None,
    T2_1=None,
    nu_1=None,
    phi0_1=None,
    Amp_2=None,
    T2_2=None,
    nu_2=None,
    phi0_2=None,
    offset=None,
    verbose=False,
):
    """
    Three Exponentially decay sin waves

    """
    return (
        Amp_0 * np.exp(-t / T2_0) * np.sin(2 * np.pi * nu_0 * t + phi0_0)
        + Amp_1 * np.exp(-t / T2_1) * np.sin(2 * np.pi * nu_1 * t + phi0_1)
        + Amp_2 * np.exp(-t / T2_2) * np.sin(2 * np.pi * nu_2 * t + phi0_2)
        + offset
    )


def expdecaywindow(num: int, decayfactor: float = 0.003, verbose=False):
    """
    Returns a exponentially-decaying array for the windowing

    Parameters
    ----------
    num : int
        Signal from lock-in amplifier and also the processed data
    decayfactor : float
        factor for deciding the decaying rate.
        optimal decayfactor should be chosen as 1/(T2star * samprate)
    verbose : bool
        Choose True to display processing information. Defaults to False.

    Returns
    -------
    An array for windowing

    [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))]

    df stands for decayfactor

    Examples
    --------
    >>>

    References
    ----------
    http://sopnmr.blogspot.com/2016/01/processing-window-functions.html
    """
    if decayfactor <= 0:
        raise ValueError("decayfactor <= 0")
    n_arr = np.arange(num, dtype=float)
    index_arr = (-1) * decayfactor * n_arr
    window = np.exp(index_arr)  # (1 - np.exp(decayfactor / num)) *
    if verbose:
        plt.figure()
        plt.scatter(n_arr, window)
        plt.plot(n_arr, window, label=f"exp({decayfactor:.1f} i / {num:d})")
        plt.xlabel("Number")
        plt.ylabel("Window")
        plt.title(
            "Exponentially decay window\nsum of window values = %.2f" % np.sum(window)
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.grid()
        plt.show()
    return window


def LIAFilterHomega(
    datax=None,
    datay=None,
    frequency=None,
    taun=None,
    order=None,
):
    """
    Return the complex array H(ω) for digital filter correction
        H(ω) = 1 / (1 + 2 * np.pi * 1j * frequency * taun) ** order

    Parameters
    ----------

    frequency : scalar
        Dmodualtor frequency of lock-in amplifier.
    taun : scalar
        taun equals TC (Time constant of the exponential running average filter).
    order : scalar
        Number of cascaded digital filters.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    Complex array H(ω) : ndarray

    Examples
    --------
    >>>

    References
    ----------
        Zurich Instruments, MFIA User Manual, Page 275, 6.4.1. Discrete-Time RC Filter
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    return 1.0 / (1.0 + 1j * 2.0 * np.pi * frequency * taun) ** order


def LIAFilterHomegaSquared(
    datax=None,
    datay=None,
    frequency=None,
    taun=None,
    order=None,
):
    return 1.0 / ((1.0 + (2.0 * np.pi * frequency * taun) ** 2.0) ** (order / 2.0))


def LIAFilterFunction(
    x=None,
    tau=None,
    order=None,
):
    return 1.0 / ((1.0 + 2.0 * np.pi * x * tau) ** order)  # no j


def LIAFilterHomegaSquared1(
    frequency=None,
    taun=None,
    # n=None,
    dmodfreq=None,
    a=None,
):
    return a / (
        (1.0 + (2.0 * np.pi * (frequency - dmodfreq) * taun) ** 2.0) ** (8.0 / 1.0)
    )


def LIAFilterHomegaSquared2(
    frequency=None,
    taun=None,
    # n=None,
    dmodfreq=None,
    a=None,
):
    return a / (
        (1.0 + (2.0 * np.pi * (frequency - dmodfreq) * taun) ** 2.0) ** (8.0 / 2.0)
    )


def LIAFilterPSD(frequency=None, taun=None, order=None, verbose=False):
    """
    Return the absolute value of complex array H(ω) for digital filter correction
        np.abs(H(ω)) = np.abs( 1 / (1 + 2 * np.pi * 1j * frequency * taun) ** order )

    Parameters
    ----------

    frequency : scalar
        Dmodualtor frequency of lock-in amplifier.
    taun : scalar
        taun equals TC (Time constant of the exponential running average filter).
    order : scalar
        Number of cascaded digital filters.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    Absolute value of complex array H(ω) : ndarray

    Examples
    --------
    >>>

    References
    ----------
        Zurich Instruments, MFIJ User Manual, Page 275, 6.4.1. Discrete-Time RC Filter
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    return np.abs(1.0 / (1.0 + 1j * 2.0 * np.pi * frequency * taun) ** order) ** 2.0


def stdPSD(
    data=None,
    samprate=None,  # in Hz
    windowfunction="rectangle",  # Hanning, Hamming, Blackman
    decayfactor=-10,
    verbose=False,
):
    """
    Return the frequency bin centers and power spectral density

    Parameters
    ----------
    data : 1-D array_like
        Time-series.

    samprate : float
        Sampling rate for the time-series.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    decayfactor : float
        parameter for window function 'expdecay'.
        The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
        where df stands for decayfactor.
        Defaults to -10.

    verbose : bool
        Choose True to display processing information. Defaults to False.

    Returns
    -------
    np.sort(frequencies)[len(frequencies)//2:] : 1-D array
        Frequency bin centers sorted by its values. Only the right half of the bin centers will be passed.

    (PSD[np.argsort(frequencies)])[len(frequencies)//2:] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.
        Only the right half of the array will be passed.

    Examples
    --------
    >>>

    References
    ----------

    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    """

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies = np.fft.fftfreq(
        len(data), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)

    # FFT and PSD
    FFT = np.fft.fft(data * window_arr, norm=None)
    PSD = 2.0 * np.abs(FFT / filtercomp) ** 2 / (S2 * samprate)

    return (
        np.sort(frequencies)[len(frequencies) // 2 :],
        (PSD[np.argsort(frequencies)])[len(frequencies) // 2 :],
    )


def stdLIAPSD(
    data_x: np.ndarray = None,
    data_y: np.ndarray = None,
    samprate: float = None,
    demodfreq: float = None,
    attenuation: str = 0,
    windowfunction: str = "rectangle",
    decayfactor: float = -10.0,
    showwindow=False,
    DTRCfilter: str = "off",
    DTRCfilter_TC: float = 1e-6,
    DTRCfilter_order: float = 8,
    verbose: bool = False,
):
    """
    Return the frequency bin centers and power spectral density
        This function processes data from MFIA lock-in amplifier.

    Parameters
    ----------
    data_x, data_y : 1-D array_like
        Time-series data of two output channels.

    samprate : float
        Sampling rate for the time-series in [Hz].

    dfreq : float
        Demodulator frequency of the lock-in amplifier in [Hz].
        Defaults to None.

    attenuation : float
        Attenuation of the data in terms of power ratio (in the unit of dB).
        Power ratio (10^(attenuation/10)).
        Positive value means signal was attenuated.
        e.g. an attenuation of 6 means 10^(6/10) = 3.981 ≈ 4
        Defaults to None.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    decayfactor : float, optional
        parameter for window function 'expdecay'.
        The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
        where df stands for decayfactor.
        Defaults to -10.

    showwindow : bool, optional
        option to plot window function array.
        Defaults to False.

    DTRCfilter : string, optional
        Whether the discrete time RC filter is on.
        Defaults to 'on'.

    DTRCfilter_TC : float, optional
        Time constant of the filter.
        Defaults to 1e-6.

    DTRCfilter_order : int / float, optional
        Order of the filter.
        Defaults to 8.

    verbose : bool, optional
        Choose True to display processing information.
        Defaults to False.

    Returns
    -------
    np.sort(frequencies) : 1-D array
        Absolute frequency bin centers sorted by its values.

    PSD[np.argsort(frequencies)] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.

    Examples
    --------
    >>>


    References
    ----------
    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    [2] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """

    # check array lengths
    assert len(data_x) == len(data_y)

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data_x))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies: np.ndarray = np.fft.fftfreq(
        len(data_x), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter == "on":
        filtercomp = LIAFilterHomega(
            frequency=frequencies,  # this frequency is [-samprate/2, samprate/2]
            taun=DTRCfilter_TC,
            order=DTRCfilter_order,
        )
    # check(filtercomp)

    # add demodulator frequency to get absolute frequency
    # Notice that this action should be not be done before LIAFilterHomega()
    # because this function accepts demodulated frequencies
    frequencies += demodfreq  #

    # show window array
    if showwindow:
        xstamp = np.arange(len(data_x))
        fig = plt.figure(figsize=(12, 5))  #
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        # fig.subplots_adjust(top=0.91,bottom=0.11,left=0.08,right=0.96,hspace=0.0,wspace=0.25)
        LIAraw_ax = fig.add_subplot(gs[0, 0])
        # LIAYraw_ax = fig.add_subplot(gs[1, 0])
        LIAwindowed_ax = fig.add_subplot(gs[0, 1])
        # LIAYwindowed_ax = fig.add_subplot(gs[1, 1])
        (LIAXrawline,) = LIAraw_ax.plot(
            xstamp, data_x, label="LIA X output", color="tab:green", alpha=1
        )
        (LIAYrawline,) = LIAraw_ax.plot(
            xstamp, data_y, label="LIA Y output", color="tab:brown", alpha=1
        )
        #'tab:brown'
        LIAraw_ax.set_ylabel("Amplitude / a.u.")
        # LIAraw_ax.legend(loc='upper right')
        LIAraw_ax.grid(True)
        # LIAraw_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAraw_ax.tick_params(axis='x',bottom=False, labelbottom=False)
        LIAraw_ax2 = LIAraw_ax.twinx()
        (windowline,) = LIAraw_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAraw_ax2.set_ylabel("Window value")
        # LIAraw_ax2.legend(loc='upper right')
        # line1, = ax.plot([1, 2, 3], label='label1')
        # line2, = ax.plot([1, 2, 3], label='label2')
        LIAraw_ax2.legend(handles=[LIAXrawline, LIAYrawline, windowline])
        # bottom, top, left, right : bool : Whether to draw the respective ticks
        # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
        # imag_ax.plot(self.frequencies, self.avgFFT.imag, \
        #     label='Imaginary part of FFT', color='tab:orange')
        # imag_ax.set_ylabel('Amplitude / a.u.')
        # imag_ax.set_xlabel('Frequency / Hz')
        # imag_ax.legend(loc='upper right')
        # imag_ax.grid(True)
        # # imag_ax.tick_params(axis='y', left=False, labelleft=False)

        (LIAXwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_x * window_arr,
            label="LIA X windowed",
            color="tab:green",
            alpha=1,
        )
        (LIAYwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_y * window_arr,
            label="LIA Y windowed",
            color="tab:brown",
            alpha=1,
        )
        LIAwindowed_ax.set_ylabel("Amplitude / a.u.")
        # LIAwindowed_ax.legend(loc='upper right')
        LIAwindowed_ax.grid(True)
        # LIAwindowed_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAwindowed_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        LIAwindowed_ax2 = LIAwindowed_ax.twinx()
        (windowline,) = LIAwindowed_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAwindowed_ax2.set_ylabel("Window value")
        LIAwindowed_ax2.legend(handles=[LIAXwinline, LIAYwinline, windowline])

        # LIAwindowed_ax2.legend(loc='upper right')
        # phase_ax.plot(self.frequencies, np.angle(self.avgFFT, deg=True), \
        #     label='Phase of  FFT', color='tab:cyan')
        # phase_ax.set_ylabel('Phase / $\degree$')
        # phase_ax.set_xlabel('Frequency / Hz')
        # phase_ax.legend(loc='upper right')
        # phase_ax.grid(True)

        # if specxlim != None:
        #     LIAraw_ax.set_xlim(specxlim[0], specxlim[1])
        #     imag_ax.set_xlim(specxlim[0], specxlim[1])
        #     amp_ax.set_xlim(specxlim[0], specxlim[1])
        #     phase_ax.set_xlim(specxlim[0], specxlim[1])

        titletext = "Window array"  # 'All shots of '+
        fig.suptitle(titletext)  # , fontsize=8
        plt.tight_layout()
        plt.show()

    # FFT and PSD
    FFT = np.fft.fft((data_x + 1j * data_y) * window_arr, norm=None)
    PSD = (
        10.0 ** (attenuation / 10.0)
        * 1.0
        * np.abs(FFT / filtercomp) ** 2.0
        / (S2 * samprate)
    )

    # to check Parsvel theorem, use:
    # TSPower == np.mean(PSD) * samprate

    if verbose:
        print("attenuation ", attenuation)
        print("FFT.shape ", FFT.shape)
        print("filtercomp.shape ", filtercomp.shape)
        print("S2 ", S2)
        print("samprate ", samprate)

    return np.sort(frequencies), PSD[np.argsort(frequencies)]
    # return frequencies, PSD
    # return np.sort(frequencies), PSD


def stdLIAFFT(
    data_x: np.ndarray = None,
    data_y: np.ndarray = None,
    samprate: float = None,
    demodfreq: float = None,
    attenuation: str = None,
    windowfunction: str = "rectangle",
    decayfactor: float = -10.0,
    showwindow=False,
    DTRCfilter: str = "off",
    DTRCfilter_TC: float = 1e-6,
    DTRCfilter_order: float = 8,
    verbose: bool = False,
):
    """
    Return the frequency bin centers and FFT results.
        This function processes data from MFIA lock-in amplifier.

    Parameters
    ----------
    data_x, data_y : 1-D array_like
        Time-series data of two output channels.

    samprate : float
        Sampling rate for the time-series in [Hz].

    dfreq : float
        Demodulator frequency of the lock-in amplifier in [Hz].
        Defaults to None.

    attenuation : float
        Attenuation of the data in terms of power ratio (in the unit of dB).
        Power ratio (10^(attenuation/10)).
        Positive value means signal was attenuated.
        e.g. an attenuation of 6 means 10^(6/10) = 3.981 ≈ 4
        Defaults to None.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    showwindow : bool, optional
        option to plot window function array.
        Defaults to False.

    DTRCfilter : string, optional
        Whether the discrete time RC filter is on.
        Defaults to 'on'.

    DTRCfilter_TC : float, optional
        Time constant of the filter.
        Defaults to 1e-6.

    DTRCfilter_order : int / float, optional
        Order of the filter.
        Defaults to 8.

    verbose : bool, optional
        Choose True to display processing information.
        Defaults to False.

    Returns
    -------
    np.sort(frequencies) : 1-D array
        Absolute frequency bin centers sorted by its values.

    FFT[np.argsort(frequencies)] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.

    Examples
    --------
    >>>


    References
    ----------
    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    [2] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """

    # check array lengths
    assert len(data_x) == len(data_y)

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data_x))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies = np.fft.fftfreq(
        len(data_x), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter == "on":
        filtercomp = LIAFilterHomega(
            frequency=frequencies,  # this frequency is [-samprate/2, samprate/2]
            taun=DTRCfilter_TC,
            order=DTRCfilter_order,
        )
    # check(filtercomp)

    # show window array
    if showwindow:
        xstamp = np.arange(len(data_x))
        fig = plt.figure(figsize=(12, 5))  #
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        # fig.subplots_adjust(top=0.91,bottom=0.11,left=0.08,right=0.96,hspace=0.0,wspace=0.25)
        LIAraw_ax = fig.add_subplot(gs[0, 0])
        # LIAYraw_ax = fig.add_subplot(gs[1, 0])
        LIAwindowed_ax = fig.add_subplot(gs[0, 1])
        # LIAYwindowed_ax = fig.add_subplot(gs[1, 1])
        (LIAXrawline,) = LIAraw_ax.plot(
            xstamp, data_x, label="LIA X output", color="tab:green", alpha=1
        )
        (LIAYrawline,) = LIAraw_ax.plot(
            xstamp, data_y, label="LIA Y output", color="tab:brown", alpha=1
        )
        #'tab:brown'
        LIAraw_ax.set_ylabel("Amplitude / a.u.")
        # LIAraw_ax.legend(loc='upper right')
        LIAraw_ax.grid(True)
        # LIAraw_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAraw_ax.tick_params(axis='x',bottom=False, labelbottom=False)
        LIAraw_ax2 = LIAraw_ax.twinx()
        (windowline,) = LIAraw_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAraw_ax2.set_ylabel("Window value")
        # LIAraw_ax2.legend(loc='upper right')
        # line1, = ax.plot([1, 2, 3], label='label1')
        # line2, = ax.plot([1, 2, 3], label='label2')
        LIAraw_ax2.legend(handles=[LIAXrawline, LIAYrawline, windowline])
        # bottom, top, left, right : bool : Whether to draw the respective ticks
        # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
        # imag_ax.plot(self.frequencies, self.avgFFT.imag, \
        #     label='Imaginary part of FFT', color='tab:orange')
        # imag_ax.set_ylabel('Amplitude / a.u.')
        # imag_ax.set_xlabel('Frequency / Hz')
        # imag_ax.legend(loc='upper right')
        # imag_ax.grid(True)
        # # imag_ax.tick_params(axis='y', left=False, labelleft=False)

        (LIAXwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_x * window_arr,
            label="LIA X windowed",
            color="tab:green",
            alpha=1,
        )
        (LIAYwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_y * window_arr,
            label="LIA Y windowed",
            color="tab:brown",
            alpha=1,
        )
        LIAwindowed_ax.set_ylabel("Amplitude / a.u.")
        # LIAwindowed_ax.legend(loc='upper right')
        LIAwindowed_ax.grid(True)
        # LIAwindowed_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAwindowed_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        LIAwindowed_ax2 = LIAwindowed_ax.twinx()
        (windowline,) = LIAwindowed_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAwindowed_ax2.set_ylabel("Window value")
        LIAwindowed_ax2.legend(handles=[LIAXwinline, LIAYwinline, windowline])

        # LIAwindowed_ax2.legend(loc='upper right')
        # phase_ax.plot(self.frequencies, np.angle(self.avgFFT, deg=True), \
        #     label='Phase of  FFT', color='tab:cyan')
        # phase_ax.set_ylabel('Phase / $\degree$')
        # phase_ax.set_xlabel('Frequency / Hz')
        # phase_ax.legend(loc='upper right')
        # phase_ax.grid(True)

        # if specxlim != None:
        #     LIAraw_ax.set_xlim(specxlim[0], specxlim[1])
        #     imag_ax.set_xlim(specxlim[0], specxlim[1])
        #     amp_ax.set_xlim(specxlim[0], specxlim[1])
        #     phase_ax.set_xlim(specxlim[0], specxlim[1])

        titletext = "Window array"  # 'All shots of '+
        fig.suptitle(titletext)  # , fontsize=8
        plt.tight_layout()
        plt.show()

    # FFT and PSD
    FFT = np.fft.fft((data_x + 1j * data_y) * window_arr, norm=None)
    # PSD = 10.0 ** (attenuation / 10.) * 1.0 * np.abs(FFT / filtercomp) ** 2. / (S2 * samprate)
    FFT = (
        10.0 ** (attenuation / 20.0) * 1.0 * (FFT / filtercomp) / (S2 * samprate) ** 0.5
    )
    frequencies += demodfreq  #
    return np.sort(frequencies), FFT[np.argsort(frequencies)]


def DTRC_filter(
    signal,
    samprate: float,
    TC: float,
    order: int,
):
    """
    Discrete-Time RC Filter

    References
    ----------
    [1] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    Ts = 1.0 / samprate

    # signal_f = signal
    def DTRC_filter_1st(signal):
        sigal_f = np.exp(-Ts / TC) * signal[0:-1] + (1 - np.exp(-Ts / TC)) * signal[1:]
        return sigal_f

    for i in range(order):
        signal = DTRC_filter_1st(signal)

    return signal


def plotaxisfmt(x, y, format_string):
    return format_string.format(x)


def plotaxisfmt_ppm2MHz(x, y, format_string, referfreq):
    return format_string.format(1e-6 * (1e-6 * referfreq * x + referfreq))


def plotaxisfmt_ppm2Hz(x, y, format_string, referfreq):
    return format_string.format((1e-6 * referfreq * x + referfreq))


def plotaxisfmt_Hz2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (x / referfreq - 1))


def plotaxisfmt_MHz2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (1e6 * x / referfreq - 1))


def plotaxisfmt_linewidth2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (x / referfreq))


def axisfmt_C2K(x, y, format_string):
    return format_string.format(x + 273.15)


def axisfmt_K2C(x, y, format_string):
    return format_string.format(x - 273.15)


def MethanolCS2temp(
    CSval=None,
    CSunit="ppm",  # 'ppm' 'Hz'
    referfreq=1e6,  # in Hz
    tempunit="K",
):

    a = -23.832
    b = -29.46
    c = 403.0
    temp = a * CSval**2 + b * CSval + c
    if tempunit == "K":
        return temp
    elif tempunit == "C":
        return temp - 273.15
    else:
        raise ValueError("tempunit wrong")


# Formatter function to display 1×10^n style
def sci_fmt(x, pos):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))  # exponent
    coeff = x / 10**exp  # coefficient
    return "${:.0f}\\times 10^{{{}}}$".format(coeff, exp)


def Npole2station(
    theta_e=None,  # scalar
    phi_e=None,  # scalar
    theta_s=None,
    phi_s=None,
    verbose=False,
):
    """
    return in cartesian coordinates
    """
    x = np.sin(theta_e) * np.cos(theta_s) * np.cos(phi_e - phi_s) - np.cos(
        theta_e
    ) * np.sin(theta_s)
    y = np.sin(theta_e) * np.sin(phi_e - phi_s)
    z = np.sin(theta_e) * np.sin(theta_s) * np.cos(phi_e - phi_s) + np.cos(
        theta_e
    ) * np.cos(theta_s)
    if verbose:
        check(np.array([x, y, z]))
        check(np.vdot(np.array([x, y, z]), np.array([x, y, z])) ** 0.5)
    return np.array([x, y, z])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):  #
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def Init_3020sphere(ax, verbose=False):
    plt.gca().invert_yaxis()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))
    # draw the cooridnate frame
    a = Arrow3D(
        [0, 2],
        [0, 0],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [0, 1.4],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [0, 0],
        [0, 1.3],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)

    ax.text(0.8, 1.55, 0, "y", color="black")
    ax.text(2.4, 0.35, 0, "x", color="black")
    ax.text(0, 0.05, 1.25, "z", color="black")
    # draw the sphere
    r = 1
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    # draw B0
    a = Arrow3D(
        [0, 0],
        [-0.95, -0.95],
        [0.75, 1.25],
        mutation_scale=10,
        lw=1.6,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    ax.text(0, -0.85, 1.15, "$\\mathbf{B}_0$", color="black")
    # ax.text(1, 0.85, 1.25, '$\\mathbf{M}$', color='g')

    # draw magnetization vectors
    # timestamp = np.linspace(start=0, stop=1, num=1000)
    # magz = np.cos(2*np.pi*nu/10*timestamp)
    # magx = np.sqrt(1 - magz**2) * np.cos(2*np.pi*nu*1*timestamp)
    # magy = np.sqrt(1 - magz**2) * np.sin(2*np.pi*nu*1*timestamp)
    # ax.quiver(
    #         0, 0, 0, # <-- starting point of vector
    #         1, 1, 1, # <-- directions of vector
    #         color = 'g', alpha = 1, lw = 1.6, length=1, normalize=False,
    #         arrow_length_ratio=.25, label='$\\vec{M}$'
    #     )
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass
    ax.set_xlim3d([-0.8, 0.99])
    ax.set_ylim3d([-0.8, 0.99])
    ax.set_zlim3d([-0.8, 0.99])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.axis("off")
    # ax.legend(loc='upper right')
    ax.set_box_aspect((1, 1, 1))


def Init_0090sphere(ax, verbose=False):
    plt.gca().invert_yaxis()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))
    # draw the cooridnates
    # draw the cooridnate frame
    a = Arrow3D(
        [-1, 1.2],
        [0, 0],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [-1, 1.2],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    # a = Arrow3D([0,0],[0,0],[0,1.3], mutation_scale=10, lw=1, arrowstyle="->", color="k", shrinkA=0, shrinkB=0)
    # ax.add_artist(a)
    ax.text(1.2, 0.15, 0, "x", color="black")
    ax.text(0.15, 1.2, 0, "y", color="black")

    # ax.text(0, 0.05, 1.25, 'z', color='black')
    # draw the sphere
    r = 1
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass
    ax.set_xlim3d([-0.8, 0.99])
    ax.set_ylim3d([-0.8, 0.99])
    ax.set_zlim3d([-0.8, 0.99])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.axis("off")
    # ax.legend(loc='upper right')
    ax.set_box_aspect((1, 1, 1))


def Add_vector(
    ax,
    start=None,
    end=None,
    mutation_scale=10,
    lw=1.6,
    color="k",
    alpha=1,
    zorder=5,
    linestyle="-",
    verbose=False,
):
    a = Arrow3D(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        mutation_scale=mutation_scale,
        lw=lw,
        arrowstyle="->",
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        linestyle=linestyle,
    )
    ax.add_artist(a)

    # ax.quiver(
    #         start[0], start[1], start[2], # <-- starting point of vector
    #         end[0], end[1], end[2], # <-- directions of vector
    #         color = 'g', alpha = 1, lw = linewidth, length=1, normalize=False,
    #         arrow_length_ratio=.25, label=''
    #     )


# def sanCheck(arr, tag: str = None):
#     print("")
#     if tag is not None:
#         print(tag)
#     print("shape =", arr.shape)
#     print("mean =", arr.mean())
#     print("std =", arr.std())
#     print("")


def sanCheck(var, tag: str = None):
    print("")
    if tag is not None:
        print(tag)

    # Convert to numpy array only for checking NaN
    arr = np.asarray(var)

    # Warn if any NaN
    if np.isnan(arr).any():
        print("Variable contains NaN values!")

    # Scalar case
    if np.isscalar(var):
        print("(scalar) value =", var)
        return

    # Array case
    print("shape =", arr.shape)
    print("mean =", np.nanmean(arr))
    print("std =", np.nanstd(arr))
    print("min =", np.nanmin(arr))
    print("max =", np.nanmax(arr))
    print("")


okabe_ito_colors = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    # "#F0E442",  # yellow
    # "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

tab10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

high_contrast_extended = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
    "#969696",
    "#dd1c77",
]

vivid_colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

dark_contrast = [
    "#0b3c5d",
    "#b82601",
    "#1c6e8c",
    "#2f4858",
    "#6a994e",
    "#bc4749",
    "#3a0ca3",
    "#4361ee",
]

soft_contrast = [
    "#a6cee3",
    "#fdbf6f",
    "#b2df8a",
    "#fb9a99",
    "#cab2d6",
    "#ffff99",
    "#1f78b4",
    "#33a02c",
]

grayscale_safe = ["#000000", "#444444", "#888888", "#bbbbbb"]

linestyles = ["-", "--", "-.", ":"]
markers = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]

# def get_ALP_wind(
#     year=None,
#     month=None,
#     day=None,
#     time=None,
#     lat=None,
#     lon=None,
#     elev=None,
#     verbose=False,
# ):
#     """
#     returns the velocity 'v_lab' between lab frame and DM halo (SHM), in the galactic rest frame, for the specified coordinates and time
#     returns the angle [rad] between the CASPEr projection axis (zenith) and 'v_lab'

#     time: needs to be in the format "15:47:18"
#         if none is specified, use current time

#     lat: latitude of experiment location
#         if none is specified, use Mainz: 49.9916 deg north

#     lon: longitude of experiment location
#         if none is specified, use Mainz: 8.2353 deg east

#     elev: height of experiment location
#         if none is specified, use Uni Campus Mainz: 130 m
#     """
#     if verbose:
#         print("now calculating wind angle")

#     # assert None is not in []
#     CASPEr_lat = 49.9916  # degrees north
#     CASPER_lon = 8.2353  # degrees east
#     CASPER_elevation = 130  # meters

#     if lat is None:
#         lat = CASPEr_lat
#         if verbose:
#             print(
#                 f"no latitute input provided, using CASPEr-Mainz location: {CASPEr_lat}"
#             )
#     else:
#         if verbose:
#             print(f"latitute: {lat}")
#     if lon is None:
#         lon = CASPER_lon
#         if verbose:
#             print(
#                 f"no longitude input provided, using CASPEr-Mainz location: {CASPER_lon}"
#             )
#     else:
#         if verbose:
#             print(f"longitude: {lon}")
#     if elev is None:
#         elev = CASPER_elevation
#         if verbose:
#             print(
#                 f"no elevation input provided, using CASPEr-Mainz location: {CASPER_elevation}"
#             )
#     else:
#         if verbose:
#             print(f"elevation: {elev}")

#     if (year or month or day or time) is None:
#         timedate_DMmeasure = Time.now()
#         print(
#             f"no date and time input provided, using current date and time: {timedate_DMmeasure}"
#         )
#     else:
#         timedate_DMmeasure = rf"{year}-{month}-{day}T{time}"
#     if verbose:
#         print(f"time input: {timedate_DMmeasure}")

#     timeastro = Time(timedate_DMmeasure, format="isot", scale="utc")
#     DMtimefrac = wind.FracDay(Y=2022, M=12, D=23)
#     if verbose:
#         print("time of DM measurement (fractional days): ", DMtimefrac)

#     LABvel = wind.ACalcV(DMtimefrac)
#     if verbose:
#         print("velocity (lab frame) @DM time: ", LABvel)

#     DMtime, unit_North, unit_East, unit_Up, Vhalo = wind.get_CASPEr_vect(
#         time=timeastro,
#         lat=CASPEr_lat,
#         lon=CASPER_lon,
#         elev=CASPER_elevation,
#     )

#     # print(type(Vhalo))
#     Vlab = Vhalo.get_d_xyz()  # convert into a vector
#     Bz = (
#         unit_Up.get_xyz()
#     )  # our leading field is along zenith, normal to the local ground tangent plane

#     alpha_ALP = angle_between(Vlab, Bz).value
#     v_ALP = np.linalg.norm(Vlab.value) * 1e3
#     v_ALP_perp = v_ALP * np.sin(alpha_ALP)

#     if verbose:
#         # print("time of DM measurement: ", DMtime)
#         print("Bz vector @DM time (galaxy frame):", Bz)
#         print("v_halo @DM time (galaxy frame):", Vhalo)
#         print("v_lab @DM time:", Vlab)
#         print("angle between projection axis & lab velocity @DM time: ", alpha_ALP)

#     ###############################################################################################
#     # do not delete
#     return v_ALP, v_ALP_perp, alpha_ALP


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def MovAvgByStep(xstamp=None, rawsignal=None, weights=None, step_len=1, verbose=False):
    """
    A moving average with tunable step length, especially designed for axion signal search.

    Parameters
    ----------
    rawsignal : array
        raw signal
    weights : array
        The weights for doing the averaging
    step_len : int
        The step length for doing the moving average.
        Default to 1.
    verbose : bool
        It is here for no reason.

    Return
    ------
    np.array(prcdsiganl) : array
        The processed signal.
    """
    assert xstamp is not None
    assert rawsignal is not None
    assert weights is not None
    assert step_len is None or type(step_len) is int
    if step_len is None:
        step_len = 1
    if step_len < 1:
        raise ValueError("step_len < 1. Increase step_len.")

    step_size = step_len * abs(xstamp[1] - xstamp[0])
    # normalization of the template signal / weights
    # if not np.isclose([np.sum(weights)], [1.0], rtol=1e-05, atol=1e-06):
    #     print(f'Warning from {MovAvgByStep.__name__}: ' + \
    #           f'np.sum(weights) = {np.sum(weights)} != 1.0. '
    #           'The normalization of the weights is done anyway.')
    #     weights /= np.sum(weights)

    # processed signal
    prcd_xstamp = []
    prcd_siganl = []

    # calculate the number of steps
    numofstep = len(rawsignal) // step_len
    #
    for i in range(numofstep):
        if i * step_len + len(weights) > len(rawsignal):
            break
        prcd_siganl.append(
            np.vdot(rawsignal[i * step_len : i * step_len + len(weights)], weights)
        )
        prcd_xstamp.append([i * step_size + xstamp[0]])
    return np.array(prcd_xstamp), np.array(prcd_siganl)


def record_runtime_YorN(RECORD_RUNTIME):
    """
    A decorator to record the runtime of a function when RECORD_RUNTIME is True.
    """

    def record_runtime(func):
        def wrapper(*args, **kwargs):
            if RECORD_RUNTIME:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                clear_lines()
                print(
                    f"Function {func.__name__} took {end_time - start_time:.2g} (s) to run."
                )
                sys.stdout.flush()
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return record_runtime


def print_progress_bar(
    iteration,
    total,
    prefix="Progress",
    suffix="Complete",
    decimals=3,
    length=50,
    fill="█",
):
    percent = ("{0:." + str(decimals) + "f}").format(
        100.0 * ((iteration) / float(total))
    )
    filled_length = int(length * (iteration) // (total))
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    # write \n when the iteration finishes
    if (iteration) == total:
        sys.stdout.write("\n")


def update_info(info):
    sys.stdout.write(f"{info}")
    sys.stdout.flush()


def clear_lines():
    sys.stdout.write("\r\033[K")  # Move cursor up and clear the line
    sys.stdout.flush()  #


def exampleofprogress():
    # Example usage
    total = 100
    print("Starting the process...")
    sys.stdout.flush()
    # time.sleep(2)
    for i in range(total + 1):
        if i % 10 == 0:
            clear_lines()
            print(f"i = {i}, asdafdqw=")
            print(f"i = {i}, asdafdqw=")
            sys.stdout.flush()
        time.sleep(0.1)  # Simulate some work being done
        print_progress_bar(i, total, prefix="Progress", suffix="Complete", length=50)
        # time.sleep(0.1)  # Simulate some work being done

    sys.stdout.write("\n")  # Move to the next line after the progress bar is complete


def getFWHM(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a curve.

    Parameters:
        x (array-like): The x-values of the curve.
        y (array-like): The y-values of the curve.

    Returns:
        float: The FWHM of the curve.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Find the maximum value of y and its half-maximum
    y_max = np.max(y)
    half_max = y_max / 2.0

    # Find indices where y crosses the half-maximum
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # Interpolate to find more precise crossing points
    x_left = np.interp(
        half_max, [y[left_index - 1], y[left_index]], [x[left_index - 1], x[left_index]]
    )
    x_right = np.interp(
        half_max,
        [y[right_index], y[right_index + 1]],
        [x[right_index], x[right_index + 1]],
    )

    # Calculate FWHM
    fwhm = x_right - x_left
    return fwhm


def calculate_fwhm(x, y, peak=True):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a curve.
    Works for both peaks and dips.

    Parameters:
        x (array-like): The x-values of the curve.
        y (array-like): The y-values of the curve (can be positive or negative).
        peak (bool): If True, calculate FWHM for a peak (maximum). If False, calculate for a dip (minimum).

    Returns:
        float: The FWHM of the curve.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Determine the maximum or minimum value and the half-maximum
    if peak:
        y_extreme = np.max(y)
        half_max = y_extreme / 2.0
        indices = np.where(y >= half_max)[0]
    else:
        y_extreme = np.min(y)
        half_max = y_extreme / 2.0
        indices = np.where(y <= half_max)[0]

    # Check if the curve crosses the half-maximum value
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # Interpolate to find more precise crossing points
    if left_index > 0:
        x_left = np.interp(
            half_max,
            [y[left_index - 1], y[left_index]],
            [x[left_index - 1], x[left_index]],
        )
    else:
        x_left = x[left_index]

    if right_index < len(y) - 1:
        x_right = np.interp(
            half_max,
            [y[right_index], y[right_index + 1]],
            [x[right_index], x[right_index + 1]],
        )
    else:
        x_right = x[right_index]

    # Calculate FWHM
    fwhm = x_right - x_left
    return fwhm


def get_FWHM_indice(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a dip in the unit of indice.

    Parameters:
        x (array-like): The x-values of the curve.
        y (array-like): The y-values of the curve.

    Returns:
        float: The FWHM of the curve.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Find the maximum value of y and its half-maximum
    y_min = np.amin(y)
    Twice_min = y_min * 2.0

    # Find indices where y crosses the half-maximum
    # check(np.where(y <= Twice_min))
    indices = np.where(y <= Twice_min)[0]
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # # Interpolate to find more precise crossing points
    # x_left = np.interp(Twice_min, [y[left_index - 1], y[left_index]], [x[left_index - 1], x[left_index]])
    # x_right = np.interp(Twice_min, [y[right_index], y[right_index + 1]], [x[right_index], x[right_index + 1]])
    x_left = x[left_index]
    x_right = x[right_index]

    # Calculate FWHM
    FWHMin = abs(x_right - x_left)

    return FWHMin


class PhysicalObject:
    """
    Base class for physical objects with Quantity attributes.
    Automatically converts units and saves quantities to HDF5.
    """

    def __init__(self):
        self.quantities = {}
        self.generalQuantities = {}

    def useCommonUnits(self, verbose: bool = False):
        """
        Convert all Quantity attributes to their common units.
        Subclasses should define a dict `quantities` mapping attribute names
        to desired units.
        """
        assert hasattr(self, "quantities")

        for attr_name, unit in self.quantities.items():
            attr = getattr(self, attr_name, None)
            if isinstance(attr, Quantity):
                setattr(self, attr_name, attr.to(unit))
            elif attr is None:
                pass
            else:
                print(
                    "WARNING: the variable "
                    + attr_name
                    + " should be an instance of Quantity but it is not. "
                )

        if verbose:
            print(
                f"Converted quantities to common units: {list(self.quantities.keys())}"
            )

    # def saveToH5(self, pathAndName: str, h5_group_name: str, verbose: bool = False):
    #     """Save this object to an HDF5 file."""
    #     suffix = "" if pathAndName.endswith(".h5") else ".h5"
    #     with h5py.File(pathAndName + suffix, "w") as h5f:
    #         group = h5f.create_group(h5_group_name)
    #         self.saveToH5group(group)
    #     if verbose:
    #         print(f"Saved {self.__class__.__name__} to {pathAndName + suffix}")

    def saveToH5group(
        self,
        group: h5py.Group,
        verbose: bool = False,
    ):
        """Save all Quantity attributes to the HDF5 group."""
        assert hasattr(self, "quantities")
        assert hasattr(self, "generalQuantities")

        if verbose:
            print(
                f"[{self.__class__.__name__}.{self.saveToH5group.__name__}] self.quantities = ",
                self.quantities,
            )
            print(
                f"[{self.__class__.__name__}.{self.saveToH5group.__name__}] self.generalQuantities = ",
                self.generalQuantities,
            )

        self.useCommonUnits()

        # Save name if exists
        if hasattr(self, "name"):
            group.create_dataset(
                "name", data=["nameless" if self.name is None else self.name]
            )

        # Save all Quantity attributes
        for attr_name, unit in self.quantities.items():
            attr = getattr(self, attr_name, None)
            if isinstance(attr, Quantity):
                save_phys_quantity(
                    group=group, name=attr_name, value=attr.value, unit=attr.unit
                )

        dtype_map = {
            "float": np.float64,
            "int": np.int64,
            "bool": np.bool_,
            "str": h5py.string_dtype(encoding="utf-8"),
        }
        if verbose:
            print("self.quantities = ", self.quantities)
        for attr_name, dtype_str in self.generalQuantities.items():
            value = getattr(self, attr_name, None)
            if value is not None:
                if dtype_str not in dtype_map:
                    raise ValueError(f"Unsupported dtype '{dtype_str}' for {attr_name}")
                # Remove existing dataset if present
                if attr_name in group:
                    del group[attr_name]
                # save the attr
                dset = group.create_dataset(
                    name=attr_name,
                    data=value,
                    dtype=dtype_map[dtype_str],
                )

    def loadFromH5group(self, group):
        """
        Load all attributes listed in self.quantities and self.generalQuantities
        from an HDF5 group.
        """
        # load physical quantities
        for name, unit_expected in self.quantities.items():

            if name not in group:
                raise KeyError(f"Missing Quantity '{name}' in HDF5 group")

            subgroup = group[name]

            # Load stored value
            value = subgroup["value"][()]  # works for scalars & arrays

            # Load unit stored in file
            unit_stored = subgroup.attrs.get("unit", None)

            # Optional: consistency check
            if unit_expected is not None and unit_stored != unit_expected:
                print(
                    f"Warning: unit mismatch for {name}: "
                    f"{unit_stored} (file) vs {unit_expected} (expected)"
                )

            # Restore into the instance
            setattr(self, name, Quantity(value, unit_stored))
        # load general quantities
        for attr_name, dtype_str in self.generalQuantities.items():

            if attr_name in group:

                dset = group[attr_name]

                # Read scalar value
                value = dset[()]

                # Convert numpy scalars to native Python types
                if isinstance(value, np.generic):
                    value = value.item()

                setattr(self, attr_name, value)

    def saveToPkl(
        self,
        fileDir: str = "",
        fileName: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        """
        Save this instance to a pickle file.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.saveToPkl.__name__}]"
        if fileDir is None:
            raise ValueError("fileDir must not be None")

        if fileName is None:
            name = getattr(self, "name", None)
            base = name if name is not None else self.__class__.__name__.lower()
            fileName = base + "_" + getDateAndTime()
        if fileName.endswith(".pkl"):
            fileName = fileName[:-4]

        os.makedirs(fileDir, exist_ok=True)
        path = os.path.join(fileDir, f"{fileName}.pkl")

        while os.path.exists(path) and not overwrite:
            print(f"File already exists: {path}")
            new = input(
                "Enter a new filename (without .pkl) or press Enter to overwrite: "
            ).strip()
            if new == "":
                break
            fileName = new
            path = os.path.join(fileDir, f"{fileName}.pkl")

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"Saved object to {path}")

    def loadFromPkl(self, path: str, verbose: bool = False):
        """
        Load an instance of this class from a pickle file.
        """
        logPrefix = f"[{self.__name__}.{self.loadFromPkl.__name__}]"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, self):
            raise TypeError(f"Pickle contains {(obj)}, expected {self}")

        if verbose:
            print(f"Loaded object from {path}")

        return obj


def save_phys_quantity(
    group: h5py.Group,
    name: str,
    value: float | int | Sequence | np.ndarray,
    unit: str,
):
    """
    Save a variable with its value and unit into an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The parent HDF5 group where the dataset will be created.
    name : str
        The name of the subgroup to store this variable under.
    value : array-like
        The numerical data to store.
    unit : str, optional
        The unit of the data (default is "dimensionless").

    Returns
    -------
    subgroup : h5py.Group
        The created subgroup containing the datasets "value" and "unit".
    """
    subgroup = group.create_group(name)
    # subgroup.create_dataset("value", data=value)
    subgroup.create_dataset("value", data=[value] if np.isscalar(value) else value)
    subgroup.create_dataset("unit", data=[unit])
    return subgroup


def check_norm(x: np.ndarray, y: np.ndarray):
    """
    Check if the array `y` is normalized with respect to `x`.

    Parameters
    ----------
    x : np.ndarray
        Array of frequencies (or variable of integration).
    y : np.ndarray
        Array of function values (e.g., lineshape).

    Raises
    ------
    Warning
        If the integral of y over x is not close to 1.
    """
    integral = np.trapezoid(y, x)
    if hasattr(integral, "unit"):
        integral_value = integral.to_value(unit.one)
    else:
        integral_value = integral
    if not np.allclose(integral_value, 1.0, rtol=1e-3):
        warnings.warn(
            f"Array is not normalized! Integral = {integral}", category=UserWarning
        )


def coh_time_g1(x, dt):
    """
    x : complex-valued time series
    dt: sampling interval
    method: "1e" or "integral"
    """
    # TODO : check if the input x long enough (compared to the coherence time) for a reliable estimation of g1 and tau
    duration = len(x) * dt
    E = np.array(x)  # complex field
    N = len(E)

    tic = time.time()
    corr = np.correlate(E, E.conj(), mode="full")
    toc = time.time()
    print(f"Time taken for correlation: {toc - tic:.3f} seconds")

    corr = corr[N - 1 :]
    check(corr.std())
    check(corr[0])
    check(corr[1])
    check(corr[2])
    check(np.sum(np.abs(x) ** 2))
    g1 = corr / np.sum(np.abs(x) ** 2)  # positive delays only

    fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
    gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(g1.real, label="real part")
    ax00.plot(g1.imag, label="imaginary part")
    # ax00.set_xlabel("time (s)")
    ax00.set_ylabel("g1 (arb. units)")
    ax00.legend()
    fig.suptitle("", wrap=True)
    plt.tight_layout()
    plt.show()

    tau = 2 * np.sum(np.abs(g1) ** 2) * dt
    if tau > duration:
        warnings.warn(
            f"Estimated coherence time tau = {tau:.3e} s is longer than the total duration of the time series ({duration:.3e} s). "
            "The estimation may be unreliable. Consider using a longer time series.",
            category=UserWarning,
        )
    if tau <= dt:
        warnings.warn(
            f"Estimated coherence time tau = {tau:.3e} s is not greater than the sampling interval dt = {dt:.3e} s. "
            "The estimation may be unreliable. Consider using a shorter sampling interval.",
            category=UserWarning,
        )
    return tau


def boltzmann_probabilities(energies: Sequence[Quantity], T: Quantity) -> np.ndarray:
    """
    Compute Boltzmann probabilities for a set of energy eigenstates.

    Parameters
    ----------
    energies : sequence of Quantity
        energies with units of energy
    T : Quantity
        Temperature (must be > 0)

    Returns
    -------
    np.ndarray
        Dimensionless probabilities
    """

    assert T.to_value(unit.K) > 0

    # energies = np.array(energies)
    # energies = np.array([E.to(energies[0].unit) for E in energies])
    energies_eV = np.array([E.to_value(unit.eV) for E in energies])

    # beta = 1 / (kB * T)
    beta_eV_1 = (1.0 / (const.kB * T)).to_value(unit.eV ** (-1))

    E_min = energies_eV.min()
    scaled_energies_eV = [E - E_min for E in energies_eV]

    exponents = np.array([(-beta_eV_1 * E) for E in scaled_energies_eV])

    weights = np.exp(exponents)
    probabilities = weights / np.sum(weights)

    return probabilities


def deBroglie_wavelength(mass: Quantity, speed: Quantity) -> Quantity:
    """
    Calculate the de Broglie wavelength of a particle given its mass and speed.
    Here we adopt SI units.
    """
    mass = mass.to(unit.kg)
    speed = speed.to("km/s")
    gamma = 1 / np.sqrt(1 - (speed.to_value(unit.km / unit.s) / const.c)) ** 2
    lambda_db = (const.h / (gamma * mass * speed)).to(unit.m)
    return lambda_db
