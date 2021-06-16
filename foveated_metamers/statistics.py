#!/usr/bin/env python3
"""Compute image statistics."""

import pyrtools as pt
import numpy as np
import pandas as pd
import plenoptic as po
import scipy
import xarray
from collections import OrderedDict


def heterogeneity(im, kernel_size=16, kernel_type='gaussian', pyramid_height=4):
    """Compute heterogeneity statistic.

    1. Convolve image with Laplacian pyramid of specified height.
    2. Rectify pyramid outputs.
    3. Compute local mean and variance of rectified pyramid outputs using a
       filter of specified type and size.
    4. Heterogeneity equals the local variance divided by the local mean; lower
       values are less homogeneous.

    Heterogeneity is computed at each scale of the pyramid.

    Parameters
    ----------
    im : np.ndarray
        2d grayscale image to compute heterogeneity on.
    kernel_size : int, optional
        Size of the kernel.
    kernel_type : {'gaussian', 'square'}, optional
        Which type of kernel to use.
    pyramid_height : int, optional
        Height of the Laplacian pyramid and thus the number of scales we
        compute statistics at.

    Returns
    -------
    heterogeneity_map : List[np.ndarray]
        Maps (same size as ``im``) of heterogeneity stat across the image, at
        each scale (note that size of each dimension decreases by half for each
        scale).
    homoegeneity_stats : pd.DataFrame
        DataFrame containing the average heterogeneity across the image, at each
        scale.

    """
    pyr = pt.pyramids.LaplacianPyramid(im, pyramid_height)
    # rectify the coefficients
    pyr_coeffs = [c.clip(min=0) for c in pyr.pyr_coeffs.values()]

    # both kernels are normalized so that their sum is 1 (and thus, they act as
    # averaging filters)
    if kernel_type == 'square':
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    elif kernel_type == 'gaussian':
        x = np.linspace(-4, 4, kernel_size)
        x, y = np.meshgrid(x, x)
        x = np.sqrt(x**2 + y**2)
        kernel = np.exp(-x**2/2)
        kernel = kernel / kernel.sum()

    means = [scipy.signal.convolve2d(c, kernel, mode='same') for c in pyr_coeffs]
    variances = [scipy.signal.convolve2d(np.square(c - m), kernel, mode='same')
                 for c, m in zip(pyr_coeffs, means)]
    heterogeneity = [v/m for v, m in zip(variances, means)]

    df = pd.DataFrame({i: h.mean() for i, h in enumerate(heterogeneity)}, [0])
    df = df.melt(var_name='scale', value_name='heterogeneity')
    size = pd.Series({i: kernel_size*2**i for i in range(pyramid_height)},
                     name='kernel_size')
    df = df.merge(size, left_on='scale', right_index=True)
    return heterogeneity, df


def amplitude_spectra(image):
    """Compute amplitude spectra of an image.

    We compute the 2d Fourier transform of an image, take its magnitude, and
    then radially average it. This averages across orientations and also
    discretizes the frequency.

    Parameters
    ----------
    image : np.ndarray
        The 2d array containing the image

    Returns
    -------
    spectra : np.ndarray
        The 1d array containing the amplitude spectra

    Notes
    -----
    See
    https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html
    for how we compute the radial mean. Note the tutorial excludes label=0, but
    we include it (corresponds to the DC term).

    """
    frq = scipy.fft.fftshift(scipy.fft.fft2(image))
    # following
    # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html.
    # Note the tutorial excludes label=0, but we include it (corresponds to the
    # DC term).
    rbin = pt.synthetic_images.polar_radius(frq.shape).astype(np.int)
    spectra = scipy.ndimage.mean(np.abs(frq), labels=rbin,
                                 index=np.arange(rbin.max()+1))
    return spectra


def image_set_amplitude_spectra(images, names, metadata=OrderedDict(),
                                name_dim='image_name'):
    """Compute amplitude spectra of a set of images.

    All images must be same size.

    Parameters
    ----------
    images : list
        This should be a list of 2d image arrays (all the same size) or of
        strings giving the paths to such images.
    names : list
        List of strings, same length as images, giving the names of the images,
        which we use to label them in the xarray.Dataset we create.
    metadata: OrderedDict, optional
        OrderedDict of extra coordinates to add to data (e.g., the model name).
        Should be an OrderedDict so we get the proper ordering of dimensions.
    name_dim : str, optional
        The name of the coordinates to label with the names list.

    Returns
    -------
    spectra : xarray.Dataset
        Dataset containing the spectra of each image.

    """
    spectra = []
    for im in images:
        if isinstance(im, str):
            im = po.to_numpy(po.load_images(im)).squeeze()
        spectra.append(amplitude_spectra(im))
    for k, v in metadata.items():
        if isinstance(v, str) or not hasattr(v, '__iter__'):
            metadata[k] = [v]
    metadata.update({name_dim: names,
                     'freq_n': np.arange(len(spectra[-1]))})
    # add extra dimensions to the front of spectra for metadata.
    spectra = np.expand_dims(spectra,
                             tuple(np.arange(len(metadata.keys())-2)))
    data = xarray.DataArray(spectra, metadata, metadata.keys(),
                            name='sf_amplitude')
    return data.to_dataset()
