#!/usr/bin/env python3
"""Compute image statistics."""

import pyrtools as pt
import numpy as np
import pandas as pd
import scipy


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
