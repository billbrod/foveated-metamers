#!/usr/bin/env python3
"""Compute image statistics."""

import pyrtools as pt
import numpy as np
import pandas as pd
import plenoptic as po
import scipy
from scipy import fft as sp_fft
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
    discretizes the frequency. We also drop a disk in frequency space to
    exclude the highest frequencies (that is, those where we don't have
    cardinal directions).

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
    frq = sp_fft.fftshift(sp_fft.fft2(image))
    # following
    # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html.
    # Note the tutorial excludes label=0, but we include it (corresponds to the
    # DC term).
    rbin = pt.synthetic_images.polar_radius(frq.shape).astype(np.int)
    # we ignore all frequencies outside a disk centered at the origin that
    # reaches to the first edge (in frequency space). This means we get all
    # frequencies that we can measure in each orientation (you can't get any
    # frequencies in the cardinal directions beyond this disk)
    frq_disk = pt.synthetic_images.polar_radius(frq.shape)
    frq_thresh = min(frq.shape)//2
    frq_disk = frq_disk < frq_thresh
    rbin[~frq_disk] = rbin.max()+1
    spectra = scipy.ndimage.mean(np.abs(frq), labels=rbin,
                                 index=np.arange(frq_thresh-1))
    return spectra


def amplitude_orientation(image, n_angle_slices=32, metadata=OrderedDict()):
    """Compute orientation energy of an image.

    We compute the 2d Fourier transform of an image, take its magnitude, and
    compile the amplitudes in angular slices. Note that, unlike
    amplitude_spectra(), we do not average within these slices to get a single
    number. That's because the distributions here can have much larger outliers
    -- whichever slice gets the DC term, for example, will have a way higher
    average energy, but that's spurious and a reflection of the pixel grid's
    alignment rather than anything meaningful. Right now, it's recommended to
    use median to summarize these, but all values are returned.

    We also drop a disk in frequency space to exclude the highest frequencies
    (that is, those where we don't have cardinal directions).

    Note that we don't window the image before taking the Fourier transform,
    and thus there may be extra vertical and horizontal energy from boundary
    artifacts. Thus, this should only be considered "relative" orientation
    energy and used in comparison across images, rather than to infer cardinal
    bias or the like.

    Parameters
    ----------
    image : np.ndarray
        The 2d array containing the image
    n_angle_slices : int, optional
        Number of slices between 0 and 2pi to break orientation into. Note that
        we only return half these slices (because orientation is symmetric,
        e.g., and orientation of 0 and pi is the same thing)
    metadata: OrderedDict, optional
        OrderedDict of extra coordinates to add to data (e.g., the model name).
        Should be an OrderedDict so we get the proper ordering of dimensions.

    Returns
    -------
    amplitude : xarray.Dataset
        Dataset containing the amplitudes in each orientation slice.

    """
    frq = sp_fft.fftshift(sp_fft.fft2(image))
    theta = pt.synthetic_images.polar_angle(frq.shape, np.pi/n_angle_slices)
    # to get this all positive and between 0 and 2pi
    theta += np.abs(theta.min())
    # following similar logic to amplitude_spectra() above
    theta = (n_angle_slices * theta/theta.max()).astype(int)
    # this will be 1 or a very small number of pixels, and we want to lump them
    # into the 0th bin (2pi is equivalent to 0)
    theta[theta == theta.max()] = 0
    # we ignore all frequencies outside a disk centered at the origin that
    # reaches to the first edge (in frequency space). This means we get all
    # frequencies that we can measure in each orientation (you can't get any
    # frequencies in the cardinal directions beyond this disk).
    frq_disk = pt.synthetic_images.polar_radius(frq.shape)
    frq_thresh = min(frq.shape)//2
    frq_disk = frq_disk < frq_thresh
    theta[~frq_disk] = theta.max()+1

    # convert this to NaN so we can use it for masking below
    frq_disk = frq_disk.astype(float)
    frq_disk[frq_disk == 0] = np.nan
    # mask out the high frequencies
    frq = frq_disk * np.abs(frq)

    slices = []
    # only need to go halfway around, because orientation is symmetric (an
    # orientation 0 is the same as an orientation of pi, i.e., up is the same
    # orientation as down)
    th = np.linspace(0, np.pi, n_angle_slices//2, endpoint=False)
    for i in range(theta.max()//2):
        # grab data from this slice...
        s = frq[theta == i]
        # ... and drop everything beyond the frequency disk
        slices.append(s[~np.isnan(s)])
    # now we want to concatenate this into a single array, which requires
    # making each slice the same length (they're slightly different because of
    # how they align with the pixel lattice).
    max_len = max([len(s) for s in slices])
    slices = np.stack([np.pad(s, (0, max_len-len(s)), constant_values=np.nan)
                       for s in slices])
    # coords need to be lists when creating a DataArray
    for k, v in metadata.items():
        if isinstance(v, str) or not hasattr(v, '__iter__'):
            metadata[k] = [v]
    metadata.update({'orientation_slice': th,
                     'samples': np.arange(slices.shape[-1])})
    # add extra dimensions to the front of slices for metadata.
    slices = np.expand_dims(slices,
                            tuple(np.arange(len(metadata.keys())-2)))
    ds = xarray.DataArray(slices, metadata, metadata.keys(),
                          name='orientation_amplitude')
    return ds.to_dataset()


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
    ori = []
    ori_metadata = metadata.copy()
    for n, im in zip(names, images):
        if isinstance(im, str):
            im = po.to_numpy(po.load_images(im)).squeeze()
        ori_metadata[name_dim] = n
        spectra.append(amplitude_spectra(im))
        ori.append(amplitude_orientation(im, metadata=ori_metadata))
    ori = xarray.concat(ori, 'image_name')
    for k, v in metadata.items():
        if isinstance(v, str) or not hasattr(v, '__iter__'):
            metadata[k] = [v]
    metadata.update({name_dim: names,
                     'freq_n': np.arange(len(spectra[-1]))})
    # add extra dimensions to the front of spectra for metadata.
    spectra = np.expand_dims(spectra,
                             tuple(np.arange(len(metadata.keys())-2)))
    data = xarray.DataArray(spectra, metadata, metadata.keys(),
                            name='sf_amplitude').to_dataset()
    return xarray.merge([data, ori])
