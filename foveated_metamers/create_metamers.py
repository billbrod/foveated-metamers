#!/usr/bin/python
"""create metamers for the experiment
"""
import torch
import re
import imageio
import warnings
import os
import time
import numpy as np
import plenoptic as po
import pyrtools as pt
import pandas as pd
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color
from .utils import convert_im_to_float, convert_im_to_int
# by default matplotlib uses the TK gui toolkit which can cause problems
# when I'm trying to render an image into a file, see
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import sys
mpl.use('Agg')
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages'))
import plenoptic_part as pop

def convert_seconds_to_str(secs):
    r"""Convert seconds into a human-readable string

    following https://stackoverflow.com/a/26277340/4659293
    """
    days = secs // 86400
    hours = secs // 3600 % 24
    minutes = secs // 60 % 60
    seconds = secs % 60
    return "%d:%02d:%02d:%.03f" % (days, hours, minutes, seconds)


def setup_image(image):
    r"""setup the image

    We load in the image, if it's not already done so (converting it to
    gray-scale in the process), make sure it lies between 0 and 1, and
    make sure it's a tensor of the correct type and specified device

    Parameters
    ----------
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    device : torch.device
        The torch device to put the image on

    Returns
    -------
    image : torch.Tensor
        The image tensor, ready to go

    """
    if isinstance(image, str):
        print("Loading in reference image from %s" % image)
        image = imageio.imread(image)
    if image.dtype == np.uint8:
        warnings.warn("Image is int8, with range (0, 255)")
        image = convert_im_to_float(image)
    elif image.dtype == np.uint16:
        warnings.warn("Image is int16 , with range (0, 65535)")
        image = convert_im_to_float(image)
    else:
        warnings.warn("Image is float 32, so we assume image range is (0, 1)")
        if image.max() > 1:
            raise Exception("Image is neither int8 nor int16, but its max is greater than 1!")
    # we use skimage.color.rgb2gray in order to handle rgb
    # correctly. this uses the ITU-R 601-2 luma transform, same as
    # matlab. we do this after the above, because it changes the image
    # dtype to float32
    if image.ndim == 3:
        # then it's a color image, and we need to make it grayscale
        image = color.rgb2gray(image)
    image = torch.tensor(image, dtype=torch.float32)
    while image.ndimension() < 4:
        image = image.unsqueeze(0)
    return image


def find_figsizes(model_name, model, image_shape):
    """Find figure sizes for various outputs, based on model.

    This gives a best guess, can definitely still be improved.

    Parameters
    ----------
    model_name : str
        str defining the model. Must begin with either RGC or V1.
    image_shape : array_like
        array_like giving the shape of the image. Height and width must be on
        the last two dimensions.

    Returns
    -------
    animate_figsize : tuple
        The figsize tuple to use with ``metamer.animate`` or
        ``metamer.plot_metamer_status`` functions
    rep_image_figsize : tuple
        The figsize tuple to pass to ``summary_plots`` to for the
        'rep_image' plot
    img_zoom : int or float
        Either an int or an inverse power of 2, how much to zoom the
        images by in the plots we'll create

    """
    if model_name.startswith('RGC'):
        animate_figsize = ((3+(image_shape[-1] / image_shape[-2])) * 5 + 2, 5.5)
        # these values were selected at 72 dpi, so will need to be adjusted if
        # ours is different
        animate_figsize = [s*72/mpl.rcParams['figure.dpi'] for s in animate_figsize]
        rep_image_figsize = [4, 13]
        # default figsize arguments work for an image that is 256x256,
        # may need to expand. we go backwards through figsize because
        # figsize and image shape are backwards of each other:
        # image_shape's last two indices are (height, width), while
        # figsize is (width, height)
        default_imgsize = np.array((256, (image_shape[-1] / image_shape[-2]) * 256))
    elif model_name.startswith('V1'):
        try:
            num_scales = int(re.findall('_s([0-9]+)_', model_name)[0])
        except (IndexError, ValueError):
            num_scales = 4
        animate_figsize = (40, 11)
        # we need about 11 per plot (and we have one of those per scale,
        # plus one for the mean luminance)
        rep_image_figsize = [11 * (num_scales+1), 30]
        # default figsize arguments work for an image that is 512x512 may need
        # to expand. we go backwards through figsize because figsize and image
        # shape are backwards of each other: image.shape's last two indices are
        # (height, width), while figsize is (width, height)
        default_imgsize = np.array((512, (image_shape[-1] / image_shape[-2]) * 512))
    # We want to figure out two things: 1. how much larger we need to
    # make the different figures so we can fit everything on them and
    # 2. if we need to shrink the images in order to fit
    # everything. here we determine how much bigger the image is than
    # the one we used to get the figsizes above
    zoom_factor = np.array([max(1, image_shape[::-1][i]/default_imgsize[i]) for
                            i in range(2)])
    if all(zoom_factor == 1):
        img_zoom = [int(np.round(default_imgsize[i] / image_shape[::-1][i]))
                    for i in range(2)]
        img_zoom = max(img_zoom)
    else:
        img_zoom = 1
    # if it's more than twice as big, then that's too much to blow
    # everything up, so we figure out how much to shrink the image by to
    # fit on a figure twice as big as above
    if (zoom_factor > 2).any():
        zoom_factor = np.array([min(i, 2) for i in zoom_factor])
        while ((np.array(image_shape[::-1][:2]) * img_zoom) > (default_imgsize*zoom_factor)).any():
            img_zoom /= 2
        zoom_factor = np.array([max(1, img_zoom*image_shape[::-1][i]/default_imgsize[i]) for i in range(2)])
    # img_zoom applies to the first image and then will increase by a factor of
    # 2 for all successive scales
    plot_shapes = np.array([img_zoom * 2**k * np.array(v.shape[-2:]) for k, v in
                            model.PoolingWindows.angle_windows.items()])
    if (plot_shapes.astype(int) != plot_shapes).any():
        raise Exception("At least one of the model scales will have a fractional image size. "
                        "Make your image size closer to a power of 2. Debug info: "
                        f"img_zoom: {img_zoom}, plot_shapes: {plot_shapes}")
    # and then update the figsizes appropriately
    animate_figsize = tuple([s*zoom_factor[i] for i, s in enumerate(animate_figsize)])
    rep_image_figsize = tuple([s*zoom_factor[i] for i, s in enumerate(rep_image_figsize)])
    rescale_factor = np.mean(zoom_factor)
    # 10 and 12 are the default font sizes for labels and titles,
    # respectively, and we want to scale them in order to keep them
    # readable. this should be global to matplotlib and so propagate
    # through
    mpl.rc('axes', labelsize=rescale_factor*10, titlesize=rescale_factor*12)
    mpl.rc('xtick', labelsize=rescale_factor*10)
    mpl.rc('ytick', labelsize=rescale_factor*10)
    mpl.rc('lines', linewidth=rescale_factor*1.5, markersize=rescale_factor*6)
    return animate_figsize, rep_image_figsize, img_zoom


def setup_model(model_name, scaling, image, min_ecc, max_ecc, cache_dir, normalize_dict=None):
    r"""setup the model

    We initialize the model, with the specified parameters, and return
    it with the appropriate figsizes.

    `model_name` is constructed of several parts, for which you have
    several chocies:
    `'{visual_area}{options}_{window_type}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.PooledRGC` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PrimaryVisualCortex` class)
    - `options`: you can additionally include the following strs,
      separated by `_`:
      - `'norm'`: if included, we normalize the models' `cone_responses`
        and (if V1) `complex_cell_responses` attributes. In this case,
        `normalize_dict` must also be set (and include those two
        keys). If not included, the model is not normalized
        (normalization makes the optimization easier because the
        different scales of the steerable pyramid have different
        magnitudes).
      - `s#` (V1 only), where `#` is an integer. The number of scales to
        inlude in the steerable pyramid that forms the basis fo the `V1`
        models. If not included, will use 4.
    - `window_type`: `'gaussian'` or `'cosine'`. whether to build the
      model with gaussian or raised-cosine windows. Regardless, scaling
      will always give the ratio between the FWHM and eccentricity of
      the windows, but the gaussian windows are much tighter packed, and
      so require more windows (and thus more memory), but also seem to
      have fewer aliasing issues.

    The recommended model_name values are: `RGC_norm_gaussian` and
    `V1_norm_s6_gaussian`.

    Parameters
    ----------
    model_name : str
        str specifying which of the `PooledVentralStream` models we should
        initialize. See above for more details.
    scaling : float
        The scaling parameter for the model
    image : torch.tensor or np.array
        The image we will call the model on. This is only necessary
        because we need to know how big it is; we just use its shape
    min_ecc : float
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    normalize_dict : dict or None, optional
        If a dict, should contain the stats to use for normalization. If
        None, we don't normalize. This can only be set (and must be set)
        if the model is "V1_norm". In any other case, we'll throw an
        Exception.

    Returns
    -------
    model : plenoptic.simul.VentralStream
        A ventral stream model, ready to use
    animate_figsize : tuple
        The figsize tuple to use with ``metamer.animate`` or
        ``metamer.plot_metamer_status`` functions
    rep_image_figsize : tuple
        The figsize tuple to pass to ``summary_plots`` to for the
        'rep_image' plot
    img_zoom : int or float
        Either an int or an inverse power of 2, how much to zoom the
        images by in the plots we'll create

    """
    if 'gaussian' in model_name:
        window_type = 'gaussian'
        t_width = None
        std_dev = 1
    elif 'cosine' in model_name:
        window_type = 'cosine'
        t_width = 1
        std_dev = None
    if model_name.startswith('RGC'):
        if 'norm' not in model_name:
            if normalize_dict:
                raise Exception("Cannot normalize RGC model (must be RGC_norm)!")
            normalize_dict = {}
        if not normalize_dict and 'norm' in model_name:
            raise Exception("If model_name is RGC_norm, normalize_dict must be set!")
        model = pop.PooledRGC(scaling, image.shape[-2:],
                              min_eccentricity=min_ecc,
                              max_eccentricity=max_ecc,
                              window_type=window_type,
                              transition_region_width=t_width,
                              cache_dir=cache_dir,
                              std_dev=std_dev,
                              normalize_dict=normalize_dict)
    elif model_name.startswith('V1'):
        if 'norm' not in model_name:
            if normalize_dict:
                raise Exception("Cannot normalize V1 model (must be V1_norm)!")
            normalize_dict = {}
        if not normalize_dict and 'norm' in model_name:
            raise Exception("If model_name is V1_norm, normalize_dict must be set!")
        try:
            num_scales = int(re.findall('_s([0-9]+)_', model_name)[0])
        except (IndexError, ValueError):
            num_scales = 4
        model = pop.PooledV1(scaling, image.shape[-2:],
                             min_eccentricity=min_ecc,
                             max_eccentricity=max_ecc,
                             std_dev=std_dev,
                             transition_region_width=t_width,
                             cache_dir=cache_dir,
                             normalize_dict=normalize_dict,
                             num_scales=num_scales,
                             window_type=window_type)
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    animate_figsize, rep_image_figsize, img_zoom = find_figsizes(model_name, model,
                                                                 image.shape)
    return model, animate_figsize, rep_image_figsize, img_zoom


def add_center_to_image(model, image, reference_image):
    r"""Add the reference image center to an image

    The VentralStream class of models will do nothing to the center of
    the image (they don't see the fovea), so we add the fovea to the
    image before synthesis.

    Parameters
    ----------
    model : plenoptic.simul.VentralStream
        The model used to create the metamer. Specifically, we need its
        windows attribute
    image : torch.Tensor
        The image to add the center back to
    reference_image : torch.Tensor
        The reference/target image for synthesis
        (``metamer.base_signal``); the center comes from this image.

    Returns
    -------
    recentered_image : torch.Tensor
        ``image`` with the reference image center added back in

    """
    model(image)
    rep = model.representation['mean_luminance']
    dummy_ones = torch.ones_like(rep)
    windows = model.PoolingWindows.project(dummy_ones).squeeze().to(image.device)
    # these aren't exactly zero, so we can't convert it to boolean
    anti_windows = 1 - windows
    return ((windows * image) + (anti_windows * reference_image))


def summary_plots(metamer, rep_image_figsize, img_zoom):
    r"""Create summary plots

    This creates two summary plots:

    1. 'rep_image': we show, in three separate rows, the representation
    of the reference image, the representation of the metamer, and the
    representation_error at the final iteration, all plotted as images
    using ``metamer.model.plot_representaiton_image``.

    2. 'windowed': on the top row we show the initial image, the
    metamer, and the reference image, and on the bottom row we show
    these with the contour lines of the windows (at value .5) plotted on
    top in red.

    Parameters
    ----------
    metamer : plenoptic.synth.Metamer
        The metamer object after synthesis
    rep_image_figsize : tuple
        Tuple of floats to use as the figsize for the 'rep_image'
    img_zoom : int or float
        Either an int or an inverse power of 2, how much to zoom the
        images by in the plots we'll create

    Returns
    -------
    rep_fig : matplotlib.figure.Figure
        The figure containing the 'rep_image' plot
    windowed_fig : matplotlib.figure.Figure
        The figure containing the 'windowed' plot

    """
    rep_fig, axes = plt.subplots(3, 1, figsize=rep_image_figsize)
    titles = ['Reference image |', 'Metamer |', 'Error |']
    if metamer.model.normalize_dict:
        # then we've z-scored the statistics and so they can be
        # negative. thus we want to use a symmetric colormap
        vranges = ['indep0', 'indep0', 'indep0']
    else:
        vranges = ['indep1', 'indep1', 'indep0']
    images = [metamer.model(metamer.base_signal), metamer.model(metamer.synthesized_signal),
              metamer.representation_error()]
    for i, (im, t, vr) in enumerate(zip(images, titles, vranges)):
        metamer.model.plot_representation_image(ax=axes[i], data=im, title=t, vrange=vr,
                                                zoom=img_zoom)
    images = [metamer.saved_signal[0], metamer.synthesized_signal, metamer.base_signal]
    images = 2*[po.to_numpy(i.to(torch.float32)).squeeze() for i in images]
    titles = ['Initial image', 'Metamer', 'Reference image']
    titles += ['Windowed '+t for t in titles]
    windowed_fig = pt.imshow(images, col_wrap=3, title=titles, vrange=(0, 1), zoom=img_zoom)
    for ax in windowed_fig.axes[3:]:
        metamer.model.plot_windows(ax)
    return rep_fig, windowed_fig


def _transform_summarized_rep(summarized_rep):
    """change around the keys of the summarized_representation dictionary

    This makes them more readable

    This function makes strong assumptions about what the keys look like
    (see PooledVentralStreams.summarize_representation for more info on this):
    a single string or tuples of the form `(a, b)`, `(a, b, c), or `((a,
    b, c), d)`, where all of `a,b,c,d` are strings or ints. We convert
    them as follows (single strings are untouched):
    - `(a, b) -> error_a_b`
    - `(a, b, c) -> error_a_scale_b_band_c`
    - `((a, b, c), d) -> error_a_scale_b_band_c_d`

    Parameters
    ----------
    summarized_rep : dict
        the dictionary whose keys we want to remap.

    Returns
    -------
    summarized_rep : dict
        dictionary with keys remapped

    """
    new_summarized_rep = {}
    for k, v in summarized_rep.items():
        if not isinstance(k, tuple):
            new_summarized_rep["error_" + k] = v
        elif isinstance(k[0], tuple):
            new_summarized_rep["error_scale_{}_band_{}_{}".format(*k[0], k[1])] = v
        else:
            if len(k) == 2:
                new_summarized_rep["error_scale_{}_band_{}".format(*k)] = v
            else:
                new_summarized_rep['error_' + '_'.join(k)] = v
    return new_summarized_rep


def summarize_history(metamer, save_path, **kwargs):
    r"""Generate and save summary of synthesis history

    In addition the `key=value` pairs passed as kwargs, we also save the
    following on each saved iteration (we sub-sample by a fair amount,
    so this will be something like every 40 iterations instead of every
    1):

    - iteration: the current iteration

    - loss: the loss on this iteration

    - num_statistics: the number of statistics in the model's
      representation

    - image_mse: the mean-squared error between the reference and
      synthesized images

    - learning_rate: learning rate on this iteration (this changes
      because we use a scheduler that reduces it when it looks like our
      loss has plateaued)

    - gradient_norm: gradient norm on this iteration

    - pixel_change: the max pixel change in the synthesized image from
      previous iteration to this

    - error terms: the summarized error terms, as returned by
      `metamer.model.summarize_representation(metamer.representation_error())`. This
      will be the error at each scale and each band.

    We also create a plot showing the loss, learning_rate,
    gradient_norm, pixel_change, and image_mse as a function of
    iterations, saved at `save_path.replace('.csv', '.png')`

    Parameters
    ----------
    metamer : pop.Metamer
        Metamer object to summarize
    save_path : str
        path to the csv where we should save the DataFrame we create
    kwargs :
        other values to save in this DataFrame. They should all be
        scalars

    Returns
    -------
    summary : pd.DataFrame
        The summary dataframe
    g : sns.FacetGrid
        FacetGrid containing line plots of the loss, learning rate,
        gradient norm, pixel change, and image mse as functions of
        iteration

    """
    num_saves = metamer.saved_signal.shape[0]
    summary = []
    keys = ['loss', 'image_mse', 'iteration', 'learning_rate', 'gradient_norm', 'num_statistics',
            'pixel_change']
    for k in keys:
        if k in kwargs.keys():
            warnings.warn(f"{k} found in the kwargs to add to history.csv, but we're going to "
                          "add that ourselves! Removing...")
            kwargs.pop(k)
    for i in range(1, num_saves):
        rep_error = metamer.representation_error(i)
        image_mse = torch.pow(metamer.base_signal - metamer.saved_signal[i], 2).mean().item()
        summarized_rep = metamer.model.summarize_representation(rep_error)
        summarized_rep = _transform_summarized_rep(summarized_rep)
        it = (i-1) * metamer.store_progress
        if it >= len(metamer.loss):
            it = -1
        data = {'loss': metamer.loss[it], 'image_mse': image_mse, 'iteration': it,
                'learning_rate': metamer.learning_rate[it], 'gradient_norm': metamer.gradient[it],
                'num_statistics': metamer.base_representation.numel(),
                'pixel_change': metamer.pixel_change[it]}
        data.update(summarized_rep)
        data.update(kwargs)
        summary.append(pd.DataFrame(data, index=[i]))
    summary = pd.concat(summary)
    melted = pd.melt(summary, ['iteration'], ['loss', 'learning_rate', 'gradient_norm',
                                              'pixel_change', 'image_mse'])
    g = sns.FacetGrid(melted, col='variable', sharey=False)
    g.map(sns.lineplot, 'iteration', 'value').set(yscale='log')
    g.savefig(save_path.replace('.csv', '.png'))
    summary.to_csv(save_path, index=False)
    return summary, g


def summarize(metamer, save_path, **kwargs):
    """Generate and save a summary of performance

    In addition the `key=value` pairs passed as kwargs, we also save the
    following:

    - num_iterations: the number of iterations synthesis ran for, which
      we grab from the length of the loss (note this means it can be
      less than the target number of iterations)

    - loss: the last (non-NaN) loss of synthesis

    - num_statistics: the number of statistics in the model's
      representation

    - image_mse: the mean-squared error between the reference and
      synthesized images

    - error terms: the summarized error terms, as returned by
      `metamer.model.summarize_representation(metamer.representation_error())`. This
      will be the error at each scale, each band

    - window  sizes:  the  summarized   window  sizes,  as  returned  by
      `metamer.model.summarize_window_sizes()`

    Parameters
    ----------
    metamer : pop.Metamer
        Metamer object to summarize
    save_path : str
        path to the csv where we should save the DataFrame we create
    kwargs :
        other values to save in this DataFrame. They should all be
        scalars

    Returns
    -------
    summary : pd.DataFrame
        The summary dataframe

    """
    loss = metamer.loss[-1]
    if np.isnan(loss):
        loss = metamer.loss[-2]
    data = {'num_iterations': len(metamer.loss), 'loss': loss,
            'num_statistics': metamer.base_representation.numel(),
            'image_mse': torch.pow(metamer.base_signal - metamer.synthesized_signal, 2).mean().item()}
    data.update(kwargs)
    summarized_rep = metamer.model.summarize_representation(metamer.representation_error())
    summarized_rep = _transform_summarized_rep(summarized_rep)
    data.update(summarized_rep)
    data.update(metamer.model.summarize_window_sizes())
    summary = pd.DataFrame(data, index=[0])
    summary.to_csv(save_path, index=False)
    return summary


def save(save_path, metamer, animate_figsize, rep_image_figsize, img_zoom,
         save_all=False):
    r"""save the metamer output

    We save several things here:
    - The metamer object itself, at ``save_path``. This contains, among
      other things, the saved image and representation over the course
      of synthesis.
    - The finished metamer image in its original float32 format (with
      values between 0 and 1, as a numpy array), at
      ``os.path.splitext(save_path)[0] + "_metamer.npy"``.
    - The finished metamer 8-bit image, at
      ``os.path.splitext(save_path)[0] + "_metamer.png"``.
    - The finished metamer 8-bit image, at
      ``os.path.splitext(save_path)[0] + "_metamer-16.png"``.
    - The 'rep_image', at ``os.path.splitext(save_path)[0]+"_rep.png"``.
      See ``summary_plots()`` docstring for a description of this plot.
    - The 'windowed_image', at ``os.path.splitext(save_path)[0] +
      "_windowed.png"``. See ``summary_plots()`` docstring for a
      description of this plot.
    - The video showing synthesis progress at
      ``os.path.splitext(save_path)[0] + "_synthesis.mp4"``. We use this
      to visualize the optimization progress.
    - Picture showing synthesis progress summary at
      ``os.path.splitext(save_path)[0] + "_synthesis.png"``.
    - The window normalization check plot for some angle slices at
      ``os.path.splitext(save_path)[0] + "_window_check.svg"``

    Parameters
    ----------
    save_path : str
        The path to save the metamer object at, which we use as a
        starting-point for the other save paths
    metamer : plenoptic.synth.Metamer
        The metamer object after synthesis
    animate_figsize : tuple
        The tuple describing the size of the figure for the synthesis
        video, as returned by ``setup_model``.
    rep_image_figsize : tuple
        The tuple describing the size of the figure for the rep_image
        plot, as returned by ``setup_model``.
    img_zoom : int or float
        Either an int or an inverse power of 2, how much to zoom the
        images by in the plots we'll create
    save_all : bool, optional
        If True, store_progress=1 and we cache the synthesized image and its
        representation each iteration. If False, we do it 100 times over the
        course of the synthesis. WARNING: This will massively increase the
        amount of RAM used (not on the GPU though), the footprint on disk, and
        the amount of time it takes to run. Because of this, we don't save the
        synthesis.mp4 movie, because it takes too long; however, snakemake
        expects a file, so we create a simple text file at that location

    """
    print("Saving at %s" % save_path)
    # We add the center back at the end because our gradients are not
    # exactly zero in the center, and thus those pixels end up getting
    # moved around a little bit. Not entirely sure why, but probably not
    # worth tracing down, since we're interested in the periphery
    metamer.synthesized_signal = torch.nn.Parameter(add_center_to_image(metamer.model,
                                                                   metamer.synthesized_signal,
                                                                   metamer.base_signal))
    metamer.save(save_path, save_model_reduced=True)
    # save png of metamer
    metamer_path = op.splitext(save_path)[0] + "_metamer.png"
    metamer_image = po.to_numpy(metamer.synthesized_signal).squeeze()
    print("Saving metamer float32 array at %s" % metamer_path.replace('.png', '.npy'))
    np.save(metamer_path.replace('.png', '.npy'), metamer_image)
    print("Saving metamer image at %s" % metamer_path)
    imageio.imwrite(metamer_path, convert_im_to_int(metamer_image))
    print("Saving 16-bit metamer image at %s" % metamer_path.replace('.png', '-16.png'))
    imageio.imwrite(metamer_path.replace('.png', '-16.png'),
                    convert_im_to_int(metamer_image, np.uint16))
    rep_fig, windowed_fig = summary_plots(metamer, rep_image_figsize, img_zoom)
    rep_path = op.splitext(save_path)[0] + "_rep.png"
    print("Saving representation image at %s" % rep_path)
    rep_fig.savefig(rep_path)
    windowed_path = op.splitext(save_path)[0] + "_windowed.png"
    print("Saving windowed image at %s" % windowed_path)
    windowed_fig.savefig(windowed_path)
    video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
    width_ratios = [metamer_image.shape[-1] / metamer_image.shape[-2], 1, 1, 1]
    if not save_all:
        print("Saving synthesis video at %s" % video_path)
        fig, axes = plt.subplots(1, 4, figsize=animate_figsize,
                                 gridspec_kw={'width_ratios': width_ratios,
                                              'left': .05, 'right': .95})
        anim = metamer.animate(fig=fig, imshow_zoom=img_zoom, plot_image_hist=True)
        anim.save(video_path)
    else:

        text = ("Because save_all was True, we're not outputting the synthesis video, "
                f"just saving small text file at {video_path}")
        print(text)
        with open(video_path, 'w') as f:
            f.writelines(text)
    synthesis_path = op.splitext(save_path)[0] + "_synthesis.png"
    print(f"Saving synthesis image at {synthesis_path}")
    fig, axes = plt.subplots(1, 4, figsize=animate_figsize,
                             gridspec_kw={'width_ratios': width_ratios,
                                          'left': .05, 'right': .95})
    fig = metamer.plot_synthesis_status(imshow_zoom=img_zoom,
                                        plot_image_hist=True, fig=fig)
    fig.savefig(synthesis_path, bbox_inches='tight')
    angle_n = np.linspace(0, metamer.model.n_polar_windows, 8, dtype=int, endpoint=False)
    fig = metamer.model.PoolingWindows.plot_window_checks(angle_n)
    window_check_path = op.splitext(save_path)[0] + "_window_check.svg"
    print(f"Saving window_check image at {window_check_path}")
    fig.savefig(window_check_path)


def setup_initial_image(initial_image_type, model, image):
    r"""setup the initial image

    Parameters
    ----------
    initial_image_type : {'white', 'pink', 'gray', 'blue'} or path to file
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere. If
        path to a file, that's what we use as our initial image (and so
        the seed will have no effect on this).
    model : plenoptic.simul.VentralStream
        The model used to create the metamer. Specifically, we need its
        windows attribute
    image : torch.Tensor
        The reference image tensor

    Returns
    -------
    initial_image : torch.Tensor
        The initial image to pass to metamer.synthesize

    """
    if initial_image_type == 'white':
        initial_image = torch.rand_like(image, dtype=torch.float32)
    elif initial_image_type == 'gray':
        initial_image = .5 * torch.ones_like(image, dtype=torch.float32)
    elif initial_image_type == 'pink':
        # this `.astype` probably isn't necessary, but just in case
        initial_image = pt.synthetic_images.pink_noise(image.shape[-2:]).astype(np.float32)
        # need to rescale this so it lies between 0 and 1
        initial_image += np.abs(initial_image.min())
        initial_image /= initial_image.max()
        initial_image = torch.Tensor(initial_image).unsqueeze(0).unsqueeze(0)
    elif initial_image_type == 'blue':
        # this `.astype` probably isn't necessary, but just in case
        initial_image = pt.synthetic_images.blue_noise(image.shape[-2:]).astype(np.float32)
        # need to rescale this so it lies between 0 and 1
        initial_image += np.abs(initial_image.min())
        initial_image /= initial_image.max()
        initial_image = torch.Tensor(initial_image).unsqueeze(0).unsqueeze(0)
    elif op.isfile(initial_image_type):
        warnings.warn("Using image %s as initial image!" % initial_image_type)
        initial_image = imageio.imread(initial_image_type)
        initial_image = convert_im_to_float(initial_image)
        initial_image = torch.tensor(initial_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        raise Exception("Don't know how to handle initial_image_type %s! Must be one of {'white',"
                        " 'gray', 'pink', 'blue'}" % initial_image_type)
    initial_image = add_center_to_image(model, initial_image, image)
    return torch.nn.Parameter(initial_image)


def setup_device(*args, gpu_id=None):
    r"""Setup device and get everything onto it

    This simple function checks whether ``torch.cuda.is_available()``
    and ``gpu_id`` is not None. If not, we use the cpu as the device

    We then call a.to(device) for every a in args (so this can be called
    with an arbitrary number of objects, each of which just needs to have
    .to method).

    Note that we always return a list (even if you only pass one item),
    so if you pass a single object, you'll need to either grab it
    specifically, either by doing ``im = setup_device(im,
    gpu_id=0)[0]`` or ``im, = setup_device(im)`` (notice the
    comma).

    Parameters
    ----------
    args :
        Some number of torch objects that we want to get on the proper
        device
    gpu_id : int or None, optional
        If not None, the GPU we will use. If None, we run on CPU. We
        don't do anything clever to handle that here, but the
        contextmanager utils.get_gpu_id does, so you should use that to
        make sure you're using a GPU that exists and is available (see
        Snakefile for example). Note that, to set this,
        you must set it as a keyword, i.e., ``setup_device(im, 0)``
        won't work but ``setup_device(im, gpu_id=True)`` will (this is
        because the ``*args`` in our function signature will greedily
        grab every non-keyword argument).

    Returns
    -------
    args : list
        Every item we were passed in arg, now on the proper device

    """
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available but gpu_id is not None!")
        device = torch.device("cuda:%s" % gpu_id)
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print("On device %s" % device)
    if dtype is not None:
        print("Changing dtype to %s" % dtype)
        args = [a.to(dtype) for a in args]
    return [a.to(device) for a in args]


def main(model_name, scaling, image, seed=0, min_ecc=.5, max_ecc=15, learning_rate=1, max_iter=100,
         loss_thresh=1e-4, loss_change_iter=50, save_path=None, initial_image_type='white',
         gpu_id=None, cache_dir=None, normalize_dict=None, optimizer='SGD', fraction_removed=0,
         loss_change_fraction=1, loss_change_thresh=.1, coarse_to_fine=False, clamper_name='clamp',
         clamp_each_iter=True, loss_func='l2', continue_path=None, save_all=False, num_threads=None):
    r"""create metamers!

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    `model_name` is constructed of several parts, for which you have
    several chocies:
    `'{visual_area}{options}_{window_type}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.PooledRGC` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PooledV1` class)
    - `options`: only for the `V1` models, you can additionally include
      the following strs, separated by `_`:
      - `'norm'`: if included, we normalize the models' `cone_responses`
        and `complex_cell_responses` attributes. In this case,
        `normalize_dict` must also be set (and include those two
        keys). If not included, the model is not normalized
        (normalization makes the optimization easier because the
        different scales of the steerable pyramid have different
        magnitudes).
      - `s#`, where `#` is an integer. The number of scales to inlude in
        the steerable pyramid that forms the basis fo the `V1`
        models. If not included, will use 4.
    - `window_type`: `'gaussian'` or `'cosine'`. whether to build the
      model with gaussian or raised-cosine windows. Regardless, scaling
      will always give the ratio between the FWHM and eccentricity of
      the windows, but the gaussian windows are much tighter packed, and
      so require more windows (and thus more memory), but also seem to
      have fewer aliasing issues.

    The recommended model_name values are: `RGC_norm_gaussian` and
    `V1_norm_s6_gaussian`.

    If you want to resume synthesis from an earlier run that didn't
    finish, set `continue_path` to the path of the `.pt` file created by
    that earlier run. We will then load it in and continue. For right
    now, we don't do anything to make sure that the arguments you pass
    to the function are the same as the first time, we just use the ones
    passed in. Generally, they should be identical, with the exception
    of learning_rate (which can be None to resume where you left off)
    and max_iter (which gives the number of extra iterations you want to
    do). Specifically, I think things might get weird if you do this
    initially on a GPU and then try to resume on a CPU (or vice versa),
    for example. When resuming, there's always a slight increase in the
    loss that, as far as I can tell, is unavoidable; it goes away
    quickly (and the loss continues its earlier trend) and so I don't
    think is an issue.

    Parameters
    ----------
    model_name : str
        str specifying which of the `PooledVentralStream` models we should
        initialize. See above for more details.
    scaling : float
        The scaling parameter for the model
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    seed : int, optional
        The number to use for initializing numpy and torch's random
        number generators
    min_ecc : float, optional
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float, optional
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    learning_rate : float, optional
        The learning rate to pass to metamer.synthesize's optimizer
    max_iter : int, optional
        The maximum number of iterations we allow the synthesis
        optimization to run for
    loss_thresh : float, optional
        The loss threshold. If the loss has changed by less than this
        over the past loss_change_iter iterations, we quit out.
    loss_change_iter : int, optional
        How many iterations back to check in order to see if the loss
        has stopped decreasing (for both loss_change_iter and
        coarse-to-fine optimization)
    save_path : str or None, optional
        If a str, the path to the file to save the metamer object to. If
        None, we don't save the synthesis output (that's probably a bad
        idea)
    initial_image_type : {'white', 'pink', 'gray', 'blue'} or path to a file
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere. If
        path to a file, that's what we use as our initial image (and so
        the seed will have no effect on this).
    gpu_id : int or None, optional
        If not None, the GPU we will use. If None, we run on CPU. We
        don't do anything clever to handle that here, but the
        contextmanager utils.get_gpu_id does, so you should use that to
        make sure you're using a GPU that exists and is available (see
        Snakefile for example)
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    normalize_dict : str or None, optional
        If a str, the path to the dictionary containing the statistics
        to use for normalization. If None, we don't normalize anything
    optimizer: {'Adam', 'SGD', 'LBFGS'}
        The choice of optimization algorithm
    fraction_removed: float, optional
        The fraction of the representation that will be ignored
        when computing the loss. At every step the loss is computed
        using the remaining fraction of the representation only.
        A new sample is drawn a every step. This gives a stochastic
        estimate of the gradient and might help optimization.
    loss_change_fraction : float, optional
        If we think the loss has stopped decreasing, the fraction of
        the representation with the highest loss that we use to
        calculate the gradients
    loss_change_thresh : float, optional
        the threshold we use to see if the loss has stopped changing
        (for either loss_change_fraction or coarse_to_fine). If
        coarse_to_fine is False, this should be .1; else, you'll have to
        play around and find the best value
    coarse_to_fine : { 'together', 'separate', False}, optional
        If False, don't do coarse-to-fine optimization. Else, there
        are two options for how to do it:
        - 'together': start with the coarsest scale, then gradually
          add each finer scale. this is like blurring the objective
          function and then gradually adding details and is probably
          what you want.
        - 'separate': compute the gradient with respect to each
          scale separately (ignoring the others), then with respect
          to all of them at the end.
    clamper_name : {'clamp', 'remap', 'clamp{a},{b}', 'clamp2', 'clamp4'}, optional
        For the image to make sense, its range must lie between 0 and
        1. We can enforce that in two ways: clamping (in which case we
        send everything below 0 to 0 and everything above 1 to 1) or
        remapping (in which case we subtract away the minimum and divide
        by the max). 'clamp{a},{b}`, where a,b are both flotas, clamps
        instead to the range (a, b). 'clamp2' clamps the range, mean,
        and variance to match that of the target image. 'clamp4' clamps
        the range and the first four moments to that of the target
        image.
    clamp_each_iter : bool, optional
        Whether we call the clamper each iteration of the optimization
        or only at the end. True, the default, is recommended
    loss_func : {'l2', 'l2_range-{a},{b}_beta-{c}', 'mse', 'mse_range-{a},{b}_beta-{c}'}
        where a,b,c are all floats. what loss function to use. If 'l2', then we
        use the L2-norm of the difference between the model representations of
        the synthesized and reference image. if 'l2_range-a,b_beta-c', then we
        use c times that loss plus (1-c) times a quadratic penalty on all
        pixels in synthesized image whose values are below a or above b. If
        'mse', we use mean-squared error instead, 'mse_range-a,b_beta-c' is
        interpreted the same way as 'l2_range-a,b_beta-c', just using MSE
        instead of the L2-norm
    continue_path : str or None, optional
        If None, we synthesize a new metamer. If str, this should be the
        path to a previous synthesis run, which we are resuming. In that
        case, you may set learning_rate to None (in which case we resume
        where we left off) and set max_iter to a different value (the
        number of extra iterations to run) otherwise the rest of the
        arguments should be the same as the first run.
    save_all : bool, optional
        If True, store_progress=1 and we cache the synthesized image and its
        representation each iteration. If False, we do it 100 times over the
        course of the synthesis. WARNING: This will massively increase the
        amount of RAM used (not on the GPU though), the footprint on disk, and
        the amount of time it takes to run.
    num_threads : int or None, optional
        If int, the number of CPU threads to use. If None, we don't restrict it
        and so we'll use all available resources. If using the GPU, this won't
        matter (all costly computations are done on the GPU). If one the CPU,
        we seem to only improve performance up to ~12 threads (at least with
        RGC model), and actively start to harm performance as we get above 40.

    """
    print("Using seed %s" % seed)
    if num_threads is not None:
        print(f"Using {num_threads} threads")
        torch.set_num_threads(num_threads)
    else:
        print("Not restricting number of threads, will probably use max "
              f"available ({torch.get_num_threads()})")
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_name = image
    image = setup_image(image)
    # this will be false if normalize_dict is None or an empty list
    if normalize_dict:
        normalize_dict = torch.load(normalize_dict)
    model, animate_figsize, rep_figsize, img_zoom = setup_model(model_name, scaling, image,
                                                                min_ecc, max_ecc, cache_dir,
                                                                normalize_dict)
    print("Using model %s from %.02f degrees to %.02f degrees" % (model_name, min_ecc, max_ecc))
    initial_image = setup_initial_image(initial_image_type, model, image)
    image, initial_image, model = setup_device(image, initial_image, model, gpu_id=gpu_id)
    if clamper_name == 'clamp':
        clamper = pop.clamps.RangeClamper((0, 1))
    elif clamper_name.startswith('clamp.'):
        a, b = re.findall('clamp([.0-9]+),([.0-9]+)', clamper_name)[0]
        clamper = pop.clamps.RangeClamper((float(a), float(b)))
    elif clamper_name == 'clamp2':
        clamper = pop.clamps.TwoMomentsClamper(image)
    elif clamper_name == 'clamp4':
        clamper = pop.clamps.FourMomentsClamper(image)
    elif clamper_name == 'remap':
        clamper = pop.clamps.RangeRemapper((0, 1))
    else:
        clamper = None
    if loss_func == 'l2':
        loss = pop.optim.l2_norm
        loss_kwargs = {}
    elif loss_func == 'mse':
        loss = pop.optim.mse
        loss_kwargs = {}
    else:
        lf, a, b, c = re.findall('([a-z0-9]+)_range-([.0-9]+),([.0-9]+)_beta-([.0-9]+)',
                                 loss_func)[0]
        if lf == 'l2':
            loss = pop.optim.l2_and_penalize_range
        elif lf == 'mse':
            loss = pop.optim.mse_and_penalize_range
        else:
            raise Exception(f"Don't know how to interpret loss func {loss_func}!")
        loss_kwargs = {'allowed_range': (float(a), float(b)), 'beta': float(c)}
    if '-' in optimizer:
        # we allow two possible addenda to SWA, s-S and f-F, where S is the
        # value for swa-start and F is the value for swa_freq, respectively. if
        # not present, we use 10 and 1, respectively. using the non-capturing
        # group (with the `?:` syntax) means this will always have two values
        kwarg_vals = re.findall('SWA(?:_s-([\d]+))?(?:_f-([\d]+))?', optimizer)[0]
        swa_kwargs = {'swa_start': 10, 'swa_freq': 1, 'swa_lr': learning_rate/2}
        for k, v in zip(['swa_start', 'swa_freq'], kwarg_vals):
            if v:
                swa_kwargs[k] = int(v)
        swa = True
        swa_str = f", with SWA and kwargs {swa_kwargs}"
        optimizer = optimizer.split('-')[0]
    else:
        swa = False
        swa_kwargs = {}
        swa_str = ""
    print(f"Using optimizer {optimizer}{swa_str}")
    # want to set store_progress before we potentially change max_iter below,
    # because if we're resuming synthesis, want to have the same store_progress
    # arg
    if save_all:
        store_progress = 1
    else:
        # don't want to store too often, otherwise we slow down and use too
        # much memory. this way we store at most 100 time points
        store_progress = max(10, max_iter//100)
    if save_path is not None:
        inprogress_path = save_path.replace('.pt', '_inprogress.pt')
    else:
        inprogress_path = None
    if continue_path is not None or (inprogress_path is not None and op.exists(inprogress_path)):
        if op.exists(inprogress_path):
            continue_path = inprogress_path
        print("Resuming synthesis saved at %s" % continue_path)
        metamer = pop.Metamer.load(continue_path, model.from_state_dict_reduced)
        if op.exists(inprogress_path):
            # run the number of extra iterations we need, not more. if it was
            # complete, then this can be 0 and we do no iterations, but will
            # still save everything else out (useful if synthesis finished
            # without a problem but hit an out of memory or time error while
            # saving outputs)
            max_iter = max(max_iter - len(metamer.loss), 0)
        initial_image = None
        learning_rate = None
    else:
        metamer = pop.Metamer(image, model, loss_function=loss,
                              loss_function_kwargs=loss_kwargs)
    print(f"Using learning rate {learning_rate}, loss_thresh {loss_thresh} (loss_change_iter "
          f"{loss_change_iter}), and max_iter {max_iter}")
    if save_path is not None:
        if max_iter < 200:
            # no sense when it's this short
            save_progress = False
        else:
            save_progress = max(200, max_iter//10)
    else:
        save_progress = False
    start_time = time.time()
    # note that there's a possibility that max_iter=0 (in particular, if we're
    # loading in an inprogress.pt file that finished synthesis but had trouble
    # when saving outputs). we still want to call synthesize, because there's a
    # small amount of wrapping up that needs to happen
    matched_im, matched_rep = metamer.synthesize(clamper=clamper,
                                                 store_progress=store_progress,
                                                 learning_rate=learning_rate,
                                                 max_iter=max_iter,
                                                 loss_thresh=loss_thresh,
                                                 loss_change_iter=loss_change_iter,
                                                 seed=seed,
                                                 initial_image=initial_image,
                                                 clamp_each_iter=clamp_each_iter,
                                                 save_progress=save_progress,
                                                 optimizer=optimizer,
                                                 swa=swa, swa_kwargs=swa_kwargs,
                                                 fraction_removed=fraction_removed,
                                                 loss_change_fraction=loss_change_fraction,
                                                 loss_change_thresh=loss_change_thresh,
                                                 coarse_to_fine=coarse_to_fine,
                                                 save_path=inprogress_path)
    duration = time.time() - start_time
    # make sure everything's on the cpu for saving
    metamer = metamer.to('cpu')
    if save_path is not None:
        summarize(metamer, save_path.replace('.pt', '_summary.csv'),
                  duration_human_readable=convert_seconds_to_str(duration), duration=duration,
                  optimizer=optimizer, fraction_removed=fraction_removed, model=model_name,
                  base_signal=image_name, seed=seed, learning_rate=learning_rate,
                  loss_change_thresh=loss_change_thresh, coarse_to_fine=coarse_to_fine,
                  loss_change_fraction=loss_change_fraction, initial_image=initial_image_type,
                  min_ecc=min_ecc, max_ecc=max_ecc, max_iter=max_iter, gpu_id=gpu_id,
                  loss_thresh=loss_thresh, scaling=scaling, clamper=clamper_name,
                  clamp_each_iter=clamp_each_iter, loss_change_iter=loss_change_iter,
                  image_name=op.basename(image_name).replace('.pgm', '').replace('.png', ''),
                  loss_function=loss_func)
        summarize_history(metamer, save_path.replace('.pt', '_history.csv'),
                          duration_human_readable=convert_seconds_to_str(duration), duration=duration,
                          optimizer=optimizer, fraction_removed=fraction_removed, model=model_name,
                          base_signal=image_name, seed=seed, loss_change_thresh=loss_change_thresh,
                          coarse_to_fine=coarse_to_fine, loss_change_fraction=loss_change_fraction,
                          initial_image=initial_image_type, min_ecc=min_ecc, max_ecc=max_ecc,
                          max_iter=max_iter, gpu_id=gpu_id, loss_thresh=loss_thresh,
                          scaling=scaling, clamper=clamper_name, clamp_each_iter=clamp_each_iter,
                          loss_function=loss_func,loss_change_iter=loss_change_iter,
                          image_name=op.basename(image_name).replace('.pgm', '').replace('.png', ''))
        save(save_path, metamer, animate_figsize, rep_figsize, img_zoom, save_all)
    if save_progress and op.exists(inprogress_path):
        os.remove(inprogress_path)
