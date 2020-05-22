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
from skimage import color
from .utils import convert_im_to_float, convert_im_to_int
# by default matplotlib uses the TK gui toolkit which can cause problems
# when I'm trying to render an image into a file, see
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
mpl.use('Agg')


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


def setup_model(model_name, scaling, image, min_ecc, max_ecc, cache_dir, normalize_dict=None):
    r"""setup the model

    We initialize the model, with the specified parameters, and return
    it with the appropriate figsizes.

    `model_name` is constructed of several parts, for which you have
    several chocies:
    `'{visual_area}_cone-{cone_power}{options}_{window_type}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.RetinalGanglionCells` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PrimaryVisualCortex` class)
    - `cone_power`: first step fo the model is to raise every pixel in
      the image to a power, specified here. It can be any float or the
      strs `'phys'` (1/3, the approximately correct physiological value
      for cones) or `'gamma'` (1/2.2, the standard gamma value, so that
      the model basically gamma-corrects the image itself). Metamer
      synthesis has difficulty with non-convex powers here
      (`cone_power<1`), so we apply `cone_power=1/3` to the image, build
      the model with `cone_power=1.0`, and then raise the pixels of the
      resulting image to `3` at the end (this all happens in
      `Snakefile`, not this script).
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

    The recommended model_name values are: `RGC_cone-1.0_gaussian` and
    `V1_cone-1.0_norm_s6_gaussian` (see above for why we use `cone-1.0`
    for synthesis).

    Parameters
    ----------
    model_name : str
        str specifying which of the `VentralModel` models we should
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
    try:
        # cone_power must be a float, but we can't have / in a path (it
        # separates directories), so in order to work around that, we
        # have to special repeating decimal floats we want to be able to
        # get: 1/2.2 (the standard gamma value) and 1/3 (the standard
        # approximation of cone physiology).
        if 'cone-gamma' in model_name:
            cone_power = 1/2.2
        elif 'cone-phys' in model_name:
            cone_power = 1/3
        else:
            cone_power = float(re.findall('cone-([.0-9]+)', model_name)[0])
    except IndexError:
        # default is 1, linear response
        cone_power = 1
    surround_std_dev = None
    center_surround_ratio = None
    transition_x = None
    if 'gaussian' in model_name:
        window_type = 'gaussian'
        t_width = None
        std_dev = 1
    elif 'cosine' in model_name:
        window_type = 'cosine'
        t_width = 1
        std_dev = None
    elif 'dog' in model_name:
        # then model_name will also include the center_surround_ratio
        # and surround_std_dev
        window_type = 'dog'
        surround_std_dev = float([n for n in model_name.split('_') if n.startswith('s-')][0].split('-')[-1])
        center_surround_ratio = float([n for n in model_name.split('_') if n.startswith('r-')][0].split('-')[-1])
        t_width = None
        std_dev = 1
        transition_x = min_ecc
        min_ecc = None
    if model_name.startswith('RGC'):
        if normalize_dict:
            raise Exception("Cannot normalize RGC model!")
        model = po.simul.RetinalGanglionCells(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                              max_eccentricity=max_ecc, window_type=window_type,
                                              transition_region_width=t_width, cache_dir=cache_dir,
                                              cone_power=cone_power, std_dev=std_dev,
                                              surround_std_dev=surround_std_dev,
                                              center_surround_ratio=center_surround_ratio,
                                              transition_x=transition_x)
        animate_figsize = (17, 5)
        if model.window_type == 'dog':
            # then our rep_image will include 3 plots, instead of 1, so
            # we want it to be wider
            rep_image_figsize = (13, 13)
        else:
            rep_image_figsize = [4, 13]
        # default figsize arguments work for an image that is 256x256,
        # may need to expand. we go backwards through figsize because
        # figsize and image shape are backwards of each other:
        # image.shape's last two indices are (height, width), while
        # figsize is (width, height)
        default_imgsize = 256
    elif model_name.startswith('V1'):
        if 'norm' not in model_name:
            if normalize_dict:
                raise Exception("Cannot normalize V1 model (must be V1_norm)!")
            normalize_dict = {}
        if not normalize_dict and 'norm' in model_name:
            raise Exception("If model_name is V1_norm, normalize_dict must be set!")
        if 'half-oct' in model_name:
            half_oct = True
        else:
            half_oct = False
        if 'highpass' in model_name:
            include_highpass = True
        else:
            include_highpass = False
        try:
            num_scales = int(re.findall('s([0-9]+)', model_name)[0])
        except (IndexError, ValueError):
            num_scales = 4
        model = po.simul.PrimaryVisualCortex(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                             max_eccentricity=max_ecc, std_dev=std_dev,
                                             transition_region_width=t_width,
                                             cache_dir=cache_dir, normalize_dict=normalize_dict,
                                             half_octave_pyramid=half_oct, num_scales=num_scales,
                                             cone_power=cone_power, window_type=window_type,
                                             include_highpass=include_highpass)
        animate_figsize = (35, 11)
        # we need about 11 per plot (and we have one of those per scale,
        # plus one for the mean luminance)
        rep_image_figsize = [11 * (num_scales+1), 30]
        if 'half-oct' in model_name:
            # in this case, we have almost twice as many plots to make
            rep_image_figsize[0] *= 2
        if 'highpass' in model_name:
            # then we have one more to make
            rep_image_figsize[0] += 11
        # default figsize arguments work for an image that is 512x512,
        # may need to expand. we go backwards through figsize because
        # figsize and image shape are backwards of each other:
        # image.shape's last two indices are (height, width), while
        # figsize is (width, height)
        default_imgsize = 512
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    # We want to figure out two things: 1. how much larger we need to
    # make the different figures so we can fit everything on them and
    # 2. if we need to shrink the images in order to fit
    # everything. here we determine how much bigger the image is than
    # the one we used to get the figsizes above
    zoom_factor = np.array([max(1, image.shape[::-1][i]/default_imgsize) for i in range(2)])
    img_zoom = 1
    # if it's more than twice as big, then that's too much to blow
    # everything up, so we figure out how much to shrink the image by to
    # fit on a figure twice as big as above
    if (zoom_factor > 2).any():
        zoom_factor = np.array([min(i, 2) for i in zoom_factor])
        while ((np.array(image.shape[::-1][:2]) * img_zoom) > (default_imgsize*zoom_factor)).any():
            img_zoom /= 2
        zoom_factor = np.array([max(1, img_zoom*image.shape[::-1][i]/default_imgsize) for i in range(2)])
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
        (``metamer.target_image``); the center comes from this image.

    Returns
    -------
    recentered_image : torch.Tensor
        ``image`` with the reference image center added back in

    """
    model(image)
    try:
        rep = model.representation['mean_luminance']
    except IndexError:
        rep = model.representation
    try:
        dummy_ones = torch.ones_like(rep)
        windows = model.PoolingWindows.project(dummy_ones).squeeze().to(image.device)
        # these aren't exactly zero, so we can't convert it to boolean
        anti_windows = 1 - windows
    except NotImplementedError:
        # then this model has DoG windows. the portion that the DoG
        # windows don't cover is the same as the portion the center
        # windows don't cover (if we do this boolean conversion)
        dummy_ones = torch.ones_like(model.PoolingWindows.forward(image, windows_key='center'))
        windows = model.PoolingWindows.project(dummy_ones, windows_key='center').squeeze()
        windows = windows.to(bool).to(image.device)
        anti_windows = ~windows
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
    if metamer.model.state_dict_reduced['model_name'] == 'V1' and metamer.model.normalize_dict:
        # then this is the V1_norm model and so we want to use symmetric
        # color maps for all of them
        vranges = ['indep0', 'indep0', 'indep0']
    else:
        vranges = ['indep1', 'indep1', 'indep0']
    if metamer.model.window_type == 'dog':
        # For DoG windows, need to pass not the model called on the image,
        # but the image itself. for representation error, difference of the
        # target and matched image will do
        images = [metamer.target_image, metamer.matched_image,
                  metamer.target_image - metamer.matched_image]
    else:
        images = [metamer.model(metamer.target_image), metamer.model(metamer.matched_image),
                  metamer.representation_error()]
    for i, (im, t, vr) in enumerate(zip(images, titles, vranges)):
        metamer.model.plot_representation_image(ax=axes[i], data=im, title=t, vrange=vr,
                                                zoom=img_zoom)
    images = [metamer.saved_image[0], metamer.matched_image, metamer.target_image]
    images = 2*[po.to_numpy(i.to(torch.float32)).squeeze() for i in images]
    titles = ['Initial image', 'Metamer', 'Reference image']
    if metamer.model.window_type == 'dog':
        titles += ['Windowed (center) '+t for t in titles]
    else:
        titles += ['Windowed '+t for t in titles]
    windowed_fig = pt.imshow(images, col_wrap=3, title=titles, vrange=(0, 1), zoom=img_zoom)
    for ax in windowed_fig.axes[3:]:
        metamer.model.plot_windows(ax)
    return rep_fig, windowed_fig


def summarize(metamer, save_path, **kwargs):
    r"""Generate and save some summaries
    """
    loss = metamer.loss[-1]
    if np.isnan(loss):
        loss = metamer.loss[-2]
    data = {'num_iterations': len(metamer.loss), 'loss': loss,
            'num_statistics': metamer.target_representation.numel(),
            'image_mse': torch.pow(metamer.target_image - metamer.matched_image, 2).mean().item()}
    data.update(kwargs)
    summary = pd.DataFrame(data, index=[0])
    summary.to_csv(save_path, index=False)
    return summary


def save(save_path, metamer, animate_figsize, rep_image_figsize, img_zoom):
    r"""save the metamer output

    We save five things here:
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

    """
    print("Saving at %s" % save_path)
    # With the Adam optimizer, it also changes the pixels in the center,
    # which the model does not see. This appears to be a feature of Adam
    # (maybe some randomness in how it selects parameters to change?),
    # since it basically doesn't happen with SGD and the gradient at
    # those pixels is always zero. So, just to make things look nice, we
    # add back the center at the end here.
    metamer.matched_image = torch.nn.Parameter(add_center_to_image(metamer.model,
                                                                   metamer.matched_image,
                                                                   metamer.target_image))
    metamer.save(save_path, save_model_reduced=True)
    # save png of metamer
    metamer_path = op.splitext(save_path)[0] + "_metamer.png"
    metamer_image = po.to_numpy(metamer.matched_image).squeeze()
    print("Saving metamer float32 array at %s" % metamer_path.replace('.png', '.npy'))
    np.save(metamer_path.replace('.png', '.npy'), metamer_image)
    print("Saving metamer image at %s" % metamer_path)
    imageio.imwrite(metamer_path, convert_im_to_int(metamer_image))
    print("Saving 16-bit metamer image at %s" % metamer_path.replace('.png', '-16.png'))
    imageio.imwrite(metamer_path.replace('.png', '-16.png'),
                    convert_im_to_int(metamer_image, np.uint16))
    video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
    rep_fig, windowed_fig = summary_plots(metamer, rep_image_figsize, img_zoom)
    rep_path = op.splitext(save_path)[0] + "_rep.png"
    print("Saving representation image at %s" % rep_path)
    rep_fig.savefig(rep_path)
    windowed_path = op.splitext(save_path)[0] + "_windowed.png"
    print("Saving windowed image at %s" % windowed_path)
    windowed_fig.savefig(windowed_path)
    print("Saving synthesis video at %s" % video_path)
    anim = metamer.animate(figsize=animate_figsize, imshow_zoom=img_zoom)
    anim.save(video_path)
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
         loss_thresh=1e-4, save_path=None, initial_image_type='white', gpu_id=None,
         cache_dir=None, normalize_dict=None, optimizer='SGD', fraction_removed=0,
         loss_change_fraction=1, coarse_to_fine=0, clamper_name='clamp', clamp_each_iter=True,
         loss_func='l2', continue_path=None):
    r"""create metamers!

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    `model_name` is constructed of several parts, for which you have
    several chocies:
    `'{visual_area}_cone-{cone_power}{options}_{window_type}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.RetinalGanglionCells` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PrimaryVisualCortex` class)
    - `cone_power`: first step fo the model is to raise every pixel in
      the image to a power, specified here. It can be any float or the
      strs `'phys'` (1/3, the approximately correct physiological value
      for cones) or `'gamma'` (1/2.2, the standard gamma value, so that
      the model basically gamma-corrects the image itself). Metamer
      synthesis has difficulty with non-convex powers here
      (`cone_power<1`), so we apply `cone_power=1/3` to the image, build
      the model with `cone_power=1.0`, and then raise the pixels of the
      resulting image to `3` at the end (this all happens in
      `Snakefile`, not this script).
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

    The recommended model_name values are: `RGC_cone-1.0_gaussian` and
    `V1_cone-1.0_norm_s6_gaussian` (see above for why we use `cone-1.0`
    for synthesis).

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
        str specifying which of the `VentralModel` models we should
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
        over the past 50 iterations, we quit out.
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
    coarse_to_fine : float, optional
        A positive float or 0. If a positive float, we do coarse-to-fine
        optimization (see Metamer.synthesize) for more details, passing
        coarse_to_fine=True and loss_change_thresh as this value. If 0,
        we set coarse_to_fine=False (and loss_change_thresh=.1)
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
        or only at the end. True, the default, is recommended, and is
        necessary for fractional values of cone_power
    loss_func : {'l2', 'l2_range-{a},{b}_beta-{c}'}
        where a,b,c are all floats. what loss function to use. If 'l2',
        then we use the L2-norm of the difference between the model
        representations of the synthesized and reference image. if
        'l2_range-a,b_beta-c', then we use a weighted average of that
        (weight given by c) and a quadratic penalty on all pixels in
        synthesized image whose values are below a or above b.
    continue_path : str or None, optional
        If None, we synthesize a new metamer. If str, this should be the
        path to a previous synthesis run, which we are resuming. In that
        case, you may set learning_rate to None (in which case we resume
        where we left off) and set max_iter to a different value (the
        number of extra iterations to run) otherwise the rest of the
        arguments should be the same as the first run.

    """
    print("Using seed %s" % seed)
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
        clamper = po.RangeClamper((0, 1))
    elif clamper_name.startswith('clamp.'):
        a, b = re.findall('clamp([.0-9]+),([.0-9]+)', clamper_name)[0]
        clamper = po.RangeClamper((float(a), float(b)))
    elif clamper_name == 'clamp2':
        clamper = po.TwoMomentsClamper(image)
    elif clamper_name == 'clamp4':
        clamper = po.FourMomentsClamper(image)
    elif clamper_name == 'remap':
        clamper = po.RangeRemapper((0, 1))
    else:
        clamper = None
    if loss_func == 'l2':
        loss = po.optim.l2_norm
        loss_kwargs = {}
    else:
        a, b, c = re.findall('l2_range-([.0-9]+),([.0-9]+)_beta-([.0-9]+)', loss_func)[0]
        loss = po.optim.l2_and_penalize_range
        loss_kwargs = {'allowed_range': (float(a), float(b)), 'beta': float(c)}
    if continue_path is None:
        metamer = po.synth.Metamer(image, model, loss_function=loss, loss_kwargs=loss_kwargs)
    else:
        print("Resuming synthesis saved at %s" % continue_path)
        metamer = po.synth.Metamer.load(continue_path, model.from_state_dict_reduced)
        initial_image = None
        learning_rate = None
    print("Using learning rate %s, loss_thresh %s, and max_iter %s" % (learning_rate, loss_thresh,
                                                                       max_iter))
    if save_path is not None:
        if max_iter < 200:
            # no sense when it's this short
            save_progress = False
        else:
            save_progress = max(200, max_iter//10)
    else:
        save_progress = False
    if coarse_to_fine > 0:
        loss_change_thresh = coarse_to_fine
        coarse_to_fine = True
    else:
        loss_change_thresh = .1
    if model.cone_power < 1:
        clip_grad_norm = 1
    else:
        clip_grad_norm = False
    start_time = time.time()
    matched_im, matched_rep = metamer.synthesize(clamper=clamper, store_progress=10,
                                                 learning_rate=learning_rate, max_iter=max_iter,
                                                 loss_thresh=loss_thresh, seed=seed,
                                                 initial_image=initial_image,
                                                 clamp_each_iter=clamp_each_iter,
                                                 save_progress=save_progress,
                                                 optimizer=optimizer, fraction_removed=fraction_removed,
                                                 loss_change_fraction=loss_change_fraction,
                                                 loss_change_thresh=loss_change_thresh,
                                                 coarse_to_fine=coarse_to_fine,
                                                 clip_grad_norm=clip_grad_norm,
                                                 save_path=save_path.replace('.pt', '_inprogress.pt'))
    duration = time.time() - start_time
    # make sure everything's on the cpu for saving
    metamer = metamer.to('cpu')
    if save_path is not None:
        summarize(metamer, save_path.replace('.pt', '_summary.csv'),
                  duration_human_readable=convert_seconds_to_str(duration), duration=duration,
                  optimizer=optimizer, fraction_removed=fraction_removed, model=model_name,
                  target_image=image_name, seed=seed, learning_rate=learning_rate,
                  loss_change_thresh=loss_change_thresh, coarse_to_fine=coarse_to_fine,
                  loss_change_fraction=loss_change_fraction, initial_image=initial_image_type,
                  min_ecc=min_ecc, max_ecc=max_ecc, max_iter=max_iter, gpu_id=gpu_id,
                  loss_thresh=loss_thresh, scaling=scaling, clamper=clamper_name,
                  clamp_each_iter=clamp_each_iter, clip_grad_norm=clip_grad_norm,
                  image_name=op.basename(image_name).replace('.pgm', '').replace('.png', ''),
                  loss_function=loss_func)
        save(save_path, metamer, animate_figsize, rep_figsize, img_zoom)
    if save_progress:
        os.remove(save_path.replace('.pt', '_inprogress.pt'))
