#!/usr/bin/python
"""create metamers for the experiment
"""
import torch
import GPUtil
import imageio
import warnings
import os
import numpy as np
import plenoptic as po
import pyrtools as pt
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
# by default matplotlib uses the TK gui toolkit which can cause problems
# when I'm trying to render an image into a file, see
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
mpl.use('Agg')


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
        # use imageio.imread in order to handle rgb correctly. this uses the ITU-R 601-2 luma
        # transform, same as matlab
        image = imageio.imread(image, as_gray=True)
    if image.max() > 1:
        warnings.warn("Assuming image range is (0, 255)")
        image /= 255
    else:
        warnings.warn("Assuming image range is (0, 1)")
    image = torch.tensor(image, dtype=torch.float32)
    while image.ndimension() < 4:
        image = image.unsqueeze(0)
    return image


def setup_model(model_name, scaling, image, min_ecc, max_ecc, cache_dir, normalize_dict=None):
    r"""setup the model

    We initialize the model, with the specified parameters, and return
    it with the appropriate figsizes.

    Parameters
    ----------
    model_name : {'RGC', 'V1', 'V1-norm'}
        Which type of model to create.
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
        if the model is "V1-norm". In any other case, we'll throw an
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

    """
    if model_name == 'RGC':
        if normalize_dict is not None:
            raise Exception("Cannot normalize RGC model!")
        model = po.simul.RetinalGanglionCells(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                              max_eccentricity=max_ecc, transition_region_width=1,
                                              cache_dir=cache_dir)
        animate_figsize = (17, 5)
        rep_image_figsize = (4, 13)
        # default figsize arguments work for an image that is 256x256,
        # may need to expand. we go backwards through figsize because
        # figsize and image shape are backwards of each other:
        # image.shape's last two indices are (height, width), while
        # figsize is (width, height)
        animate_figsize = tuple([s*max(1, image.shape[::-1][i]/256) for i, s in
                                 enumerate(animate_figsize)])
        rep_image_figsize = tuple([s*max(1, image.shape[::-1][i]/256) for i, s in
                                   enumerate(rep_image_figsize)])
        rescale_factor = np.mean([image.shape[i+2]/256 for i in range(2)])
    elif model_name.startswith('V1'):
        if not model_name.endswith('norm'):
            if normalize_dict is not None:
                raise Exception("Cannot normalize V1 model!")
            normalize_dict = {}
        if normalize_dict is None and model_name.endswith('norm'):
            raise Exception("If model_name is V1-norm, normalize_dict must be set!")
        model = po.simul.PrimaryVisualCortex(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                             max_eccentricity=max_ecc, transition_region_width=1,
                                             cache_dir=cache_dir, normalize_dict=normalize_dict)
        animate_figsize = (35, 11)
        rep_image_figsize = (27, 15)
        # default figsize arguments work for an image that is 512x512,
        # may need to expand. we go backwards through figsize because
        # figsize and image shape are backwards of each other:
        # image.shape's last two indices are (height, width), while
        # figsize is (width, height)
        animate_figsize = tuple([s*max(1, image.shape[::-1][i]/512) for i, s in
                                 enumerate(animate_figsize)])
        # default rep_image_figsize arguments are for 256x256 image
        rep_image_figsize = tuple([s*max(1, image.shape[::-1][i]/256) for i, s in
                                   enumerate(rep_image_figsize)])
        rescale_factor = np.mean([image.shape[i+2]/512 for i in range(2)])
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    # 10 and 12 are the default font sizes for labels and titles,
    # respectively, and we want to scale them in order to keep them
    # readable. this should be global to matplotlib and so propagate
    # through
    mpl.rc('axes', labelsize=rescale_factor*10, titlesize=rescale_factor*12)
    mpl.rc('xtick', labelsize=rescale_factor*10)
    mpl.rc('ytick', labelsize=rescale_factor*10)
    mpl.rc('lines', linewidth=rescale_factor*1.5, markersize=rescale_factor*6)
    return model, animate_figsize, rep_image_figsize


def add_center_to_image(model, initial_image, reference_image):
    r"""Add the center back to the metamer image

    The VentralStream class of models will do nothing to the center of
    the image (they don't see the fovea), so we add the fovea to the
    initial image before synthesis.

    Parameters
    ----------
    model : plenoptic.simul.VentralStream
        The model used to create the metamer. Specifically, we need its
        windows attribute
    initial_image : torch.Tensor
        The initial image we will use for metamer synthesis. Probably a
        bunch of white noise
    reference_image : torch.Tensor
        The reference/target image for synthesis
        (``metamer.target_image``)

    Returns
    -------
    metamer_image : torch.Tensor
        The metamer image with the center added back in

    """
    windows = torch.einsum('ahw,ehw->hw', [model.PoolingWindows.angle_windows[0],
                                           model.PoolingWindows.ecc_windows[0]])
    # for some reason ~ (invert) is not implemented for booleans in
    # pytorch yet, so we do this instead.
    return ((windows * initial_image) + ((1 - windows) * reference_image))


def summary_plots(metamer, rep_image_figsize):
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

    Returns
    -------
    rep_fig : matplotlib.figure.Figure
        The figure containing the 'rep_image' plot
    windowed_fig : matplotlib.figure.Figure
        The figure containing the 'windowed' plot

    """
    rep_fig, axes = plt.subplots(3, 1, figsize=rep_image_figsize)
    images = [metamer.model(metamer.target_image), metamer.model(metamer.matched_image),
              metamer.representation_error()]
    titles = ['Reference image |', 'Metamer |', 'Error |']
    vranges = ['indep1', 'indep1', 'indep0']
    for i, (im, t, vr) in enumerate(zip(images, titles, vranges)):
        metamer.model.plot_representation_image(ax=axes[i], data=im, title=t, vrange=vr)
    images = [metamer.saved_image[0], metamer.matched_image, metamer.target_image]
    images = 2*[po.to_numpy(i.to(torch.float32)).squeeze() for i in images]
    titles = ['Initial image', 'Metamer', 'Reference image']
    titles += ['Windowed '+t for t in titles]
    windowed_fig = pt.imshow(images, col_wrap=3, title=titles, vrange=(0, 1))
    for ax in windowed_fig.axes[3:]:
        metamer.model.plot_windows(ax)
    return rep_fig, windowed_fig


def save(save_path, metamer, animate_figsize, rep_image_figsize):
    r"""save the metamer output

    We save five things here:
    - The metamer object itself, at ``save_path``. This contains, among
      other things, the saved image and representation over the course
      of synthesis.
    - The finished metamer image, at ``os.path.splitext(save_path)[0] +
      "_metamer.png"``.
    - The 'rep_image', at ``os.path.splitext(save_path)[0]+"_rep.png"``.
      See ``summary_plots()`` docstring for a description of this plot.
    - The 'windowed_image', at ``os.path.splitext(save_path)[0] +
      "_windowed.png"``. See ``summary_plots()`` docstring for a
      description of this plot.
    - The video showing synthesis progress at
      ``os.path.splitext(save_path)[0] + "_synthesis.mp4"``. We use this
      to visualize the optimization progress.

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

    """
    print("Saving at %s" % save_path)
    metamer.save(save_path, save_model_reduced=True)
    # save png of metamer
    metamer_path = op.splitext(save_path)[0] + "_metamer.png"
    print("Saving metamer image at %s" % metamer_path)
    imageio.imwrite(metamer_path, po.to_numpy(metamer.matched_image).squeeze())
    video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
    rep_fig, windowed_fig = summary_plots(metamer, rep_image_figsize)
    rep_path = op.splitext(save_path)[0] + "_rep.png"
    print("Saving representation image at %s" % rep_path)
    rep_fig.savefig(rep_path)
    windowed_path = op.splitext(save_path)[0] + "_windowed.png"
    print("Saving windowed image at %s" % windowed_path)
    windowed_fig.savefig(windowed_path)
    print("Saving synthesis video at %s" % video_path)
    anim = metamer.animate(figsize=animate_figsize)
    anim.save(video_path)


def setup_initial_image(initial_image_type, model, image):
    r"""setup the initial image

    Parameters
    ----------
    initial_image_type : {'white', 'pink', 'gray', 'blue'}
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere (note
        that this one should only be used for the RGC model; it will
        immediately break the V1 and V2 models, since it has no energy
        at many frequencies)
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
    else:
        raise Exception("Don't know how to handle initial_image_type %s! Must be one of {'white',"
                        " 'gray', 'pink', 'blue'}" % initial_image_type)
    initial_image = add_center_to_image(model, initial_image, image)
    return torch.nn.Parameter(initial_image)


def setup_device(*args, use_cuda=False):
    r"""Setup device and get everything onto it

    This simple function checks whether ``torch.cuda.is_available()``
    and ``use_cuda`` are True and, if so, uses GPUtil to try and find
    the first available and un-used one. If not, we use the cpu as the
    device

    We then call a.to(device) for every a in args (so this can be called
    with an arbitrary number of objects, each of which just needs to have
    .to method).

    Note that we always return a list (even if you only pass one item),
    so if you pass a single object, you'll need to either grab it
    specifically, either by doing ``im = setup_device(im,
    use_cuda=True)[0]`` or ``im, = setup_device(im)`` (notice the
    comma).

    Parameters
    ----------
    args :
        Some number of torch objects that we want to get on the proper
        device
    use_cuda : bool, optional
        Whether to try and use the GPU or not. Note that, to set this,
        you must set it as a keyword, i.e., ``setup_device(im, True)``
        won't work but ``setup_device(im, use_cuda=True)`` will (this is
        because the ``*args`` in our function signature will greedily
        grab every non-keyword argument).

    Returns
    -------
    args : list
        Every item we were passed in arg, now on the proper device
    """
    if torch.cuda.is_available() and use_cuda:
        gpu_num = GPUtil.getAvailable(order='first', maxLoad=.1, maxMemory=.1, includeNan=False)[0]
        device = torch.device("cuda:%s" % gpu_num)
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
         loss_thresh=1e-4, save_path=None, initial_image_type='white', use_cuda=False,
         cache_dir=None, normalize_dict=None, num_gpus=0, optimizer='SGD', fraction_removed=0,
         loss_change_fraction=1):
    r"""create metamers!

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    Parameters
    ----------
    model_name : {'RGC', 'V1', 'V1-norm'}
        Which type of model to create.
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
        The loss threshold. If our loss is every below this, we stop
        synthesis and consider ourselves done.
    save_path : str or None, optional
        If a str, the path to the file to save the metamer object to. If
        None, we don't save the synthesis output (that's probably a bad
        idea)
    initial_image_type : {'white', 'pink', 'gray', 'blue'}
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere (note
        that this one should only be used for the RGC model; it will
        immediately break the V1 and V2 models, since it has no energy
        at many frequencies)
    use_cuda : bool, optional
        If True and if torch.cuda.is_available(), we try to use find a
        gpu we can use. We do this with GPUtil. else, we use the cpu
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    normalize_dict : str or None, optional
        If a str, the path to the dictionary containing the statistics
        to use for normalization. If None, we don't normalize anything
    num_gpus : int, optional
        The number of gpus to use. If use_cuda is False, this must be
        0. Otherwise, if it's greater than 1, we'll use
        ``torch.nn.DataParallel`` to try and spread it across multiple
        GPUs.
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

    """
    print("Using seed %s" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    image = setup_image(image)
    if normalize_dict is not None:
        normalize_dict = torch.load(normalize_dict)
    model, animate_figsize, rep_figsize = setup_model(model_name, scaling, image, min_ecc, max_ecc,
                                                      cache_dir, normalize_dict)
    print("Using model %s from %.02f degrees to %.02f degrees" % (model_name, min_ecc, max_ecc))
    print("Using learning rate %s, loss_thresh %s, and max_iter %s" % (learning_rate, loss_thresh,
                                                                       max_iter))
    clamper = po.RangeClamper((0, 1))
    initial_image = setup_initial_image(initial_image_type, model, image)
    if num_gpus <= 1:
        image, initial_image, model = setup_device(image, initial_image, model, use_cuda=use_cuda)
    else:
        # in this case, we're going to parallelize the model anyway, so
        # don't put it on a single device
        image, initial_image = setup_device(image, initial_image, use_cuda=use_cuda)
    if num_gpus > 0:
        if not use_cuda:
            raise Exception("Can only use GPUs if use_cuda is True!")
        if num_gpus > 1:
            gpus = GPUtil.getAvailable(maxLoad=.5, maxMemory=.5, limit=num_gpus, order='first')
            # make sure the original gpu is on the list, or we throw an exception
            if image.device.index not in gpus:
                if len(gpus) < num_gpus:
                    gpus += [image.device.index]
                else:
                    gpus = [image.device.index] + gpus[:-1]
            print("Will put device on multiple gpus: %s" % gpus)
            model = model.parallel(gpus)
    metamer = po.synth.Metamer(image, model)
    if save_path is not None:
        save_progress = True
    else:
        save_progress = False
    matched_im, matched_rep = metamer.synthesize(clamper=clamper, store_progress=10,
                                                 learning_rate=learning_rate, max_iter=max_iter,
                                                 loss_thresh=loss_thresh, seed=seed,
                                                 initial_image=initial_image,
                                                 save_progress=save_progress,
                                                 optimizer=optimizer, fraction_removed=fraction_removed,
                                                 loss_change_fraction=loss_change_fraction,
                                                 loss_change_thresh=.1,
                                                 save_path=save_path.replace('.pt', '_inprogress.pt'))
    if save_path is not None:
        save(save_path, metamer, animate_figsize, rep_figsize)
    os.remove(save_path.replace('.pt', '_inprogress.pt'))
