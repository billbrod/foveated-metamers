#!/usr/bin/env python3
"""create MAD Competition images, analogous to create_metamers.py

Note that, unlike create_metamers.py, this uses the synthesis methods within
plenoptic.

"""
from . import create_metamers
import re
import torch
import plenoptic as po
import pyrtools as pt
import time
import numpy as np
import os.path as op
import imageio
from .utils import convert_im_to_int
import matplotlib as mpl


def setup_metric(metric_name, image, min_ecc, max_ecc, cache_dir,
                 normalize_dict=None, gpu_id=None):
    r"""Setup the metric.

    We initialize the metric, with the specified parameters.

    `metric_name` must either specify a metric (currently: 'ssim' or 'mse') or
    one of our foveated models. If a foveated model, it must be constructed of
    several parts, for which you have several chocies:
    `'{visual_area}{options}_{window_type}_scaling-{scaling}'`:
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
    - `scaling`: float giving the scaling values of these models

    The recommended metric_name values that correspond to our foveated models
    are: `RGC_norm_gaussian_scaling-{scaling}`,
    `V1_norm_s6_gaussian_scaling-{scaling}` (pick whatever scaling value you
    like).

    Parameters
    ----------
    metric_name : str
        str specifying which of the `PooledVentralStream` models we should
        initialize or another metric. See above for more details.
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
    metric : function
        A metric that gives the distance between two images whose pixel values
        lie between 0 and 1.
    model : torch.nn.Module
        The model used to create the metric. If metric is not based on a model,
        this is None.

    """
    if metric_name in ['ssim', 'mse']:
        if metric_name == 'ssim':
            def ssim(*args):
                return po.metric.ssim(*args, weighted=True, pad='reflect', dynamic_range=255)
            metric = ssim
        elif metric_name == 'mse':
            metric = po.metric.mse
        model = None
    else:
        metric_name, scaling = re.findall('([a-zA-z_]+)_scaling-([0-9.]+)', metric_name)[0]
        model = create_metamers.setup_model(metric_name, float(scaling), image,
                                            min_ecc, max_ecc, cache_dir,
                                            normalize_dict)[0]
        model = create_metamers.setup_device(model, gpu_id=gpu_id)[0]
        # do this because we want the metric name to reflect which model we're
        # using. We'll never be comparing the same model family against itself
        # (e.g., two V1 models with different scaling), so we just use visual
        # area tag
        if metric_name.startswith('RGC'):
            def fov_rgc_metric(x1, x2):
                return po.metric.mse(model(x1), model(x2))
            metric = fov_rgc_metric
        elif metric_name.startswith('V1'):
            def fov_v1_metric(x1, x2):
                return po.metric.mse(model(x1), model(x2))
            metric = fov_v1_metric
    return metric, model


def _get_min_window_ecc(model, units='pixels', scale=0):
    """Get eccentricities where pooling windows exceed 1 degree in area.

    Currently, only works for Gaussian windows (will raise an exception
    otherwise).

    See pooling.py and pooling_windows.py for more details on the "full" and
    "half" areas.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check.
    units : {'pixels', 'degrees'}, optional
        Whether to return the eccentricities in pixels or degrees
    scale : int, optional
        The scale of the window to check. Only matters if units == 'pixels'.

    Returns
    -------
    full_min_ecc : float
        Eccentricity where the full area exceeds 1 pixel
    half_min_ecc : float
        Eccentricity where the area at full-width half-max exceeds 1 pixel.

    """
    if not hasattr(model, 'PoolingWindows'):
        raise Exception("Model must have a PoolingWindows attribute!")
    if model.window_type != 'gaussian':
        raise Exception("Currently, only gaussian windows are supported!")
    if units == 'pixels':
        full_idx = (model.window_approx_area_pixels[scale]['full'] < 1).argmin()
        full_min_ecc = model.central_eccentricity_pixels[scale][full_idx]
        half_min_ecc = model.calculated_min_eccentricity_pixels[scale]
    elif units == 'degrees':
        full_idx = (model.window_approx_area_degrees[scale]['full'] < 1).argmin()
        full_min_ecc = model.central_eccentricity_degrees[scale][full_idx]
        half_min_ecc = model.calculated_min_eccentricity_degrees[scale]
    return full_min_ecc, half_min_ecc


def plot_image_diff(mad, fix_model=None, synthesis_model=None):
    """Create image difference figure.

    If either metric is based on a model, we will plot contours showing where
    their windows get larger than a single pixel in area (because that's
    relevant for understanding why the MAD images look the way they do).

    Parameters
    ----------
    mad : plenoptic.synth.MADCompetition
        The mad object after synthesis
    fix_model : torch.nn.Module or None, optional
        The model used to create the fix_metric. If metric is not based on a
        model, this is None.
    synthesis_model : torch.nn.Module or None, optional
        The model used to create the synthesis_metric. If metric is not based
        on a model, this is None.

    Returns
    -------
    fig :
        Figure containing the plot.

    """
    if fix_model is not None:
        fix_full, fix_half = _get_min_window_ecc(fix_model)
    if synthesis_model is not None:
        synthesis_full, synthesis_half = _get_min_window_ecc(synthesis_model)
    imgs = [mad.reference_signal, mad.synthesized_signal,
            mad.synthesized_signal - mad.reference_signal,
            mad.reference_signal, mad.synthesized_signal,
            mad.synthesized_signal - mad.reference_signal]
    titles = ['Reference image',
              (f'MAD image\n{mad.synthesis_target}imize {mad.synthesis_metric.__name__}'
               f'\n({mad.fixed_metric.__name__} held constant)'),
              'MAD image - reference image',
              'Initial image', '',
              'MAD image - initial image']
    # want the same vrange is for the four images that aren't differences
    vrange = (min([im.min() for im in [*imgs[:2], *imgs[3:5]]]),
              max([im.max() for im in [*imgs[:2], *imgs[3:5]]]))
    vranges = [vrange, vrange, 'indep0', vrange, vrange, 'indep0']
    fig = pt.make_figure(2, 3, imgs[0].shape[-2:])
    for im, ax, title, vr in zip(imgs, fig.axes, titles, vranges):
        po.imshow(im, vrange=vr, title=title, ax=ax, zoom=1)
        # then we're plotting the difference image
        if vr == 'indep0':
            handles = []
            labels = []
            if fix_model is not None:
                fix_full_circ = mpl.patches.Circle([s//2 for s in im.shape[-2:]],
                                                   round(fix_full), fc='none', ec='r')
                ax.add_artist(fix_full_circ)
                fix_half_circ = mpl.patches.Circle([s//2 for s in im.shape[-2:]],
                                                   round(fix_half), fc='none', ec='r',
                                                   linestyle='--')
                ax.add_artist(fix_half_circ)
                handles += [fix_full_circ, fix_half_circ]
                labels += [f'{mad.fixed_metric.__name__} window full area threshold',
                           f'{mad.fixed_metric.__name__} window FWHM area threshold']
            if synthesis_model is not None:
                synth_full_circ = mpl.patches.Circle([s//2 for s in im.shape[-2:]],
                                                     round(synthesis_full), fc='none', ec='b')
                ax.add_artist(synth_full_circ)
                synth_half_circ = mpl.patches.Circle([s//2 for s in im.shape[-2:]],
                                                     round(synthesis_half), fc='none', ec='b',
                                                     linestyle='--')
                ax.add_artist(synth_half_circ)
                handles += [synth_full_circ, synth_half_circ]
                labels += [f'{mad.synthesis_metric.__name__} window full area threshold',
                           f'{mad.synthesis_metric.__name__} window FWHM area threshold']
    # only need only plot of the synthesized image, so hide the second one and
    # move the first one down.
    fig.axes[-2].set_visible(False)
    ax = fig.axes[1]
    # inspired by
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom-.5*height, width, height])
    fig.legend(handles, labels, frameon=False, loc='center left', bbox_to_anchor=(1, .5),
               bbox_transform=fig.transFigure, borderaxespad=0)
    return fig


def save(save_path, mad, fix_model=None, synthesis_model=None):
    """Save MADCompetition object and its outputs.

    We save the object itself, plus:
    - The finished MADimage in its original float32 format (with
      values between 0 and 1, as a numpy array), at
      ``os.path.splitext(save_path)[0] + "_mad.npy"``.
    - The finished metamer 8-bit image, at
      ``os.path.splitext(save_path)[0] + "_mad.png"``.
    - The video showing synthesis progress at
      ``os.path.splitext(save_path)[0] + "_synthesis.mp4"``. We use this
      to visualize the optimization progress.
    - Picture showing synthesis progress summary at
      ``os.path.splitext(save_path)[0] + "_synthesis.png"``.
    - Picture showing difference between synthesized image and the reference
      and initial images at
    ``os.path.splitext(save_path)[0] + "_image-diff.png"``.

    Parameters
    ----------
    save_path : str
        The path to save the MADCompetition object at, which we use as a
        starting-point for the other save paths
    mad : plenoptic.synth.MADCompetition
        The mad object after synthesis
    fix_model : torch.nn.Module or None, optional
        The model used to create the fix_metric. If metric is not based on a
        model, this is None.
    synthesis_model : torch.nn.Module or None, optional
        The model used to create the synthesis_metric. If metric is not based
        on a model, this is None.

    """
    print("Saving at %s" % save_path)
    mad.save(save_path)
    # save png of mad
    mad_path = op.splitext(save_path)[0] + "_mad.png"
    mad_image = po.to_numpy(mad.synthesized_signal).squeeze()
    print("Saving mad float32 array at %s" % mad_path.replace('.png', '.npy'))
    np.save(mad_path.replace('.png', '.npy'), mad_image)
    print("Saving mad image at %s" % mad_path)
    # this already lies between 0 and 255, so we convert it to ints
    imageio.imwrite(mad_path, mad_image.astype(np.uint8))
    synthesis_path = op.splitext(save_path)[0] + "_synthesis.png"
    print(f"Saving synthesis image at {synthesis_path}")
    fig, _ = po.synth.mad_competition.plot_synthesis_status(mad)
    fig.axes[-1].set(yscale='log')
    fig.savefig(synthesis_path)
    video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
    print(f"Saving synthesis video at {video_path}")
    anim = po.synth.mad_competition.animate(mad)
    anim.save(video_path)
    diff_path = op.splitext(save_path)[0] + "_image-diff.png"
    print(f"Saving image diff figure at {diff_path}")
    fig = plot_image_diff(mad, fix_model, synthesis_model)
    fig.savefig(diff_path, bbox_inches='tight')


def main(fix_metric_name, synthesis_metric_name, image, synthesis_target,
         seed=0, min_ecc=.5, max_ecc=15, learning_rate=1, max_iter=100,
         stop_criterion=1e-4, stop_iters_to_check=50, save_path=None,
         initial_image=.2, gpu_id=None, cache_dir=None,
         fix_metric_normalize_dict=None, synthesis_metric_normalize_dict=None,
         optimizer='Adam', metric_tradeoff_lambda=None,
         range_penalty_lambda=.1, continue_path=None, num_threads=None):
    r"""Create MAD images.

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    `fix_metric_name` and `synthesis_metric_name` must either specify a metric
    (currently: 'ssim' or 'mse') or one of our foveated models. If a foveated
    model, it must be constructed of several parts, for which you have several
    chocies:
    `'{visual_area}{options}_{window_type}_scaling-{scaling}'`:
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
    - `scaling`: float giving the scaling values of these models

    The recommended model_name values are:
    `RGC_norm_gaussian_scaling-{scaling}` and
    `V1_norm_s6_gaussian_scaling-{scaling}` (pick whatever scaling value ou
    like). If one of our foveated models is specified, the corresponding metric
    is the MSE between the representation of two images.

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
    fix_metric_name, synthesis_metric_name : str
        str specifying which of the `PooledVentralStream` models we should
        use or another metric. See above for more details.
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    synthesis_target : {'min', 'max'}
        Whether we're minimizing or maximizing synthesis_metric.
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
    stop_criterion : float, optional
        The stop criterion. If the loss has changed by less than this over the
        past stop_iters_to_check iterations, we quit out. Corresponds to
        loss_thresh in create_metamers.py
    stop_iters_to_check : int, optional
        How many iterations back to check in order to see if the loss has
        stopped decreasing. Corresponds to loss_change_iter in
        create_metamers.py
    save_path : str or None, optional
        If a str, the path to the file to save the metamer object to. If
        None, we don't save the synthesis output (that's probably a bad
        idea)
    initial_image : float, optional
        std dev of Gaussian noise added to image to initialize synthesis.
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
    fix_metric_normalize_dict, fix_metric_normalize_dict : str or None, optional
        If a str, the path to the dictionary containing the statistics to use
        for normalization for the fix_metric and synthesis_metric models,
        respectively. If None, we don't normalize anything
    optimizer: {'Adam', 'SGD'}
        The choice of optimization algorithm
    metric_tradeoff_lambda :
        Lambda to multiply by ``fixed_metric`` loss and add to
        ``synthesis_metric`` loss. If ``None``, we pick a value so the two
        initial losses are approximately equal in magnitude.
    range_penalty_lambda :
        Lambda to multiply by range penalty and add to loss.
    continue_path : str or None, optional
        If None, we synthesize a new metamer. If str, this should be the
        path to a previous synthesis run, which we are resuming. In that
        case, you may set learning_rate to None (in which case we resume
        where we left off) and set max_iter to a different value (the
        number of extra iterations to run) otherwise the rest of the
        arguments should be the same as the first run.
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
    po.tools.set_seed(seed)
    # for this, we want the images to be between 0 and 255
    image = 255 * create_metamers.setup_image(image)
    print(f"Using initial noise level {initial_image}")
    # this will be false if normalize_dict is None or an empty list
    if fix_metric_normalize_dict:
        fix_metric_normalize_dict = torch.load(fix_metric_normalize_dict)
    if synthesis_metric_normalize_dict:
        synthesis_metric_normalize_dict = torch.load(synthesis_metric_normalize_dict)
    fix_metric, fix_model = setup_metric(fix_metric_name, image, min_ecc,
                                         max_ecc, cache_dir,
                                         fix_metric_normalize_dict)
    fix_metric_str = f"Using fix metric {fix_metric_name}"
    if fix_metric_name not in ['mse', 'ssim']:
        fix_metric_str += f" from {min_ecc} degrees to {max_ecc} degrees"
    synthesis_metric, synthesis_model = setup_metric(synthesis_metric_name,
                                                     image, min_ecc, max_ecc,
                                                     cache_dir,
                                                     synthesis_metric_normalize_dict)
    synthesis_metric_str = f"Will {synthesis_target} synthesis metric {synthesis_metric_name}"
    if synthesis_metric_name not in ['mse', 'ssim']:
        synthesis_metric_str += f" from {min_ecc} degrees to {max_ecc} degrees"
    print(synthesis_metric_str)
    print(fix_metric_str)
    image = create_metamers.setup_device(image, gpu_id=gpu_id)[0]
    # want to set store_progress before we potentially change max_iter below,
    # because if we're resuming synthesis, want to have the same store_progress
    # arg. don't want to store too often, otherwise we slow down and use too
    # much memory. this way we store at most 100 time points
    store_progress = max(10, max_iter//100)
    mad = po.synth.MADCompetition(image, synthesis_metric, fix_metric, synthesis_target, initial_image,
                                  metric_tradeoff_lambda, range_penalty_lambda, allowed_range=(0, 255))
    print(f"Using optimizer {optimizer}")
    if optimizer == 'Adam':
        opt = torch.optim.Adam([mad.synthesized_signal], lr=learning_rate, amsgrad=True)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD([mad.synthesized_signal], lr=learning_rate)
    if continue_path is not None:
        print("Resuming synthesis saved at %s" % continue_path)
        mad = mad.load(continue_path)
        opt = None
    print(f"Using learning rate {learning_rate}, stop_criterion {stop_criterion} (stop_iters_to_check "
          f"{stop_iters_to_check}), and max_iter {max_iter}")
    start_time = time.time()
    # note that there's a possibility that max_iter=0 (in particular, if we're
    # loading in an inprogress.pt file that finished synthesis but had trouble
    # when saving outputs). we still want to call synthesize, because there's a
    # small amount of wrapping up that needs to happen
    mad.synthesize(max_iter=max_iter, optimizer=opt,
                   store_progress=store_progress,
                   stop_criterion=stop_criterion,
                   stop_iters_to_check=stop_iters_to_check)
    duration = time.time() - start_time
    print(f"Synthesis took {duration} seconds")
    # make sure everything's on the cpu for saving
    mad.to('cpu')
    if save_path is not None:
        save(save_path, mad, fix_model, synthesis_model)
