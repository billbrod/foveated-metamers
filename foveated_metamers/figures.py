"""code to generate figures for the project
"""
import imageio
import torch
import re
import numpy as np
import pyrtools as pt
import plenoptic as po
from skimage import measure
import matplotlib.pyplot as plt
import os.path as op
from . import utils, create_metamers

V1_TEMPLATE_PATH = op.join('/home/billbrod/Desktop/metamers', 'metamers_display', 'V1_norm_s6_'
                           'gaussian', '{image_name}', 'scaling-{scaling}', 'opt-Adam',
                           'fr-0_lc-1_cf-0.01_clamp-True', 'seed-{seed}_init-white_lr-0.01'
                           'rate}_e0-0.5_em-30.2_iter-{max_iter}_thresh-1e-08_gpu-{gpu}_metamer_'
                           'gamma-corrected.png')
RGC_TEMPLATE_PATH = op.join('/home/billbrod/Desktop/metamers', 'metamers_display', 'RGC_gaussian',
                            '{image_name}', 'scaling-{scaling}', 'opt-Adam', 'fr-0_lc-'
                            '1_cf-0_clamp-True', 'seed-{seed}_init-white_lr-0.01_e0-3.71_em-30.2_'
                            'iter-750_thresh-1e-08_gpu-0_metamer_gamma-corrected.png')


def add_cutout_box(axes, window_size=400, periphery_offset=(-800, -1000), colors='r',
                   linestyles='--', plot_fovea=True, plot_periphery=True, **kwargs):
    """add square to axes to show where the cutout comes from

    Parameters
    ----------
    axes : array_like
        arrays to add square to (different images should be indexed
        along first dimension)
    window_size : int
        The size of the cut-out to plot, in pixels (this is the length
        of one side of the square).
    periphery_offset : tuple
        Tuple of ints. How far from the fovea we want our peripheral
        cut-out to be. The order of this is the same as that returned by
        image.shape. Can be positive or negative depending on which
        direction you want to go
    colors, linestyle : str, optional
        color and linestyle to use for cutout box, see `plt.vlines()`
        and `plt.hlines()` for details
    plot_fovea : bool, optional
        whether to plot the foveal box
    plot_periphery : bool, optional
        whether to plot peripheral box
    kwargs :
        passed to `plt.vlines()` and `plt.hlines()`

    """
    axes = np.array(axes).flatten()
    for ax in axes:
        if len(ax.images) != 1:
            raise Exception("axis should only have one image on it!")
        im = ax.images[0]
        im_ctr = [s//2 for s in im.get_size()]
        fovea_bounds = np.array([im_ctr[0]-window_size//2, im_ctr[0]+window_size//2,
                                 im_ctr[1]-window_size//2, im_ctr[1]+window_size//2])
        if plot_fovea:
            ax.vlines(fovea_bounds[2:], fovea_bounds[0], fovea_bounds[1], colors=colors,
                      linestyles=linestyles, **kwargs)
            ax.hlines(fovea_bounds[:2], fovea_bounds[2], fovea_bounds[3], colors=colors,
                      linestyles=linestyles, **kwargs)
        if plot_periphery:
            ax.vlines(fovea_bounds[2:]-periphery_offset[1], fovea_bounds[0]-periphery_offset[0],
                      fovea_bounds[1]-periphery_offset[0], colors=colors,
                      linestyles=linestyles, **kwargs)
            ax.hlines(fovea_bounds[:2]-periphery_offset[0], fovea_bounds[2]-periphery_offset[1],
                      fovea_bounds[3]-periphery_offset[1], colors=colors,
                      linestyles=linestyles, **kwargs)


def add_fixation_cross(axes, cross_size=50, colors='r', linestyles='-', **kwargs):
    """add fixation cross to center of axes

    Parameters
    ----------
    axes : array_like
        arrays to add square to (different images should be indexed
        along first dimension)
    cross_size : int, optional
        total size of the lines in the cross, in pixels
    colors, linestyle : str, optional
        color and linestyle to use for cutout box, see `plt.vlines()`
        and `plt.hlines()` for details
    kwargs :
        passed to `plt.vlines()` and `plt.hlines()`

    """
    axes = np.array(axes).flatten()
    for ax in axes:
        if len(ax.images) != 1:
            raise Exception("axis should only have one image on it!")
        im = ax.images[0]
        im_ctr = [s//2 for s in im.get_size()]
        ax.vlines(im_ctr[1], im_ctr[0]-cross_size/2, im_ctr[0]+cross_size/2, colors=colors,
                  linestyles=linestyles, **kwargs)
        ax.hlines(im_ctr[0], im_ctr[1]-cross_size/2, im_ctr[1]+cross_size/2, colors=colors,
                  linestyles=linestyles, **kwargs)


def get_image_cutout(images, window_size=400, periphery_offset=(-800, -1000)):
    """get foveal and peripheral cutouts from images

    Parameters
    ----------
    images : array_like
        images to plot (different images should be indexed along first
        dimension)
    window_size : int
        The size of the cut-out to plot, in pixels (this is the length
        of one side of the square).
    periphery_offset : tuple
        Tuple of ints. How far from the fovea we want our peripheral
        cut-out to be. The order of this is the same as that returned by
        image.shape. Can be positive or negative depending on which
        direction you want to go

    Returns
    -------
    fovea, periphery : list
        lists of foveal and peripheral cutouts from images

    """
    if images.ndim == 2:
        images = images[None, :]
    im_ctr = [s//2 for s in images.shape[-2:]]
    fovea_bounds = [im_ctr[0]-window_size//2, im_ctr[0]+window_size//2,
                    im_ctr[1]-window_size//2, im_ctr[1]+window_size//2]
    fovea = [im[fovea_bounds[0]:fovea_bounds[1], fovea_bounds[2]:fovea_bounds[3]] for im in images]
    periphery = [im[fovea_bounds[0]-periphery_offset[0]:fovea_bounds[1]-periphery_offset[0],
                    fovea_bounds[2]-periphery_offset[1]:fovea_bounds[3]-periphery_offset[1]]
                 for im in images]
    return fovea, periphery


def cutout_figure(images, window_size=400, periphery_offset=(-800, -1000), max_ecc=30.2,
                  plot_fovea=True, plot_periphery=True):
    """create figure showing cutout views of different images

    if both `plot_fovea` and `plot_periphery` are False, this just
    returns None

    Parameters
    ----------
    images : array_like
        images to plot (different images should be indexed along first
        dimension)
    window_size : int
        The size of the cut-out to plot, in pixels (this is the length
        of one side of the square).
    periphery_offset : tuple
        Tuple of ints. How far from the fovea we want our peripheral
        cut-out to be. The order of this is the same as that returned by
        image.shape. Can be positive or negative depending on which
        direction you want to go
    max_ecc : float, optional
        The maximum eccentricity of the metamers, as passed to the
        model. Used to convert from pixels to degrees so we know the
        extent and location of the cut-out views in degrees.
    plot_fovea : bool, optional
        whether to plot the foveal cutout
    plot_periphery : bool, optional
        whether to plot peripheral cutout

    Returns
    -------
    fig :
        The matplotlib figure with the cutouts plotted on it

    """
    if not plot_fovea and not plot_periphery:
        return None
    fovea, periphery = get_image_cutout(images, window_size, periphery_offset)
    # max_ecc is the distance from the center to the edge of the image,
    # so we want double this to get the full width of the image
    pix_to_deg = (2 * max_ecc) / max(images.shape[-2:])
    window_extent_deg = (window_size//2) * pix_to_deg
    periphery_ctr_deg = np.sqrt(np.sum([(s*pix_to_deg)**2 for s in periphery_offset]))
    imgs_to_plot = []
    if plot_fovea:
        imgs_to_plot += fovea
    if plot_periphery:
        imgs_to_plot += periphery
        if plot_fovea:
            periphery_ax_idx = len(fovea)
        else:
            periphery_ax_idx = 0
    fig = pt.imshow(imgs_to_plot, vrange=(0, 1), title=None, col_wrap=len(fovea))
    if plot_fovea:
        fig.axes[0].set(ylabel='Fovea\n($\pm$%.01f deg)' % window_extent_deg)
    if plot_periphery:
        ylabel = 'Periphery\n(%.01f$\pm$%.01f deg)' % (periphery_ctr_deg, window_extent_deg)
        fig.axes[periphery_ax_idx].set(ylabel=ylabel)
    return fig


def scaling_comparison_figure(model_name, image_name, scaling_vals, seed, window_size=400,
                              periphery_offset=(-800, -1000), max_ecc=30.2, **kwargs):
    r"""Create a figure showing cut-out views of all scaling values

    We want to be able to easily visually compare metamers across
    scaling values (and with the reference image), but they're very
    large. In order to facilitate this, we create this figure with
    'cut-out' views, where we compare the reference image and metamers
    made with a variety of scaling values (all same seed) at the fovea
    and the periphery, with some information about the extent.

    Parameters
    ----------
    model_name : str
        Name(s) of the model(s) to run. Must begin with either V1 or
        RGC. If model name is just 'RGC' or just 'V1', we will use the
        default model name for that brain area from config.yml
    image_name : str
        The name of the reference image we want to examine
    scaling_vals : list
        List of floats which give the scaling values to compare. We'll
        plot the metamers in this order, so they should probably be in
        increasing order
    seed : int
        The metamer seed we're examining
    window_size : int
        The size of the cut-out to plot, in pixels (this is the length
        of one side of the square).
    periphery_offset : tuple
        Tuple of ints. How far from the fovea we want our peripheral
        cut-out to be. The order of this is the same as that returned by
        image.shape. Can be positive or negative depending on which
        direction you want to go
    max_ecc : float, optional
        The maximum eccentricity of the metamers, as passed to the
        model. Used to convert from pixels to degrees so we know the
        extent and location of the cut-out views in degrees.
    kwargs : dict
        Additional key, value pairs to pass to
        utils.generate_metamers_path for finding the images to include.

    Returns
    -------
    fig :
        The matplotlib figure with the scaling comparison plotted on it

    """
    gamma_corrected_image_name = utils.get_gamma_corrected_ref_image(image_name)
    ref_path = utils.get_ref_image_full_path(gamma_corrected_image_name)
    images = [utils.convert_im_to_float(imageio.imread(ref_path))]
    image_paths = utils.generate_metamer_paths(model_name, image_name=image_name,
                                               scaling=scaling_vals, seed=seed,
                                               max_ecc=max_ecc, **kwargs)
    for p in image_paths:
        corrected_p = p.replace('.png', '_gamma-corrected.png')
        images.append(utils.convert_im_to_float(imageio.imread(corrected_p)))
    # want our images to be indexed along the first dimension
    images = np.einsum('ijk -> kij', np.dstack(images))
    fig = cutout_figure(images, window_size, periphery_offset, max_ecc)
    fig.axes[0].set(title='Reference')
    for i, sc in zip(range(1, len(images)), scaling_vals):
        fig.axes[i].set(title='scaling=%.03f' % sc)
    return fig


def pooling_window_size(windows, image, target_eccentricity=24,
                        windows_scale=0, **kwargs):
    """Plot example window on image.

    This plots a single window, as close to the target_eccentricity as
    possible, at half-max amplitude, to visualize the size of the pooling
    windows

    Parameters
    ----------
    windows : po.simul.PoolingWindows
        The PoolingWindows object to plot.
    image : np.ndarray or str
        The image to plot the window on. If a np.ndarray, then this should
        already lie between 0 and 1. If a str, must be the path to the image
        file,and we'll load it in.
    target_eccentricity : float, optional
        The approximate central eccentricity of the window to plot
    windows_scale : int, optional
        The scale of the windows to plot. If greater than 0, we down-sampled
        image by a factor of 2 that many times so they plot correctly.
    kwargs :
        Passed to pyrtools.imshow.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.

    """
    if isinstance(image, str):
        image = utils.convert_im_to_float(imageio.imread(image))
    target_ecc_idx = abs(windows.central_eccentricity_degrees -
                         target_eccentricity).argmin()
    ecc_windows = (windows.ecc_windows[windows_scale] /
                   windows.norm_factor[windows_scale])
    target_amp = windows.window_max_amplitude / 2
    window = torch.einsum('hw,hw->hw',
                          windows.angle_windows[windows_scale][0],
                          ecc_windows[target_ecc_idx])

    # need to down-sample image for these scales
    for i in range(windows_scale):
        image = measure.block_reduce(image, (2, 2))
    fig = pt.imshow(image, title=None, **kwargs)
    fig.axes[0].contour(po.to_numpy(window).squeeze(), [target_amp],
                        colors='r')
    return fig


def synthesis_video(metamer_save_path, model_name=None):
    """Create video showing synthesis progress, for presentations.

    Creates videos showing the metamer, representation, and pixels over time.
    Works best if synthesis was run with store_progress=1, or some other low
    value. Will create three videos, so can be used for a build. Will be saved
    in the same directory as metamer_save_path, replacing the extension with
    _synthesis-0.mp4, _synthesis-1.mp4, and synthesis-2.mp4

    WARNING: This will be very memory-intensive and may take a long time to
    run, depending on how many iterations synthesis ran for.

    Parameters
    ----------
    metamer_save_path : str
        Path to the .pt file containing the complete saved metamer
    model_name : str or None, optional
        str giving the model name. If None, we try and infer it from
        metamer_save_path

    """
    if model_name is None:
        # try to infer from path
        model_name = re.findall('/((?:RGC|V1)_.*?)/', path)[0]
    if model_name.startswith('RGC'):
        model_constructor = po.simul.PooledRGC.from_state_dict_reduced
    elif model_name.startswith('V1'):
        model_constructor = po.simul.PooledV1.from_state_dict_reduced
    metamer = po.synth.Metamer.load(metamer_save_path, model_constructor=model_constructor)
    animate_figsize, _, img_zoom = create_metamers.find_figsizes(model_name,
                                                                 metamer.model,
                                                                 metamer.base_signal.shape)
    width_ratios = [metamer.synthesized_signal.shape[-1] / metamer.synthesized_signal.shape[-2],
                    1, 1]
    vid_kwargs = {}
    for i in range(3):
        video_path = op.splitext(metamer_save_path)[0] + f"_synthesis-{i}.mp4"
        print(f"Saving synthesis-{i} video at {video_path}")
        figsize_2 = ((animate_figsize[0]-2) * .75 + 2, animate_figsize[1])
        fig, axes = plt.subplots(1, 3, figsize=figsize_2,
                                 subplot_kw={'aspect': 1},
                                 gridspec_kw={'width_ratios': width_ratios,
                                              'left': .05, 'right': .95})
        for j in range(i+1, 3):
            fig.axes[j].set_visible(False)
        for j in range(0, i+1):
            axes[j].locator_params(nbins=3)
        if i == 1:
            vid_kwargs['plot_rep_comparison'] = True
        elif i == 2:
            vid_kwargs['plot_signal_comparison'] = True
        anim = metamer.animate(fig=fig, imshow_zoom=img_zoom, plot_loss=False,
                               plot_representation_error=False, **vid_kwargs)
        anim.save(video_path)
