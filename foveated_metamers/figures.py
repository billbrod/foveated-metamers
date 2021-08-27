"""code to generate figures for the project
"""
import imageio
import itertools
import torch
import re
from fractions import Fraction
import pandas as pd
import numpy as np
import pyrtools as pt
import plenoptic as po
from skimage import measure
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as op
import arviz as az
from . import utils, plotting, analysis, mcmc, other_data
import sys
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages'))
import plenoptic_part as pop


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


def pooling_window_example(windows, image, target_eccentricity=24,
                           windows_scale=0, **kwargs):
    """Plot example window on image.

    This plots a single window, as close to the target_eccentricity as
    possible, at half-max amplitude, to visualize the size of the pooling
    windows

    Parameters
    ----------
    windows : pooling.PoolingWindows
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
                        colors='r', linewidths=5)
    return fig


def synthesis_schematic(metamer, iteration=0, plot_synthesized_image=True,
                        plot_rep_comparison=True, plot_signal_comparison=True,
                        **kwargs):
    """Create schematic of synthesis, for animating.

    WARNING: Currently, only works with images of size (256, 256), will need a
    small amount of tweaking to work with differently sized images. (And may
    never look quite as good with them)

    Parameters
    ----------
    metamer : pop.Metamer
        The Metamer object to grab data from
    iteration : int or None, optional
        Which iteration to display. If None, we show the most recent one.
        Negative values are also allowed.
    plot_synthesized_image : bool, optional
        Whether to plot the synthesized image or not.
    plot_rep_comparison : bool, optional
        Whether to plot a scatter plot comparing the synthesized and base
        representation.
    plot_signal_comparison : bool, optional
        Whether to plot the comparison of the synthesized and base
        images.
    kwargs :
        passed to metamer.plot_synthesis_status

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    axes_idx : dict
        dictionary specifying which plot is where, for use with animate()

    Notes
    -----
    To successfully animate, call with same values for the args that start with
    `plot_`, pass fig and axes_idx, and set init_figure, plot_loss,
    plot_representation_error to False.

    """
    # arrangement was all made with 72 dpi
    mpl.rc('figure', dpi=72)
    mpl.rc('axes', titlesize=25)
    image_shape = metamer.base_signal.shape
    figsize = ((1.5+(image_shape[-1] / image_shape[-2])) * 4.5 + 2.5, 3*4.5+1)
    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(3, 10, figure=fig, hspace=.25, bottom=.05,
                               top=.95, left=.05, right=.95)
    fig.add_subplot(gs[0, 0:3], aspect=1)
    fig.add_subplot(gs[0, 4:7], aspect=1)
    fig.add_subplot(gs[1, 1:4], aspect=1)
    fig.add_subplot(gs[1, 6:9], aspect=1)
    fig.add_subplot(gs[2, 0:3], aspect=1)
    fig.add_subplot(gs[2, 4:7], aspect=1)
    axes_idx = {'image': 0, 'signal_comp': 2, 'rep_comp': 3,
                'misc': [1] + list(range(4, len(fig.axes)))}
    po.imshow(metamer.base_signal, ax=fig.axes[4], title=None)
    if not plot_rep_comparison:
        axes_idx['misc'].append(axes_idx.pop('rep_comp'))
    if not plot_signal_comparison:
        axes_idx['misc'].append(axes_idx.pop('signal_comp'))
    for i in [0] + axes_idx['misc']:
        fig.axes[i].xaxis.set_visible(False)
        fig.axes[i].yaxis.set_visible(False)
        fig.axes[i].set_frame_on(False)
    model_axes = [5]
    if plot_synthesized_image:
        model_axes += [1]
    arrowkwargs = {'xycoords': 'axes fraction', 'textcoords': 'axes fraction',
                   'ha': 'center', 'va': 'center'}
    arrowprops = {'color': '0', 'connectionstyle': 'arc3', 'arrowstyle': '->',
                  'lw': 3}
    for i in model_axes:
        p = mpl.patches.Rectangle((0, .25), .5, .5, fill=False)
        p.set_transform(fig.axes[i].transAxes)
        fig.axes[i].add_patch(p)
        fig.axes[i].text(.25, .5, 'M', {'size': 50}, ha='center', va='center',
                         transform=fig.axes[i].transAxes)
        fig.axes[i].annotate('', (0, .5), (-.4, .5), arrowprops=arrowprops,
                             **arrowkwargs)
    if plot_rep_comparison:
        arrowprops['connectionstyle'] += ',rad=.3'
        fig.axes[5].annotate('', (1.2, 1.25), (.53, .5), arrowprops=arrowprops,
                             **arrowkwargs)
        if plot_synthesized_image:
            arrowprops['connectionstyle'] = 'arc3,rad=.2'
            fig.axes[1].annotate('', (.6, -.8), (.25, .22), arrowprops=arrowprops,
                                 **arrowkwargs)
    else:
        fig.axes[5].annotate('', (1.05, .5), (.53, .5), arrowprops=arrowprops,
                             **arrowkwargs)
        vector = "[{:.3f}, {:.3f}, ..., {:.3f}]".format(*np.random.rand(3))
        fig.axes[5].text(1.05, .5, vector, {'size': '25'}, transform=fig.axes[5].transAxes,
                         va='center', ha='left')
        if plot_synthesized_image:
            fig.axes[1].annotate('', (1.05, .5), (.53, .5), arrowprops=arrowprops,
                                 **arrowkwargs)
            vector = "[{:.3f}, {:.3f}, ..., {:.3f}]".format(*np.random.rand(3))
            fig.axes[1].text(1.05, .5, vector, {'size': '25'}, transform=fig.axes[1].transAxes,
                             va='center', ha='left')
    if plot_signal_comparison:
        arrowprops['connectionstyle'] = 'arc3'
        fig.axes[4].annotate('', (.8, 1.25), (.8, 1.03), arrowprops=arrowprops,
                             **arrowkwargs)
        if plot_synthesized_image:
            arrowprops['connectionstyle'] += ',rad=.1'
            fig.axes[0].annotate('', (.25, -.8), (.15, -.03), arrowprops=arrowprops,
                                 **arrowkwargs)
    fig = metamer.plot_synthesis_status(axes_idx=axes_idx, iteration=iteration,
                                        plot_rep_comparison=plot_rep_comparison,
                                        plot_synthesized_image=plot_synthesized_image,
                                        plot_loss=False,
                                        plot_signal_comparison=plot_signal_comparison,
                                        plot_representation_error=False, fig=fig,
                                        **kwargs)
    # plot_synthesis_status can update axes_idx
    axes_idx = metamer._axes_idx
    # I think plot_synthesis_status will turn this into a list (in the general
    # case, this can contain multiple plots), but for these models and Metamer,
    # it will always be a single value
    if 'rep_comp' in axes_idx.keys() and isinstance(axes_idx['rep_comp'], list):
        assert len(axes_idx['rep_comp']) == 1
        axes_idx['rep_comp'] = axes_idx['rep_comp'][0]
    fig.axes[0].set_title('')
    if plot_signal_comparison:
        fig.axes[2].set(xlabel='', ylabel='', title='Pixel values')
    if plot_rep_comparison:
        fig.axes[3].set(xlabel='', ylabel='')
    return fig, axes_idx


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
    mpl.rc('axes.spines', right=False, top=False)
    if model_name is None:
        # try to infer from path
        model_name = re.findall('/((?:RGC|V1)_.*?)/', metamer_save_path)[0]
    if model_name.startswith('RGC'):
        model_constructor = pop.PooledRGC.from_state_dict_reduced
    elif model_name.startswith('V1'):
        model_constructor = pop.PooledV1.from_state_dict_reduced
    metamer = pop.Metamer.load(metamer_save_path, model_constructor=model_constructor)
    kwargs = {'plot_synthesized_image': False, 'plot_rep_comparison': False,
              'plot_signal_comparison': False}
    formats = ['png', 'png', 'png', 'mp4', 'png', 'mp4']
    for i, f in enumerate(formats):
        path = op.splitext(metamer_save_path)[0] + f"_synthesis-{i}.{f}"
        print(f"Saving synthesis-{i} {f} at {path}")
        if i == 1:
            kwargs['plot_synthesized_image'] = True
        elif i == 2:
            kwargs['plot_rep_comparison'] = True
        elif i == 4:
            kwargs['plot_signal_comparison'] = True
            kwargs['iteration'] = None
        else:
            # otherwise, don't specify iteration
            kwargs.pop('iteration', None)
        np.random.seed(0)
        fig, axes_idx = synthesis_schematic(metamer, **kwargs)
        # remove ticks because they don't matter here
        if i >= 2:
            fig.axes[axes_idx['rep_comp']].set(xticks=[], yticks=[])
        if i >= 4:
            fig.axes[axes_idx['signal_comp']].set(xticks=[], yticks=[])
        if f == 'mp4':
            anim = metamer.animate(fig=fig, axes_idx=axes_idx,
                                   plot_loss=False, init_figure=False,
                                   plot_representation_error=False, **kwargs)
            anim.save(path)
        else:
            fig.savefig(path)


def pooling_window_area(windows, windows_scale=0, units='degrees'):
    """Plot window area as function of eccentricity.

    Plots the area of the window bands as function of eccentricity, with a
    horizontal line corresponding to a single pixel.

    Parameters
    ----------
    windows : pooling.PoolingWindows
        The PoolingWindows object to plot.
    windows_scale : int, optional
        The scale of the windows to plot. If units=='degrees', only the
        one-pixel line will change for different scales (in pixels, areas will
        drop by factor of 4).
    units: {'degrees', 'pixels'}, optional
        Which unit to plot eccentricity and area in.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.

    """
    fig = windows.plot_window_areas(units, scale_num=windows_scale,
                                    figsize=(15, 5))
    if units == 'degrees':
        # half is the smallest windows (for our models, which use gaussian
        # windows), full the largest.
        ylim = (windows.window_approx_area_degrees['half'].min(),
                windows.window_approx_area_degrees['full'].max())
        one_pixel_line = 1 / windows.deg_to_pix[windows_scale]
    elif units == 'pixels':
        # half is the smallest windows (for our models, which use gaussian
        # windows), full the largest.
        ylim = (windows.window_approx_area_pixels['half'].min(),
                windows.window_approx_area_pixels['full'].max())
        one_pixel_line = 1
    ylim = plotting.get_log_ax_lims(np.array(ylim), base=10)
    xlim = fig.axes[0].get_xlim()
    fig.axes[0].hlines(one_pixel_line, *xlim, colors='r', linestyles='--',
                       label='one pixel')
    fig.axes[0].set(yscale='log', xscale='log', ylim=ylim, xlim=xlim,
                    title=("Window area as function of eccentricity.\n"
                           "Half: at half-max amplitude, Full: $\pm$ 3 std dev, Top: 0 for Gaussians\n"
                           "Area is radial width * angular width * $\pi$/4\n"
                           "(radial width is double angular at half-max, "
                           "more than that at full, but ratio approaches "
                           "two as scaling shrinks / windows get smaller)"))
    fig.axes[0].legend()
    return fig


def synthesis_pixel_diff(stim, stim_df, scaling):
    """Show average pixel-wise squared error for a given scaling value.

    WARNING: This is reasonably memory-intensive.

    stim has dtype np.uint8. We convert back to np.float32 (and rescale to [0,
    1] interval) for these calculations.

    Parameters
    ----------
    stim : np.ndarray
        The array of metamers we want to check, should correspond to stim_df
    stim_df : pd.DataFrame
        The metamer information dataframe, as created by
        stimuli.create_metamer_df
    scaling : float
        The scaling value to check

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    errors : np.ndarray
        array of shape (stim_df.image_name.nunique(), *stim.shape[-2:])
        containing the squared pixel-wise errors

    """
    stim_df = stim_df.query('scaling in [@scaling, None]')
    num_seeds = stim_df.groupby('image_name').seed.nunique().mean()
    if int(num_seeds) != num_seeds:
        raise Exception("not all images have same number of seeds!")
    errors = np.zeros((int(num_seeds), stim_df.image_name.nunique(),
                       *stim.shape[-2:]), dtype=np.float32)
    errors *= np.nan
    for i, (im, g) in enumerate(stim_df.groupby('image_name')):
        target_img = stim[g.query('scaling in [None]').index[0]]
        # convert to float in a piecemeal fashion (rather than all at once in
        # the beginning) to reduce memory load. Can't select the subset of
        # images we want either, because then indices no longer line up
        target_img = utils.convert_im_to_float(target_img)
        for j, (seed, h) in enumerate(g.groupby('seed')):
            if len(h.index) > 1:
                raise Exception(f"Got more than 1 image with seed {seed} and "
                                f"image name {im}")
            synth_img = utils.convert_im_to_float(stim[h.index[0]])
            errors[j, i] = np.square(synth_img - target_img)
    errors = np.nanmean(errors, 0)
    titles = [t.replace('_range-.05,.95_size-2048,2600', '')
              for t in stim_df.image_name.unique()]
    fig = pt.imshow([e for e in errors], zoom=.5, col_wrap=5,
                    title=sorted(titles))
    fig.suptitle(f'Pixelwise squared errors for scaling {scaling}, averaged across seeds\n',
                 va='bottom', fontsize=fig.axes[0].title.get_fontsize()*1.25)
    return fig, errors


def simulate_num_trials(params, row='critical_scaling_true', col='variable'):
    """Create figure summarizing num_trials simulations.

    Assumes only one true value of proportionality_factor (will still work if
    not true, just might not be as good-looking).

    Parameters
    ----------
    params : pd.DataFrame
        DataFrame containing results from several num_trials simulations

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid with the plot.

    """
    tmp = params.melt(value_vars=['critical_scaling_true', 'proportionality_factor_true'],
                      value_name='true_value')
    tmp['variable'] = tmp.variable.apply(lambda x: x.replace('_true', ''), )

    params = params.melt(['bootstrap_num', 'max_iter', 'lr', 'scheduler',
                          'num_trials', 'num_bootstraps', 'proportionality_factor_true',
                          'critical_scaling_true'],
                         value_vars=['critical_scaling', 'proportionality_factor'])
    params = params.merge(tmp, left_index=True, right_index=True, suffixes=(None, '_y'))
    params = params.drop('variable_y', 1)

    g = sns.FacetGrid(params, row=row, col=col, aspect=1.5, sharey=False)
    g.map_dataframe(plotting.scatter_ci_dist, 'num_trials', 'value')
    g.map(plt.plot, 'num_trials', 'true_value', color='k', linestyle='--')
    return g


def performance_plot(expt_df, col='image_name', row=None, hue=None, style=None,
                     col_wrap=5, ci=95, curve_fit=False, logscale_xaxis=False,
                     **kwargs):
    """Plot performance as function of scaling.

    With default arguments, this is meant to show the results for all sessions
    and a single subject, showing the different images on each column. It
    should be flexible enough to handle other variants.

    Parameters
    ----------
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    col, row, hue, style : str or None, optional
        The variables in expt_df to facet along the columns, rows, hues, and
        styles, respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception.
        Ignored if col=None.
    ci : int, optional
        What confidence interval to draw on the performance.
    curve_fit : {True, False, 'to_chance'}, optional
        If True, we'll fit the psychophysical curve (as given in
        mcmc.proportion_correct_curve) to the mean of each faceted subset of
        data and plot that. If False, we'll instead join the points. If
        'to_chance', then we plot the psychophysical curve (as in True), and
        extend the x-values until performance hits chance.
    logscale_xaxis : bool, optional
        If True, we logscale the x-axis. Else, it's a linear scale.
    kwargs :
        passed to plotting.lineplot_like_pointplot

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    # set defaults based on hue and style args
    if curve_fit:
        kwargs['linestyle'] = ''
        kwargs.setdefault('dashes', False)
    dashes_dict = {}
    marker_adjust = {}
    if style is not None:
        style_dict = plotting.get_style(style, expt_df[style].unique())
        dashes_dict = style_dict.pop('dashes_dict', {})
        marker_adjust = style_dict.pop('marker_adjust', {})
        kwargs.setdefault('dashes', dashes_dict)
        kwargs.update(style_dict)
    # seaborn raises an error if col_wrap is non-None when col is None or row
    # is not None, so prevent that possibility
    if col is None or row is not None:
        col_wrap = None
    # remap the image names to be better for plotting
    expt_df = plotting._remap_image_names(expt_df)
    if hue is not None:
        kwargs.setdefault('palette', plotting.get_palette(hue, expt_df[hue].unique()))
    else:
        kwargs.setdefault('color', 'k')
    if col == 'image_name':
        img_order = plotting.get_order('image_name')
        kwargs.setdefault('col_order', img_order)
    g = plotting.lineplot_like_pointplot(expt_df, 'scaling',
                                         'hit_or_miss_numeric', ci=ci, col=col,
                                         row=row, hue=hue, col_wrap=col_wrap,
                                         style=style, legend=False, **kwargs)
    if marker_adjust:
        labels = {v: k for k, v in kwargs.get('markers', {}).items()}
        final_markers = plotting._marker_adjust(g.axes.flatten(),
                                                marker_adjust, labels)
    else:
        final_markers = {}

    if curve_fit:
        to_chance = True if curve_fit == 'to_chance' else False
        g.map_dataframe(plotting.fit_psychophysical_curve, 'scaling',
                        'hit_or_miss_numeric', pal=kwargs.get('palette', {}),
                        color=kwargs.get('color', 'k'), dashes_dict=dashes_dict,
                        to_chance=to_chance)
    g.map_dataframe(plotting.map_flat_line, x='scaling', y=.5, colors='k')

    # add some nice labels and titles
    plotting._label_and_title_psychophysical_curve_plot(g, expt_df,
                                                        'Performance', ci=ci)
    # get decent looking tick marks
    plotting._psychophysical_curve_ticks(expt_df, g.axes.flatten(),
                                         logscale_xaxis,
                                         kwargs.get('height', 5),
                                         col)
    # create the legend
    plotting._add_legend(expt_df, g, None, hue, style,
                         kwargs.get('palette', {}), final_markers,
                         dashes_dict)
    return g


def run_length_plot(expt_df, col=None, row=None, hue=None, col_wrap=None,
                    comparison='ref'):
    """Plot run length.

    With default arguments, this is meant to show the results for all sessions
    and a single subject. It should be flexible enough to handle other
    variants.

    Parameters
    ----------
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    col, row, hue : str or None, optional
        The variables in expt_df to facet along the columns, rows, and hues,
        respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception
    comparison : {'ref', 'met'}, optional
        Whether this comparison is between metamers and reference images
        ('ref') or two metamers ('met').

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    expt_df['approximate_run_length_min'] = expt_df.approximate_run_length / 60
    g = sns.catplot(x='session_number', y='approximate_run_length_min',
                    kind='strip', col=col, row=row, hue=hue, col_wrap=col_wrap,
                    data=expt_df.drop_duplicates(['subject_name', 'session_number',
                                                  'run_number']))
    g.set_ylabels("Approximate run length (in minutes)")
    g = plotting.title_experiment_summary_plots(g, expt_df, 'Run length')
    return g


def compare_loss_and_performance_plot(expt_df, stim_df, x='loss', col='scaling',
                                      row=None, hue='image_name', col_wrap=4,
                                      plot_kind='scatter', height=3,
                                      logscale_xaxis=True):
    """Compare an image metric with behavioral performance.

    By default, this plots synthesis loss on the x-axis and proportion correct
    on the y-axis, to see if there's any relationship there. Hopefully, there's
    not (that is, synthesis progressed to the point where there's no real
    difference in image from more iterations). By changing `x` to
    `'image_mse'`, we compare the behavioral performance against the MSE
    between the reference and synthesized images.

    Currently, only works for comparison='ref' (comparison between reference
    and natural images), because we plot each seed separately and
    comparison='met' shows multiple seeds per trial.

    Parameters
    ----------
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    stim_df : pd.DataFrame
        The metamer information dataframe, as created by
        `stimuli.create_metamer_df`
    x : str, optional
        Variable to plot on the x-axis. Must be a column in either expt_df or
        stim_df.
    col, row, hue : str or None, optional
        The variables in expt_df to facet along the columns, rows, and hues,
        respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception
    plot_kind : {'scatter', 'line'}, optional
        Whether to plot this as a scatter or line plot.
    height : float, optional
        Height of the axes.
    logscale_xaxis : bool, optional
        If True, we logscale the x-axis. Else, it's a linear scale.

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    if expt_df.unique_seed.hasnans:
        raise Exception("There's a NaN in expt_df.unique_seed! This means that "
                        "this expt_df comes from a metamer_vs_metamer run. That "
                        "means there are multiple synthesized images per trial "
                        "and so this plot comparing performance and loss for a"
                        " single synthesized image doesn't make sense!")
    # need to get proportion_correct, not the raw responses, for this plot.
    # adding session_number here doesn't change results except to make sure
    # that the session_number column is preserved in the output (each session
    # contains all trials, all scaling for a given image)
    expt_df = analysis.summarize_expt(expt_df, ['subject_name', 'session_number', 'scaling',
                                                'trial_type', 'unique_seed'])
    if x not in expt_df.columns:
        expt_df = expt_df.set_index(['subject_name', 'session_number', 'image_name',
                                     'scaling', 'unique_seed'])
        stim_df = stim_df.rename(columns={'seed': 'unique_seed'})
        stim_df = stim_df.set_index(['image_name', 'scaling',
                                     'unique_seed'])[x].dropna()
        expt_df = expt_df.merge(stim_df, left_index=True,
                                right_index=True,).reset_index()
    col_order, hue_order, row_order = None, None, None
    expt_df = plotting._remap_image_names(expt_df)
    if col is not None:
        col_order = plotting.get_order(col, expt_df[col].unique())
    else:
        col_wrap = None
    if row is not None:
        row_order = plotting.get_order(row, expt_df[row].unique())
    if hue is not None:
        hue_order = plotting.get_order(hue, expt_df[hue].unique())
    g = sns.relplot(data=expt_df, x=x, y='proportion_correct',
                    hue=hue, col=col, kind=plot_kind, col_wrap=col_wrap,
                    height=height, row=row, col_order=col_order,
                    hue_order=hue_order, row_order=row_order)
    if logscale_xaxis:
        g.set(xscale='log', xlim=plotting.get_log_ax_lims(expt_df[x]))
    g.set(ylabel='Proportion correct')
    if x == 'loss':
        title = ('Performance vs. synthesis loss', '\nHopefully no relationship here')
    else:
        title = (f'Performance vs. {x}', '')
    g = plotting.title_experiment_summary_plots(g, expt_df, *title)
    return g


def posterior_predictive_check(inf_data, col=None, row=None, hue=None,
                               style=None, col_wrap=5, comparison='ref',
                               logscale_xaxis=False, hdi=.95, query_str=None,
                               **kwargs):
    """Plot posterior predictive check.

    In order to make sure that our MCMC gave us a reasonable fit, we plot the
    posterior predictive probability correct curve against the observed
    responses.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`.
    col, row, hue, style : str or None, optional
        The dimensions in inf_data to facet along the columns, rows, hues, and
        styles, respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. Ignored if col=None.
    comparison : {'ref', 'met', 'both'}, optional
        Whether inf_data contains data on comparisons is between metamers and
        reference images ('ref'), two metamers ('met'), or both ('both').
    logscale_xaxis : bool, optional
        If True, we logscale the x-axis. Else, it's a linear scale.
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'").
    kwargs :
        passed to sns.FacetGrid

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    if not isinstance(inf_data, pd.DataFrame):
        df = mcmc.inf_data_to_df(inf_data, 'predictive', query_str, hdi=hdi)
    else:
        df = inf_data
    df = df.query('distribution!="prior_predictive"')
    # remove the responses from posterior predictive, we just want the
    # probability correct curve
    df = df.set_index('distribution')
    df.loc['posterior_predictive', 'responses'] = np.nan
    df = df.reset_index()
    # set defaults based on hue and style args
    if hue is not None:
        kwargs.setdefault('palette', plotting.get_palette(hue, df[hue].unique()))
        color = None
    else:
        color = kwargs.pop('color', 'k')
    if style is not None:
        style_dict = plotting.get_style(style, df[style].unique())
        dashes_dict = style_dict.pop('dashes_dict', {})
        marker_adjust = style_dict.pop('marker_adjust', {})
        markers = style_dict.pop('markers', {})
        kwargs.update(style_dict)
    else:
        dashes_dict = {}
        marker_adjust = {}
        markers = {}
    # seaborn raises an error if col_wrap is non-None when col is None or if
    # row is not None, so prevent that possibility
    if col is None or row is not None:
        col_wrap = None
    # remap the image names to be better for plotting
    df = plotting._remap_image_names(df)
    img_order = plotting.get_order('image_name')
    if col == 'image_name':
        kwargs.setdefault('col_order', img_order)
    if row == 'image_name':
        kwargs.setdefault('row_order', img_order)
    g = sns.FacetGrid(df, row=row, col=col, hue=hue, col_wrap=col_wrap,
                      **kwargs)
    g.map_dataframe(plotting.lineplot_like_pointplot, x='scaling',
                    y='responses', ci=None, style=style, legend=False,
                    linestyle='', dashes=False, ax='map',
                    markers=markers, color=color)
    if marker_adjust:
        labels = {v: k for k, v in markers.items()}
        final_markers = plotting._marker_adjust(g.axes.flatten(),
                                                marker_adjust, labels)
    else:
        final_markers = {}

    g.map_dataframe(plotting.scatter_ci_dist, x='scaling',
                    y='probability_correct', like_pointplot=True, ci='hdi',
                    join=True, ci_mode='fill', draw_ctr_pts=False, style=style,
                    dashes_dict=dashes_dict)
    if col is None and row is None:
        assert len(g.axes)==1, "If col is None and row is None, there should only be one axis!"
        plotting.map_flat_line(x='scaling', y=.5, colors='k', ax=g.ax,
                               data=df, color=None)
    else:
        g.map_dataframe(plotting.map_flat_line, x='scaling', y=.5, colors='k')

    # title and label plot
    model_type = df.mcmc_model_type.dropna().unique()
    if len(model_type) > 1:
        model_type = ['multiple']
    title = f'Posterior predictive check for {model_type[0]}'
    plotting._label_and_title_psychophysical_curve_plot(g, df, title,
                                                        hdi=hdi)
    title_ = []
    if row is not None:
        title_.append("{row_name}")
    if col is not None:
        title_.append("{col_name}")
    g.set_titles('|'.join(title_))
    # get decent looking tick marks
    plotting._psychophysical_curve_ticks(df, g.axes.flatten(),
                                         logscale_xaxis,
                                         kwargs.get('height', 5),
                                         col)
    # create the legend
    plotting._add_legend(df, g, None, hue, style,
                         kwargs.get('palette', {}),
                         final_markers, dashes_dict)
    return g


def mcmc_diagnostics_plot(inf_data):
    """Plot MCMC diagnostics.

    This plot contains the posterior distributions and sampling trace for all
    parameters (each chain showne), with r-hat and effective sample size (both
    diagnostic stats) on the plots.

    r-hat: ratio of average variance of samples within each chain to the
    variance of pooled samples across chains. If all chains have converged,
    this should be 1.

    effective sample size (ESS): computed, from autocorrelation, measures
    effective number of samples. different draws in a chain should be
    independent samples from the posterior, so they shouldn't be
    autocorrelated. therefore, this number should be large. if it's small,
    probably need more warmup steps and draws.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`.

    Returns
    -------
    fig : plt.Figure
        matplotlib figure containing the plots.

    """
    axes = az.plot_trace(inf_data)
    rhat = az.rhat(inf_data.posterior)
    ess = az.ess(inf_data.posterior)
    for ax in axes:
        var = ax[0].get_title()
        ax[0].set_title(ax[0].get_title()+
                        f', nanmean r_hat={np.nanmean(rhat[var].data):.05f}')
        ax[1].set_title(ax[1].get_title()+
                        f', nanmean effective sample size={np.nanmean(ess[var].data):.02f}')
    fig = axes[0, 0].figure
    # want monospace so table prints correctly
    rhat = rhat.to_dataframe().reorder_levels(['model', 'trial_type', 'image_name', 'subject_name'])
    fig.text(1, 1, "rhat\n"+rhat.sort_index().to_markdown(),
             ha='left', va='top', family='monospace')
    ess = ess.to_dataframe().reorder_levels(['model', 'trial_type', 'image_name', 'subject_name'])
    fig.text(1, .5, "effective sample size\n"+ess.sort_index().to_markdown(),
             ha='left', va='top', family='monospace')
    model_type = inf_data.metadata.mcmc_model_type.values
    if len(model_type) > 1:
        model_type = ['multiple']
    fig.suptitle(f"Diagnostics plot for {model_type[0, 0]} MCMC, showing distribution and sampling"
                 " trace for each parameter", va='baseline')
    return fig


def parameter_pairplot(inf_data, vars=None,
                       query_str="distribution=='posterior'", **kwargs):
    """Joint distributions of posterior parameter values.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`.
    vars : list or None, optional
        List of strs giving the parameters to plot here. If None, will plot all.
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'"). Should almost certainly
        include that distribution selection to your query_str for this plot.

    kwargs :
        passed to sns.pairplot

    Returns
    -------
    g : sns.PairGrid
        sns PairGrid containing the plots.

    """
    df = mcmc.inf_data_to_df(inf_data, 'parameters', query_str=query_str)
    pivot_idx = [c for c in df.columns if c not in ['value', 'variable']]
    df = df.pivot_table('value', pivot_idx, 'variable')
    def key_func(x):
        # want these to be first
        if 'log' in x:
            return '__' + x
        # then these
        elif 'global' in x:
            return '_' + x
        # and this last
        elif x == 'pi_l':
            return 'z' + x
        else:
            return x
    if vars is None:
        vars = sorted(df.columns, key=key_func)
    g = sns.pairplot(df.reset_index(), vars=vars, corner=True, diag_kind='kde',
                     kind='kde', diag_kws={'cut': 0}, **kwargs)
    model_type = df.mcmc_model_type.unique()
    if len(model_type) > 1:
        model_type = ['multiple']
    g.fig.suptitle(f'Joint distributions of {model_type[0]} model parameters')
    return g


def psychophysical_curve_parameters(inf_data, x='image_name', y='value',
                                    hue='subject_name', col='parameter',
                                    row='trial_type', style=None,
                                    query_str="distribution=='posterior'",
                                    height=2.5, x_dodge=.15, hdi=.95,
                                    rotate_xticklabels=False,
                                    title_str="{row_val} | {col} = {col_val}",
                                    **kwargs):
    """Show psychophysical curve parameters, with HDI error bars.

    This plots the psychophysical curve parameters for all full curves we can
    draw. That is, we combine the effects of our model and show the values for
    each trial type, image, and subject.

    Parameters
    ----------
    inf_data : arviz.InferenceData or df.
        arviz InferenceData object (xarray-like) created by `run_inference`. If
        df, we assume it's already been turned into a dataframe by
        `mcmc.inf_data_to_df` and use as is.
    x, y, hue, col, row, style : str, optional
        variables to plot on axes or facet along. 'value' is the value of the
        parameters, 'parameter' is the identity of the parameter (e.g., 's0',
        'a0'), all other are the coords of inf_data. Note that col and row
        cannot be None. style can be a list with multiple values.
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'").
        posterior, so we clip to get a reasonable view
    height : float, optional
        Height of the facets
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same x-values (or
        are categorical), we can dodge the data along the x-axis,
        deterministically shifting it. If a float, x_dodge is the amount we
        shift each level of hue by; if None, we don't dodge at all; if True, we
        dodge as if x_dodge=.01
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate.
    title_str : str, optional
        Format string for axes titles. Can include {row_val}, {col_val}, {row},
        {col} (for the values and names of those facets, respectively) and
        plain text.
    kwargs :
        passed to plt.subplots

    Returns
    -------
    fig : plt.Figure
        figure containing the figure.

    """
    kwargs.setdefault('sharey', False)
    kwargs.setdefault('sharex', True)
    if not isinstance(inf_data, pd.DataFrame):
        df = mcmc.inf_data_to_df(inf_data, 'psychophysical curve parameters',
                                 query_str=query_str, hdi=hdi)
    else:
        df = inf_data
    # set defaults based on hue and style args
    if hue is not None:
        palette = kwargs.pop('palette', plotting.get_palette(hue,
                                                             df[hue].unique()))
    else:
        palette = {None: kwargs.pop('color', 'C0')}
    if style is not None:
        try:
            col_unique = df[style].unique()
            style = style
        except AttributeError:
            # then there are multiple values in style
            col_unique = [df[s].unique().tolist() for s in style]
        style_dict = plotting.get_style(style, col_unique)
        marker_adjust = style_dict.pop('marker_adjust', {})
    else:
        marker_adjust = {}
    # remap the image names to be better for plotting
    if 'image_name' in df.columns:
        df = plotting._remap_image_names(df)
    x_order = kwargs.pop('x_order', None)
    if x == 'image_name' and x_order is None:
        x_order = plotting.get_order(x)
    fig, axes, cols, rows = plotting._setup_facet_figure(df, col, row,
                                                         height=height,
                                                         rotate_xticklabels=rotate_xticklabels,
                                                         gridspec_kw={'wspace': .05},
                                                         **kwargs)
    final_markers = {}
    label, all_labels = plotting._prep_labels(df, hue, style, col, row)
    for i, c in enumerate(cols):
        for j, r in enumerate(rows):
            ax = axes[j, i]
            d = df.query(f"{col}=='{c}' & {row}=='{r}'")
            xlabel, ylabel = '', ''
            if i == 0:
                ylabel = f'{y} with {int(hdi*100)}% HDI'
            if j == len(rows)-1:
                xlabel = x
            title_ = title_str.format(row_val=r, col_val=c, col=col, row=row)
            markers_tmp = plotting._facetted_scatter_ci_dist(d, x, y, hue,
                                                             style, x_order,
                                                             label, all_labels,
                                                             x_dodge,
                                                             marker_adjust,
                                                             palette,
                                                             rotate_xticklabels,
                                                             xlabel, ylabel,
                                                             title_, ax=ax)
            final_markers.update(markers_tmp)

    # create the legend
    plotting._add_legend(df, None, fig, hue, style, palette,
                         final_markers, {k: '' for k in marker_adjust.keys()})
    model_type = df.mcmc_model_type.unique()
    if len(model_type) > 1:
        model_type = ['multiple']
    fig.suptitle(f"Psychophysical curve parameter values for {model_type[0]} MCMC", va='bottom')
    return fig


def ref_image_summary(stim, stim_df, zoom=.125):
    """Display grid of reference images used for metamer synthesis.

    We gamma-correct the reference images before display and title each with
    the simple name (e.g., "llama", "troop")

    Parameters
    ----------
    stim : np.ndarray
        The array of metamers we want to check, should correspond to stim_df
    stim_df : pd.DataFrame
        The metamer information dataframe, as created by
        stimuli.create_metamer_df
    zoom : float or int, optional
        How to zoom the images. Must result in an integer number of pixels

    Returns
    -------
    fig : plt.Figure
        Figure containing the images.

    """
    ref_ims = stim_df.fillna('None').query("model=='None'").image_name
    ref_ims = ref_ims.apply(lambda x: x.replace('symmetric_', '').replace('_range-.05,.95_size-2048,2600', ''))
    img_order = plotting.get_order('image_name')
    ref_ims = ref_ims.sort_values(key=lambda x: [img_order.index(i) for i in x])
    refs = stim[ref_ims.index]

    ax_size = np.array([2048, 2600]) * zoom
    fig = pt.tools.display.make_figure(4, 5, ax_size, vert_pct=.9)
    for ax, im, t in zip(fig.axes, refs, ref_ims.values):
        # gamma-correct the image
        ax.imshow((im/255)**(1/2.2), vmin=0, vmax=1, cmap='gray')
        ax.set_title(t)
    return fig


def synthesis_distance_plot(distances, xy='trial_type', hue='ref_image',
                            col='distance_scaling',
                            row='image_1_init_supertype', x=None, y=None,
                            **kwargs):
    """Plot distances between metamers and their reference images.

    With default arguments, scatterplot comparing the metamer_vs_metamer
    distance to the metamer_vs_reference distance (where distance and synthesis
    were computed using the same model). We can then see that metamers
    initialized with uniform noise all end up near each other in synthesis model
    space (and thus, probably, human perceptual space), clumped up together so
    that their distance to each other is smaller than the distance to their
    reference image. We hypothesize that initializing with a natural image helps
    minimize this problem.

    We average across all distances for a given scaling and reference image,
    since it's hard to think about how we could align them (e.g., if we have
    three metamers, we'll have three metamer_vs_metamer distances and three
    metamer_vs_reference distances, but each metamer_vs_reference distance
    corresponds to a single metamer, while each metamer_vs_metamer distance
    corresponds to *two* metamers. this problem gets worse as the number of
    metamers increases).

    Parameters
    ----------
    distances : pd.DataFrame
        distance dataframe, containing multiple outputs of
        foveated_metamers.distances.model_distance
    xy : str, optional
        str corresponding to one the columns in distances. Must have only two
        unique values. We will then pivot distances in such a way as to plot
        the two values against each other, one as x, one as y. If you want to
        specify which value goes on which axis, set x and y.
    hue, col, row : str, optional
        strs corresponding to columns in distances, to map along the various
        dimensions.
    x, y : str or None, optional
        If strs, values of the ``distances[xy]`` column, specifying which
        should go on x, which on y. If None,
        ``sorted(distances[xy].unique)[0]`` goes on x, the other goes on y
    kwargs :
        passed to sns.relplot

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot

    """
    idx = np.logical_or(distances.image_2_init_supertype == 'reference',
                        distances.image_2_init_supertype == distances.image_1_init_supertype)
    distances = distances[idx]
    idx = np.logical_and(distances.synthesis_model == distances.distance_model,
                         distances.synthesis_scaling == distances.distance_scaling)
    distances = distances[idx]
    if distances[xy].nunique() != 2:
        raise Exception(f"Column {xy} must have two values, but found {distances[xy].unique()} instead!")
    vals = sorted(distances[xy].unique())
    if x is None and y is None:
        x = vals[0]
        y = vals[1]
    elif x is not None:
        vals.remove(x)
        y = vals[0]
    elif y is not None:
        vals.remove(y)
        x = vals[0]
    assert x in distances[xy].unique(), f"x must be a value of distances column {xy}, but got {x}!"
    assert y in distances[xy].unique(), f"y must be a value of distances column {xy}, but got {y}!"

    cols = list(itertools.product(distances[xy].unique(),
                                  distances[row].unique()))
    distances = pd.pivot_table(distances, 'distance',
                               ['distance_model', 'ref_image', col],
                               [xy, row])
    distances = pd.melt(distances.reset_index(), [('distance_model', ''), ('ref_image', ''), (col, '')], cols)
    distances = distances.rename(columns={c: c[0] for c in distances.columns if isinstance(c, tuple)})
    distances = distances.pivot(['distance_model', 'ref_image', row, col],
                                xy, 'value').reset_index()
    if hue == 'ref_image':
        kwargs.setdefault('hue_order', plotting.get_order('image_name'))
    g = sns.relplot(x=x, y=y, data=distances, hue=hue, kind='scatter', col=col, row=row, **kwargs)
    for ax in g.axes.flatten():
        ax.set(xscale='log', yscale='log')
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    return g


def partially_pooled_metaparameters(inf_data, hue='model', style='trial_type',
                                    distribution='posterior', hdi=.95,
                                    height=5, aspect=1,
                                    rotate_xticklabels=False,
                                    x_dodge=False, **kwargs):
    """Plot the metaparameters of the partially pooled mcmc model.

    The metaparametesr are the lapse rate and those that control the
    distribution across images/subjects of a0 and s0.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`
    hue, style : str, optional
        variables to facet along.
    distribution : str, optional
        what distribution to grab from inf_data
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.
    height : float, optional
        Height of the axes
    aspect : float, optional
        Aspect of the axes
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate.
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same x-values (or
        are categorical), we can dodge the data along the x-axis,
        deterministically shifting it. If a float, x_dodge is the amount we
        shift each level of hue by; if None, we don't dodge at all; if True, we
        dodge as if x_dodge=.01
    kwargs :
        Passed to plt.subplots

    Returns
    -------
    fig : plt.Figure
        figure containing the figure.

    """
    if inf_data.metadata.mcmc_model_type[0, 0] != 'partially-pooled':
        raise Exception("Can only create this plot with partially-pooled mcmc"
                        " model but got "
                        f"{inf_data.metadata.mcmc_model_type.values[0, 0]}!")
    inf_data = mcmc._compute_hdi(inf_data[distribution], hdi)
    keys_to_exclude = [f'log_{p}_{t}' for p, t in
                       itertools.product(['a0', 's0'], ['image', 'subject'])]
    cols = [k for k in inf_data.data_vars.keys() if k
            not in keys_to_exclude + ['pi_l']]
    inf1 = inf_data[cols].to_dataframe()
    inf1 = inf1.reset_index().melt(inf1.index.names)
    inf2 = inf_data[['pi_l']].to_dataframe()
    inf2 = inf2.reset_index().melt(inf2.index.names)
    inf1['var_type'] = inf1.variable.map(lambda x: x.replace('image_', '').replace('subject_', ''))
    inf1['variable'] = inf1.variable.map(lambda x: x.replace('a0_', '').replace('s0_', ''))
    inf2['var_type'] = 'Lapse rate'
    inf2['variable'] = inf2.subject_name
    inf2 = inf2.drop(columns=['subject_name'])

    metaparams = pd.concat([inf1, inf2])
    metaparam_order = ['log_global_mean', 'image_sd', 'subject_sd']
    # set defaults based on hue and style args
    if hue is not None:
        palette = kwargs.pop('palette', plotting.get_palette(hue,
                                                             metaparams[hue].unique()))
    else:
        palette = {None: kwargs.pop('color', 'C0')}
    if style is not None:
        try:
            col_unique = metaparams[style].unique()
            style = style
        except AttributeError:
            # then there are multiple values in style
            col_unique = [metaparams[s].unique().tolist() for s in style]
        style_dict = plotting.get_style(style, col_unique)
        marker_adjust = style_dict.pop('marker_adjust', {})
    else:
        marker_adjust = {}
    figsize = (aspect*2*height, height)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    axes_dict = fig.subplot_mosaic([['log_a0_global_mean', 'a0_sd', 'Lapse rate'],
                                    ['log_s0_global_mean', 's0_sd', 'Lapse rate']],
                                   gridspec_kw={'width_ratios': [.75, 1.25, 2]})
    final_markers = {}
    label, all_labels = plotting._prep_labels(metaparams, hue, style,
                                              'var_type', None)
    for k, ax in axes_dict.items():
        if 'a0' in k:
            ax.sharex(axes_dict[k.replace('a0', 's0')])
        if k == 'a0_sd':
            ax.sharey(axes_dict['s0_sd'])
        data = metaparams.query(f"var_type=='{k}'")
        title_, xlabel, ylabel = '', '', ''
        if 'a0' not in k and 's0' not in k:
            title_ = k
        if k == 'Lapse rate':
            xlabel = 'subject_name'
        markers_tmp = plotting._facetted_scatter_ci_dist(data, 'variable',
                                                         'value', hue, style,
                                                         None, label,
                                                         all_labels, x_dodge,
                                                         marker_adjust,
                                                         palette,
                                                         rotate_xticklabels,
                                                         xlabel, ylabel,
                                                         title_, ax=ax)
        final_markers.update(markers_tmp)
        if k == 'log_s0_global_mean':
            ax.set_ylabel(f'value with {int(hdi*100)}% HDI', y=1.2)
        if 'sd' in k:
            ax.set_title(k.replace('_sd', ''), x=0, va='bottom')
            xlim = ax.get_xlim()
            ax.set_xlim((xlim[0]-.25, xlim[1]+.25))
        if 'a0' in k:
            ax.tick_params('x', labelbottom=False)
        elif 'Lapse' in k:
            ax.set_ylim((0, ax.get_ylim()[1]))
    plotting._add_legend(metaparams, None, fig, hue, style, palette,
                         final_markers, {k: '' for k in marker_adjust.keys()})
    fig.suptitle("Parameter values 1 for partially-pooled MCMC\n", y=.95, va='bottom')
    return fig


def partially_pooled_parameters(inf_data, hue='model', style='trial_type',
                                distribution='posterior', hdi=.95, height=4,
                                aspect=2, rotate_xticklabels=True,
                                x_dodge=False, **kwargs):
    """Plot the subject/image level parameters of partially pooled mcmc model.

    These are log_a0/s0_image/subject.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`
    hue, style : str, optional
        variables to facet along.
    distribution : str, optional
        what distribution to grab from inf_data
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.
    height : float, optional
        Height of the axes
    aspect : float, optional
        Aspect of the axes
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate.
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same x-values (or
        are categorical), we can dodge the data along the x-axis,
        deterministically shifting it. If a float, x_dodge is the amount we
        shift each level of hue by; if None, we don't dodge at all; if True, we
        dodge as if x_dodge=.01
    kwargs :
        Passed to plt.subplots

    Returns
    -------
    fig : plt.Figure
        figure containing the figure.

    """
    if inf_data.metadata.mcmc_model_type[0, 0] != 'partially-pooled':
        raise Exception("Can only create this plot with partially-pooled mcmc"
                        " model but got "
                        f"{inf_data.metadata.mcmc_model_type.values[0, 0]}!")
    img_order = np.array(plotting.get_order('image_name'))
    inf_data = mcmc._compute_hdi(inf_data[distribution], hdi)
    keys_to_include = [f'log_{p}_{t}' for p, t in
                       itertools.product(['a0', 's0'], ['image', 'subject'])]
    inf1 = inf_data[[k for k in keys_to_include if 'image' in k]].to_dataframe()
    inf1 = inf1.reset_index().melt(inf1.index.names)
    inf2 = inf_data[[k for k in keys_to_include if 'subject' in k]].to_dataframe()
    inf2 = inf2.reset_index().melt(inf2.index.names)

    inf1['var_type'] = 'image-level'
    inf1['x_var'] = inf1.image_name
    inf1 = inf1.drop(columns=['image_name'])
    inf2['var_type'] = 'subject-level'
    inf2['x_var'] = inf2.subject_name
    inf2 = inf2.drop(columns=['subject_name'])

    params = pd.concat([inf1, inf2])
    params.variable = params.variable.map(lambda x: '_'.join(x.split('_')[:-1]))
    params.x_var = params.x_var.map(lambda x: x.split('_')[0])

    # set defaults based on hue and style args
    if hue is not None:
        palette = kwargs.pop('palette', plotting.get_palette(hue,
                                                             params[hue].unique()))
    else:
        palette = {None: kwargs.pop('color', 'C0')}
    if style is not None:
        try:
            col_unique = params[style].unique()
            style = style
        except AttributeError:
            # then there are multiple values in style
            col_unique = [params[s].unique().tolist() for s in style]
        style_dict = plotting.get_style(style, col_unique)
        marker_adjust = style_dict.pop('marker_adjust', {})
    else:
        marker_adjust = {}
    fig, axes, cols, rows = plotting._setup_facet_figure(params, 'var_type',
                                                         'variable',
                                                         height=height,
                                                         aspect=aspect,
                                                         rotate_xticklabels=rotate_xticklabels,
                                                         sharex='col',
                                                         sharey='row',
                                                         gridspec_kw={'wspace': .04,
                                                                      'hspace': .12,
                                                                      'width_ratios': [2, 1]},
                                                         **kwargs)
    final_markers = {}
    label, all_labels = plotting._prep_labels(params, hue, style, 'var_type', 'variable')
    for i, c in enumerate(cols):
        for j, r in enumerate(rows):
            if c == 'image-level':
                x_order = img_order
            else:
                x_order = None
            ax = axes[j, i]
            d = params.query(f"var_type=='{c}' & variable=='{r}'")
            xlabel, ylabel = '', ''
            if i == 0:
                ylabel = f'value with {int(hdi*100)}% HDI'
            if j == len(rows)-1:
                if i == 0:
                    xlabel = 'image_name'
                elif i == 1:
                    xlabel = 'subject_name'
            title_ = f"{r} | {c}"
            markers_tmp = plotting._facetted_scatter_ci_dist(d, 'x_var',
                                                             'value', hue,
                                                             style, x_order,
                                                             label, all_labels,
                                                             x_dodge,
                                                             marker_adjust,
                                                             palette,
                                                             rotate_xticklabels,
                                                             xlabel, ylabel,
                                                             title_, ax=ax)
            final_markers.update(markers_tmp)
            xlim = ax.get_xlim()
            ax.axhline(xmin=xlim[0], xmax=xlim[1], linestyle='--', c='k')
            ax.set_xlim(xlim)

    # create the legend
    plotting._add_legend(params, None, fig, hue, style, palette,
                         final_markers, {k: '' for k in marker_adjust.keys()})
    fig.suptitle("Parameter values 2 for partially-pooled MCMC", y=.95)
    return fig


def psychophysical_grouplevel_means(inf_data, x='dependent_var', y='value',
                                    hue='model', col='level', row='parameter',
                                    style='trial_type',
                                    query_str="distribution=='posterior'",
                                    height=4, aspect=2, x_dodge=.15, hdi=.95,
                                    rotate_xticklabels=True,
                                    title_str="{row_val} | {col_val}",
                                    **kwargs):
    """Show psychophysical group-level means, with HDI error bars.

    This plots the psychophysical group-level means for all full curves we can
    draw. That is, we compute the psychophysical curve parameters and then
    average over the images and subjects (separately), plotting their
    distributions.

    Parameters
    ----------
    inf_data : arviz.InferenceData or df.
        arviz InferenceData object (xarray-like) created by `run_inference`. If
        df, we assume it's already been turned into a dataframe by
        `mcmc.inf_data_to_df` and use as is.
    x, y, hue, col, row, style : str, optional
        variables to plot on axes or facet along. 'value' is the value of the
        parameters, 'parameter' is the identity of the parameter (e.g., 's0',
        'a0'), 'level' is subject / image, 'dependent_var' is the variables of
        those levels, all other are the coords of inf_data. Note that col and
        row cannot be None. style can be a list with multiple values.
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'").
        posterior, so we clip to get a reasonable view
    height : float, optional
        Height of the axes.
    aspect : float, optional
        Aspect of the axes.
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same x-values (or
        are categorical), we can dodge the data along the x-axis,
        deterministically shifting it. If a float, x_dodge is the amount we
        shift each level of hue by; if None, we don't dodge at all; if True, we
        dodge as if x_dodge=.01
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate.
    title_str : str, optional
        Format string for axes titles. Can include {row_val}, {col_val}, {row},
        {col} (for the values and names of those facets, respectively) and
        plain text.
    kwargs :
        passed to plt.subplots

    Returns
    -------
    fig : plt.Figure
        figure containing the figure.

    """
    kwargs.setdefault('sharey', 'row')
    kwargs.setdefault('sharex', 'col')
    if not isinstance(inf_data, pd.DataFrame):
        df = mcmc.inf_data_to_df(inf_data, 'parameter grouplevel means',
                                 query_str=query_str, hdi=hdi)
    else:
        df = inf_data
    # set defaults based on hue and style args
    if hue is not None:
        palette = kwargs.pop('palette', plotting.get_palette(hue,
                                                             df[hue].unique()))
    else:
        palette = {None: kwargs.pop('color', 'C0')}
    if style is not None:
        try:
            col_unique = df[style].unique()
            style = style
        except AttributeError:
            # then there are multiple values in style
            col_unique = [df[s].unique().tolist() for s in style]
        style_dict = plotting.get_style(style, col_unique)
        marker_adjust = style_dict.pop('marker_adjust', {})
    else:
        marker_adjust = {}
    img_order = plotting.get_order('image_name')
    x_order = kwargs.pop('x_order', None)
    overall_means = df.query("level=='all'")
    df = df.query("level!='all'")
    fig, axes, cols, rows = plotting._setup_facet_figure(df, col, row,
                                                         height=height, aspect=aspect,
                                                         rotate_xticklabels=rotate_xticklabels,
                                                         gridspec_kw={'wspace': .05, 'hspace': .12,
                                                                      'width_ratios': [2, 1]},
                                                         **kwargs)
    final_markers = {}
    label, all_labels = plotting._prep_labels(df, hue, style, col, row)
    for i, c in enumerate(cols):
        for j, r in enumerate(rows):
            ax = axes[j, i]
            if x == 'dependent_var' and x_order is None and 'image' in c:
                x_ord = img_order
            else:
                x_ord = x_order
            query_str = f"{col}=='{c}' & {row}=='{r}'"
            d = df.query(query_str)
            xlabel, ylabel = '', ''
            if i == 0:
                ylabel = f'{y} with {int(hdi*100)}% HDI'
            if j == len(rows)-1:
                xlabel = c
            title_ = title_str.format(row_val=r, col_val=c, col=col, row=row)
            # if level is row or col, we want to ignore it and take the other part
            if col == 'level':
                query_str = query_str.split('&')[1]
            elif row == 'level':
                query_str = query_str.split('&')[0]
            means = overall_means.query(query_str)
            markers_tmp = plotting._facetted_scatter_ci_dist(d, x, y, hue,
                                                             style, x_ord,
                                                             label, all_labels,
                                                             x_dodge,
                                                             marker_adjust,
                                                             palette,
                                                             rotate_xticklabels,
                                                             xlabel, ylabel,
                                                             title_, ax=ax)
            xlim = ax.get_xlim()
            if d[hue].nunique() > 1:
                color = 'k'
            else:
                color = palette[d[hue].unique()[0]]
            ax.fill_between(xlim, means[y].min(), means[y].max(), color=color,
                            alpha=.2)
            ax.axhline(means.query("hdi==50")[y].values,
                       linestyle='--', color=color)
            ax.set_xlim(xlim)
            final_markers.update(markers_tmp)

    # create the legend
    plotting._add_legend(df, None, fig, hue, style, palette,
                         final_markers, {k: '' for k in marker_adjust.keys()})
    model_type = df.mcmc_model_type.unique()
    if len(model_type) > 1:
        model_type = ['multiple']
    fig.suptitle(f"Psychophysical group-level means for {model_type[0]} MCMC", y=.95)
    return fig


def amplitude_spectra(spectra, hue='scaling', style=None, col='image_name',
                      row=None, col_wrap=5, kind='line', estimator=None,
                      height=2.5, aspect=1, **kwargs):
    """Compare amplitude spectra of natural and synthesized images.
    
    Parameters
    ----------
    spectra : xarray.Dataset
        Dataset containing the spectra for synthesized metamers and our natural
        reference images.
    hue, style, col, row : str or None, optional
        The dimensions in spectra to facet along the columns, rows, hues, and
        styles, respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. Ignored if col=None.
    kind : {'line', 'scatter'}, optional
        Type of plot to make.
    estimator : name of pandas method or callable or None
        Method for aggregating across multiple observations of the y variable
        at the same x level. If None, all observations will be drawn
        (recommended).
    height : float, optional
        Height of the axes.
    aspect : float, optional
        Aspect of the axes.
    kwargs :
        Passed to sns.relplot

    Returns
    -------
    g : sns.FacetGrid
        Facetgrid with the plot.

    """
    df = plotting._spectra_dataset_to_dataframe(spectra, 'sf')
    # seaborn raises an error if col_wrap is non-None when col is None or if
    # row is not None, so prevent that possibility
    if col is None or row is not None:
        col_wrap = None
    # remap the image names to be better for plotting
    df = plotting._remap_image_names(df)
    img_order = plotting.get_order('image_name')
    if col == 'image_name':
        kwargs.setdefault('col_order', img_order)
    if row == 'image_name':
        kwargs.setdefault('row_order', img_order)
    if hue is not None:
        kwargs.setdefault('palette', plotting.get_palette(hue, df[hue].unique()))
    else:
        kwargs.setdefault('color', 'k')
    marker_adjust = {}
    dashes_dict = {}
    if style is not None:
        style_dict = plotting.get_style(style, df[style].unique())
        dashes_dict = style_dict.pop('dashes_dict', {})
        marker_adjust = style_dict.pop('marker_adjust', {})
        kwargs.setdefault('dashes', dashes_dict)
        kwargs.update(style_dict)
    # need to fill in the NaNs in a particular way so our plot is created
    # correctly. freq_n and sf_amplitude are mapped to x and y, so we need to
    # have positive finite values for the plot ranges to make sense
    df.freq_n = df.freq_n.fillna(1)
    df.sf_amplitude = df.sf_amplitude.fillna(df.sf_amplitude.min())
    # not entirely sure why it's necessary for these to be filled in correctly,
    # but it is
    df.seed_n = df.seed_n.fillna(df.seed_n.dropna().unique()[0])
    if hue is not None:
        df[hue] = df[hue].fillna(df[hue].dropna().unique()[0])
    if style is not None:
        df[style] = df[style].fillna(df[style].dropna().unique()[0])
    g = sns.relplot(x='freq_n', y='sf_amplitude', hue=hue, units='seed_n',
                    col=col, row=row, col_wrap=col_wrap, estimator=estimator,
                    height=height, aspect=aspect, data=df, kind=kind,
                    legend=False, **kwargs)
    if marker_adjust:
        labels = {v: k for k, v in kwargs.get('markers', {}).items()}
        final_markers = plotting._marker_adjust(g.axes.flatten(),
                                                marker_adjust, labels)
    else:
        final_markers = {}
    g.set(xscale='log', yscale='log')
    g.set_ylabels('Amplitude')
    g.set_xlabels('Spatial frequency (cycles/image)')
    # create the legend
    plotting._add_legend(df, g, None, hue, style,
                         kwargs.get('palette', {}), final_markers,
                         dashes_dict)
    # we use spectra because it doesn't include np.nan from dummy rows
    title_str = (f"Amplitude spectra for {' and '.join(spectra.model.values)}"
                 f" metamers, {' and '.join(spectra.trial_type.values)}"
                 " comparisons\n")
    g.fig.suptitle(title_str, va='bottom')
    return g 


def amplitude_orientation(spectra, hue='scaling', style=None, col='image_name',
                          row=None, col_wrap=5, kind='point',
                          estimator=np.median, height=2.5, aspect=2,
                          demean=False, **kwargs):
    """Compare orientation distributions of natural and synthesized images.

    Note this is fairly memory intensive. Setting `n_boot` to a lower number
    (defaults to 1000) seems to help with this.

    Parameters
    ----------
    spectra : xarray.Dataset
        Dataset containing the spectra for synthesized metamers and our natural
        reference images.
    hue, style, col, row : str or None, optional
        The dimensions in spectra to facet along the columns, rows, hues, and
        styles, respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. Ignored if col=None.
    kind : {'line', 'scatter'}, optional
        Type of plot to make.
    estimator : name of pandas method or callable
        Method for aggregating across multiple observations of the y variable
        at the same x level. median is recommended because of the outliers in
        the dataset
    height : float, optional
        Height of the axes.
    aspect : float, optional
        Aspect of the axes.
    demean : bool, optional
        Whether to demean the amplitudes before plotting (mean is computed per
        model, image_name, scaling, seed_n).
    kwargs :
        Passed to sns.catplot

    Returns
    -------
    g : sns.FacetGrid
        Facetgrid with the plot.

    """
    if demean:
        # quicker (and easier) to do this demeaning in xarray, rather than
        # pandas.
        ori_mean = spectra.mean(['orientation_slice', 'samples'])
        spectra = spectra - ori_mean
    df = plotting._spectra_dataset_to_dataframe(spectra, 'orientation')
    # seaborn raises an error if col_wrap is non-None when col is None or if
    # row is not None, so prevent that possibility
    if col is None or row is not None:
        col_wrap = None
    # remap the image names to be better for plotting
    df = plotting._remap_image_names(df)
    img_order = plotting.get_order('image_name')
    if col == 'image_name':
        kwargs.setdefault('col_order', img_order)
    if row == 'image_name':
        kwargs.setdefault('row_order', img_order)
    if hue is not None:
        kwargs.setdefault('palette', plotting.get_palette(hue, df[hue].unique()))
    else:
        kwargs.setdefault('color', 'k')
    marker_adjust = {}
    dashes_dict = {}
    if style is not None:
        style_dict = plotting.get_style(style, df[style].unique())
        dashes_dict = style_dict.pop('dashes_dict', {})
        marker_adjust = style_dict.pop('marker_adjust', {})
        kwargs.setdefault('dashes', dashes_dict)
        kwargs.update(style_dict)

    kwargs.setdefault('join', False)
    kwargs.setdefault('sharey', True)
    g = sns.catplot(x='orientation_slice', y='orientation_amplitude', hue=hue,
                    col=col, row=row, col_wrap=col_wrap, estimator=estimator,
                    height=height, aspect=aspect, data=df, kind=kind,
                    legend=False, **kwargs)

    if marker_adjust:
        labels = {v: k for k, v in kwargs.get('markers', {}).items()}
        final_markers = plotting._marker_adjust(g.axes.flatten(),
                                                marker_adjust, labels)
    else:
        final_markers = {}
    g.set_ylabels('Amplitude')
    g.set_xlabels('Orientation')

    # make some nice xticklabels
    angles = [a/np.pi for a in sorted(df.orientation_slice.dropna().unique())]
    ticklabels = []
    for a in angles:
        if a % .25 == 0:
            f = Fraction(int(a*4), 4)
            if f.denominator == 1:
                ticklabels.append(str(int(a)))
            else:
                ticklabels.append(r"$\frac{%s\pi}{%s}$" %
                                  (f.numerator, f.denominator))
        else:
            ticklabels.append('')
    g.set_xticklabels(ticklabels)
    # create the legend
    plotting._add_legend(df, g, None, hue, style,
                         kwargs.get('palette', {}), final_markers,
                         dashes_dict)
    # we use spectra because it doesn't include np.nan from dummy rows
    title_str = (f"{'Demeaned ' if demean else ''}Orientation energy for"
                 f" {' and '.join(spectra.model.values)}"
                 f" metamers, {' and '.join(spectra.trial_type.values)}"
                 " comparisons\n")
    g.fig.suptitle(title_str, va='bottom')
    g.fig.subplots_adjust(wspace=.1)
    return g


def dacey_mcmc_plot(inf_data, df, aspect=1, logscale_axes=True, hdi=.95):
    """Plot data and predicted lines from Dacey 1992 data and our MCMC fit.

    We use MCMC to fit hinged line (with intercept) to Dacey 1992's dendritic
    field diameter data. This plots that data with the line with HDI.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by
        `other_data.run_phys_scaling_inference`.
    df : pd.DataFrame
        DataFrame containing the Dacey1992 data, saved at
        data/Dacey1992_RGC.csv
    aspect : float, optional
        Aspect of the axes
    logscale_axes : bool, optional
        If True, we logscale both axes. Else, they're linear.
    hdi : float, optional
        The width of the HDI to draw (in range (0, 1]). See docstring of
        fov.mcmc.inf_data_to_df for more details.

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the plot.

    """
    params = other_data.inf_data_to_df(inf_data, 'parameters', hdi=True,
                                       query_str="distribution=='posterior'")

    # helper function to get the predicted values
    def df_hinged_line(df, ecc=np.linspace(0, 80, 100)):
        df = df.set_index('variable')
        return other_data.hinged_line(ecc, df.loc['diameter_slope'].value,
                                      df.loc['diameter_hinge_ecc'].value,
                                      df.loc['diameter_intercept'].value)

    ecc = np.linspace(0, 80, 100)
    lines = []
    for n, g in params.groupby(['distribution', 'cell_type', 'hdi']):
        data = dict(zip(['distribution', 'cell_type', 'hdi'], n))
        line = df_hinged_line(g, ecc)
        data.update({'diameter': line, 'eccentricity': ecc})
        lines.append(pd.DataFrame(data))
    lines = pd.concat(lines)

    df['dendritic_field_diameter_deg'] = df.dendritic_field_diameter_min / 60

    pal = plotting.get_palette('cell_type', df.cell_type.unique())
    g = sns.relplot(data=df, x='eccentricity_deg',
                    y='dendritic_field_diameter_deg',
                    hue='cell_type', aspect=aspect, palette=pal)
    for n, gb in lines.groupby('cell_type'):
        plotting.scatter_ci_dist(data=gb, x='eccentricity', y='diameter',
                                 ci='hdi', join=True, ci_mode='fill',
                                 ax=g.ax, draw_ctr_pts=False, color=pal[n])
    if logscale_axes:
        g.set(xscale='log', yscale='log')
    g.set(xlabel="Eccentrictiy (degrees)",
          ylabel="Dendritic field diameter (degrees)")
    return g


def compare_distance_and_performance(expt_df, dist_df, col='image_name',
                                     row=None, hue='scaling',
                                     style='trial_type', col_wrap=5, height=3,
                                     logscale_xaxis=False):
    """Compare model distance and human performance.

    This is intended for use with a single distance model (e.g., the
    ObserverModel), to visualize the relationship between model distance and
    human behavioral performance. We just plot them against each other, not
    doing anything to try and use distance to predict discriminability or
    performance directly; this is more preliminary.

    Two notes:

    - We expect this to be called with images from a single synthesis model.

    - We perform an inner join between the dist_df and expt_df, so we will only
      plot points for those pairs of images that have behavioral data (and
      distances, but that's cheap to calculate). In particular, this means
      there will not be many RGC metamer_vs_metamer points.

    - We average over different samples. For each comparison, synthesis model,
      reference image, and scaling we have 3 separate synthesized images. We
      average the distance and performance across these.
   
    Parameters
    ----------
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    dist_df : pd.DataFrame
        DataFrame containing the distance between images for a single distance
        model.
    col, row, hue, style : str or None, optional
        The variables in expt_df to facet along the columns, rows, hues, and
        styles, respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception.
        Ignored if col=None.
    height : float, optional
        Height of the axes.
    logscale_xaxis : bool, optional
        If True, we logscale the x-axis. Else, it's a linear scale.

    """
    if dist_df.distance_model.nunique() > 1 or dist_df.distance_scaling.nunique() > 1:
        raise Exception("Haven't thought through how to create this plot with more than one"
                        " distance model / scaling!")
    if expt_df.model.nunique() > 1:
        raise Exception("Haven't thought through how to create this plot with more than one"
                        " synthesis model!")
    model_name = f'{dist_df.distance_model.unique()[0]}({dist_df.distance_scaling.unique()[0]})'
    dist_df = dist_df.groupby(['distance_model', 'distance_scaling', 'synthesis_model',
                               'ref_image', 'synthesis_scaling', 'trial_type']).distance.mean().reset_index()

    expt_df = analysis.summarize_expt(expt_df, ['scaling', 'trial_type'])
    expt_df = plotting._remap_image_names(expt_df)
    dist_df = dist_df.rename(columns={'synthesis_scaling': 'scaling', 'synthesis_model': 'model',
                                      'ref_image': 'image_name'})

    # using an inner merge means we drop the columns where we don't have
    # behavioral data
    dist_df = dist_df.merge(expt_df, 'inner')

    kwargs = {}
    marker_adjust = {}
    if style is not None:
        style_dict = plotting.get_style(style, dist_df[style].unique())
        # we never want to connect these points, so don't do anything for
        # dashes
        dashes_dict = style_dict.pop('dashes_dict', {})
        marker_adjust = style_dict.pop('marker_adjust', {})
        kwargs.update(style_dict)
    # seaborn raises an error if col_wrap is non-None when col is None or row
    # is not None, so prevent that possibility
    if col is None or row is not None:
        col_wrap = None
    if hue is not None:
        kwargs.setdefault('palette', plotting.get_palette(hue, dist_df[hue].unique()))
    if col == 'image_name':
        img_order = plotting.get_order('image_name')
        kwargs.setdefault('col_order', img_order)

    g = plotting.lineplot_like_pointplot(dist_df, x='distance',
                                         y='proportion_correct', hue=hue,
                                         col=col, style=style,
                                         col_wrap=col_wrap, height=height,
                                         legend=False, linestyle='',
                                         **kwargs)
    if logscale_xaxis:
        g.set(xscale='log')

    if marker_adjust:
        labels = {v: k for k, v in kwargs.get('markers', {}).items()}
        final_markers = plotting._marker_adjust(g.axes.flatten(),
                                                marker_adjust, labels)
    else:
        final_markers = {}

    # create the legend
    plotting._add_legend(dist_df, g, None, hue, style,
                         kwargs.get('palette', {}), final_markers,
                         dashes_dict, 'brief')

    # Clean up labels

    # got this from https://stackoverflow.com/a/36369238/4659293
    n_rows, n_cols = g.fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    # we want to add some newlines at end of title, based on number of rows, to
    # make sure there's enough space
    end_newlines = ''
    if n_rows > 1:
        end_newlines += '\n\n'
    if n_rows > 3:
        end_newlines += '\n'
    if n_rows > 10:
        end_newlines += '\n\n'
    g.fig.suptitle(f'{model_name} distance vs. behavioral performance for '
                   f'synthesis model {dist_df.model.unique()[0]}{end_newlines}',
                   va='bottom')
    g.set_titles('{col_name}')
    # got this from https://stackoverflow.com/a/36369238/4659293
    n_rows, n_cols = g.axes[0].get_subplotspec().get_gridspec().get_geometry()
    y_idx = n_cols * ((n_rows-1)//2)
    if n_rows % 2 == 0:
        yval = 0
    else:
        yval = .5
    x_idx = -((n_cols+1)//2)
    if n_cols % 2 == 0:
        xval = 0
    else:
        xval = .5
    ylabel = 'Average proportion correct'
    xlabel = 'Model distance'
    g.set(xlabel='', ylabel='')
    g.fig.subplots_adjust(hspace=.2, wspace=.1, top=1)
    g.axes[y_idx].set_ylabel(ylabel, y=yval, ha='center')
    g.axes[x_idx].set_xlabel(xlabel, x=xval, ha='center')

    return g
