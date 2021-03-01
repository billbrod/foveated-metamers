"""code to generate figures for the project
"""
import imageio
import torch
import re
import numpy as np
import pyrtools as pt
import plenoptic as po
from skimage import measure
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as op
from . import utils, plotting, analysis

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


def pooling_window_example(windows, image, target_eccentricity=24,
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
    metamer : po.synth.Metamer
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
        model_constructor = po.simul.PooledRGC.from_state_dict_reduced
    elif model_name.startswith('V1'):
        model_constructor = po.simul.PooledV1.from_state_dict_reduced
    metamer = po.synth.Metamer.load(metamer_save_path, model_constructor=model_constructor)
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
    windows : po.simul.PoolingWindows
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
    print(errors.shape)
    print(titles, len(titles))
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


def performance_plot(expt_df, col='image_name', row=None, hue=None, col_wrap=5,
                     ci=95, comparison='ref'):
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
    col, row, hue : str or None, optional
        The variables in expt_df to facet along the columns, rows, and hues,
        respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception
    ci : int, optional
        What confidence interval to draw on the performance.
    comparison : {'ref', 'met'}, optional
        Whether this comparison is between metamers and reference images
        ('ref') or two metamers ('met').

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    ## CHANGE THIS TO A LINEPLOT, using markers=True, err_style='bars' (and
    ## might need to turn on style)
    g = sns.catplot(x='scaling', y='hit_or_miss_numeric', data=expt_df,
                    kind='point', col=col, row=row, hue=hue,
                    col_order=sorted(expt_df[col].unique()), ci=ci,
                    col_wrap=col_wrap)

    g.map_dataframe(plotting.map_flat_line, x='scaling', y=.5, colors='k')
    g.set_ylabels(f'Proportion correct (with {ci}% CI)')
    g.set_xlabels('Scaling')
    g.set(ylim=(.3, 1.05))
    g = plotting.title_experiment_summary_plots(g, expt_df, 'Performance',
                                                comparison)
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
    g = plotting.title_experiment_summary_plots(g, expt_df, 'Run length',
                                                comparison)
    return g


def compare_loss_and_performance_plot(expt_df, stim_df, col='scaling',
                                      row=None, hue='image_name', col_wrap=4):
    """Plot synthesis loss and behavioral performance against each other.

    Plots synthesis loss on the x-axis and proportion correct on the y-axis, to
    see if there's any relationship there. Hopefully, there's not (that is,
    synthesis progressed to the point where there's no real difference in
    image from more iterations)

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
    col, row, hue : str or None, optional
        The variables in expt_df to facet along the columns, rows, and hues,
        respectively.
    col_wrap : int or None, optional
        If row is None, how many columns to have before wrapping to the next
        row. If this is not None and row is not None, will raise an Exception

    Returns
    -------
    g : sns.FacetGrid
        FacetGrid containing the figure.

    """
    if expt_df.unique_seed.hasnans:
        raise Exception("There's a NaN in expt_df.unique_seed! This means that "
                        "this expt_df comes from a metamer_vs_metamer run and "
                        "we don't know how to handle that yet!")
    # need to get proportion_correct, not the raw responses, for this plot.
    # adding session_number here doesn't change results except to make sure
    # that the session_number column is preserved in the output (each session
    # contains all trials, all scaling for a given image)
    expt_df = analysis.summarize_expt(expt_df, ['session_number', 'scaling',
                                                'trial_type', 'unique_seed'])
    expt_df = expt_df.set_index(['subject_name', 'session_number', 'image_name',
                                 'scaling', 'unique_seed'])
    stim_df = stim_df.rename(columns={'seed': 'unique_seed'})
    stim_df = stim_df.set_index(['image_name', 'scaling',
                                 'unique_seed'])['loss'].dropna()
    expt_df = expt_df.merge(stim_df, left_index=True,
                            right_index=True,).reset_index()
    col_order, hue_order, row_order = None, None, None
    if col is not None:
        col_order = sorted(expt_df[col].unique())
    if row is not None:
        row_order = sorted(expt_df[row].unique())
    if hue is not None:
        hue_order = sorted(expt_df[hue].unique())
    g = sns.relplot(data=expt_df, x='loss', y='proportion_correct',
                    hue=hue, col=col, kind='scatter', col_wrap=col_wrap,
                    height=3, row=row, col_order=col_order,
                    hue_order=hue_order, row_order=row_order)
    g.set(xscale='log', xlim=plotting.get_log_ax_lims(expt_df.loss),
          ylabel='Proportion correct')
    g = plotting.title_experiment_summary_plots(g, expt_df,
                                                'Performance vs. synthesis loss',
                                                'ref', '\nHopefully no relationship here')
    return g
