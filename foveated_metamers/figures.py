"""code to generate figures for the project
"""
import imageio
import numpy as np
import pyrtools as pt
import os.path as op
from .utils import convert_im_to_float

V1_TEMPLATE_PATH = op.join('/home/billbrod/Desktop/metamers', 'metamers_display', 'V1_cone-1.0_'
                           'norm_s6_gaussian', '{image_name}', 'scaling-{scaling}', 'opt-Adam',
                           'fr-0_lc-1_cf-0.01_clamp-True', 'seed-{seed}_init-white_lr-{learning_'
                           'rate}_e0-0.5_em-41_iter-{max_iter}_thresh-1e-08_gpu-{gpu}_metamer_'
                           'gamma-corrected.png')
RGC_TEMPLATE_PATH = op.join('/home/billbrod/Desktop/metamers', 'metamers_display', 'RGC_cone-1.0_'
                            'gaussian', '{image_name}', 'scaling-{scaling}', 'opt-Adam', 'fr-0_lc-'
                            '1_cf-0_clamp-True', 'seed-{seed}_init-white_lr-0.1_e0-3.71_em-41_iter'
                            '-750_thresh-1e-08_gpu-0_metamer_gamma-corrected.png')
REFERENCE_PATH = op.join('/home/billbrod/Desktop/metamers', 'ref_images_preproc',
                         '{image_name}.png')


def scaling_comparison_figure(image_name, scaling_vals, seed, window_size=400,
                              periphery_offset=(-800, -1000), max_ecc=41,
                              ref_template_path=REFERENCE_PATH,
                              metamer_template_path=V1_TEMPLATE_PATH, **template_kwargs):
    r"""Create a figure showing cut-out views of all scaling values

    We want to be able to easily visually compare metamers across
    scaling values (and with the reference image), but they're very
    large. In order to facilitate this, we create this figure with
    'cut-out' views, where we compare the reference image and metamers
    made with a variety of scaling values (all same seed) at the fovea
    and the periphery, with some information about the extent.

    Parameters
    ----------
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
    max_ecc : float
        The maximum eccentricity of the metamers, as passed to the
        model. Used to convert from pixels to degrees so we know the
        extent and location of the cut-out views in degrees.
    ref_template_path : str
        Template path to the reference image, should contain
        '{image_name}'. See figures.REFERENCE_PATH global variable for
        an example (and recommended version)
    metamer_template_path : str
        Template path to gamma-corrected metamers, should contain
        '{image_name}', '{scaling}', '{seed}'. It can contain more
        format strs, in which case you should pass dictionaries as
        template_kwargs to specify how to fill them in. See
        figures.RGC_TEMPLATE_PATH or figures.V1_TEMPLATE_PATH for
        examples (these are the recommended ones)
    template_kwargs : dict
        Every additional kwarg should be a dictionary of (scaling, val)
        pairs (with an entry for each value in ``scaling_vals``) that
        tells us how to fill in the extra format strs in
        metamer_template_path. figures.V1_TEMPLATE_PATH contains
        '{gpu}', and so you'll need to also pass, for example,
        ``gpu=dict((sc, 1) for sc in scaling_vals)`` (though, since this
        has the same value for each scaling, it's uninteresting; the
        point of this is to enable you to have different values for
        different scaling)

    Returns
    -------
    fig
        The matplotlib figure with the scaling comparison plotted on it

    """
    ref_path = ref_template_path.format(image_name=image_name.replace('cone_', 'gamma-corrected_'))
    images = [convert_im_to_float(imageio.imread(ref_path))]
    for sc in scaling_vals:
        sc_kwargs = dict((k, v[sc]) for k, v in template_kwargs.items())
        im_path = metamer_template_path.format(image_name=image_name, seed=seed, scaling=sc,
                                               **sc_kwargs)
        images.append(convert_im_to_float(imageio.imread(im_path)))
    # want our images to be indexed along the first dimension
    images = np.einsum('ijk -> kij', np.dstack(images))
    im_ctr = [s//2 for s in images.shape[1:]]
    fovea_bounds = [im_ctr[0]-window_size//2, im_ctr[0]+window_size//2,
                    im_ctr[1]-window_size//2, im_ctr[1]+window_size//2]
    fovea = [im[fovea_bounds[0]:fovea_bounds[1], fovea_bounds[2]:fovea_bounds[3]] for im in images]
    periphery = [im[fovea_bounds[0]-periphery_offset[0]:fovea_bounds[1]-periphery_offset[0],
                    fovea_bounds[2]-periphery_offset[1]:fovea_bounds[3]-periphery_offset[1]]
                 for im in images]
    # max_ecc is the distance from the center to the edge of the image,
    # so we want double this to get the full width of the image
    pix_to_deg = (2 * max_ecc) / max(images.shape[1:])
    window_extent_deg = (window_size//2) * pix_to_deg
    periphery_ctr_deg = np.sqrt(np.sum([(s*pix_to_deg)**2 for s in periphery_offset]))
    fig = pt.imshow(fovea+periphery, vrange=(0, 1), title=None, col_wrap=len(fovea))
    fig.axes[0].set(title='Reference', ylabel='Fovea\n($\pm$%.01f deg)' % window_extent_deg)
    fig.axes[len(fovea)].set(ylabel='Periphery\n(%.01f$\pm$%.01f deg)' % (periphery_ctr_deg,
                                                                          window_extent_deg))
    for i, sc in zip(range(1, len(fovea)), scaling_vals):
        fig.axes[i].set(title='scaling=%.03f' % sc)
    return fig
