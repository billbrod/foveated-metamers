#!/usr/bin/env python3
"""functions related to calculating distances
"""

import yaml
import pandas as pd
import numpy as np
import plenoptic as po
from . import utils
import os.path as op
import itertools
from collections import OrderedDict
import re
import sys
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages'))
import plenoptic_part as pop


def _create_bar_mask(bar_height, bar_width, fringe_proportion=.5):
    """Create central bar with raised-cosine edges.

    This is almost the same as the one we use for the experiment, except it
    runs from 0 (fully opaque) to 1 (fully transparent), whereas, for
    psychophy's mask, it ran from 1 (fully opaque) to -1 (fully transparent).

    """
    x = np.linspace(-bar_width//2, bar_width//2, bar_width)
    fringe_width = fringe_proportion * x.max()
    def raised_cos(x, start_x, end_x):
        x = (x-start_x) / (end_x - start_x)
        return .5*(1+np.cos(np.pi*x))
    mask = np.piecewise(x, [x < -fringe_width, (x > -fringe_width) & (x < fringe_width), fringe_width < x],
                       [lambda x: raised_cos(x, -fringe_width, x.min()), 1,
                        lambda x: raised_cos(x, fringe_width, x.max())])
    # this is different than our psychopy mask, where 1 means fully opaque and -1 means fully transparent.
    # here, we want it to go from 0 (fully opaque) to 1 (fully transparent)
    mask = 1 - mask
    mask = np.repeat(np.expand_dims(mask, 0), bar_height, 0)
    return mask


def _add_bar(img, bar):
    """Add bar to center of img."""
    img_half_width = img.shape[-1] // 2
    bar_half_width = bar.shape[-1] // 2
    img[..., img_half_width-bar_half_width:img_half_width+bar_half_width] *= bar
    return img


def _find_seed(x):
    """Grabs seed from image name.

    if can't find seed, returns 'reference'

    """
    try:
        return re.findall('seed-(\d+)_', x)[0]
    except IndexError:
        return 'reference'


def _find_init_type(x):
    """Grab init string from image name.

    If can't find it, returns 'reference'
    """
    try:
        return re.findall('init-(.+)_lr', x)[0]
    except IndexError:
        return 'reference'


def _grab_seed_n(x):
    """Grab seed n from full seed

    If it's not an int, return 'reference'.
    """
    try:
        return int(str(x)[-1])
    except ValueError:
        return x


def model_distance(model, synth_model_name, ref_image_name, scaling,
                   distance_func=pop.optim.l2_norm):
    """Calculate distances between images for a model.

    We want to reason about the model distance of our best model (by default,
    the L2-norm of the difference in model space) between images synthesized by
    other models / scaling values. This is a step on the way towards getting a
    human perceptual metric, and should show us that the metamer-metamer
    distance, even for high scaling values, is still pretty small, while the
    metamer-reference distance is fairly large.

    This loads in the specified reference image and all metamers with the given
    scaling value (we use `utils.generate_metamer_paths` to find them), then
    computes the distance between the reference image and each metamer, as well

    Note: the distance computed here between a metamer and its reference image
    will not be the same as the synthesis loss, because we are here comparing
    the 8bit images (as those are the ones shown in the experiment), rather than
    the 32bit tensors used during synthesis (there's also a small difference
    between model outputs when run on GPU vs CPU, but the save precision is the
    main one).

    Parameters
    ----------
    model : po.synth model
        Instantiated model, which takes image tensor as input and returns some
        output.
    synth_model_name : str
        str defining the name of the model used to synthesize the images we're
        checking (e.g., "V1_norm_s6_gaussian").
    ref_image_name : str
        str giving the name of the reference image (like those in
        config.yml:DEFAULT_METAMERS:image_name) for the metamers to compare.
    scaling : float
        Scaling value for the synthesized images.
    distance_func : function
        Function that accepts two tensors and returns the distance between
        them. By default, this is the L2-norm of their difference. Synthesis
        loss used pop.optim.mse_and_penalize_range, the weighted average of the
        MSE and a range penalty

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the distances. Contains column identifying the
        synthesis model and scaling, but not the distance model and scaling

    """
    paths = utils.generate_metamer_paths(synth_model_name,
                                         image_name=ref_image_name,
                                         scaling=scaling)
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        config = yaml.safe_load(f)
    # the scaling values used for the ref-natural comparison are the same as
    # those used for the ref comparison
    ref_natural_scaling = config[synth_model_name.split('_')[0]]['scaling']
    if synth_model_name.startswith("V1") and scaling in ref_natural_scaling:
        paths += utils.generate_metamer_paths(synth_model_name,
                                              image_name=ref_image_name,
                                              comp='ref-natural',
                                              scaling=scaling)
    met_natural_scaling = config[synth_model_name.split('_')[0]]['scaling'][2:]
    met_natural_scaling += config[synth_model_name.split('_')[0]]['met_v_met_scaling'][:2]
    if synth_model_name.startswith("V1") and scaling in met_natural_scaling:
        paths += utils.generate_metamer_paths(synth_model_name,
                                              image_name=ref_image_name,
                                              comp='met-natural',
                                              scaling=scaling)
    synth_images = po.load_images(paths)
    ref_image = po.load_images(utils.get_ref_image_full_path(ref_image_name))
    ref_image_rep = model(ref_image)
    df = []
    reps = OrderedDict()
    # the unsqueeze is to make sure that the images are 4d when passed to the
    # model, as expected
    for i, (im, p) in enumerate(zip(synth_images.unsqueeze(1), paths)):
        image_name = op.splitext(op.basename(p))[0]
        reps[image_name] = model(im)
        dist = distance_func(reps[image_name], ref_image_rep).item()
        df.append(pd.DataFrame({'distance': dist, 'image_1': image_name,
                                'image_2': ref_image_name}, index=[0]))
    rep_keys = list(reps.keys())
    if len(rep_keys) != 3 and len(rep_keys) != 6:
        raise Exception("We need either 3 (all init-white) or 6 (init-white"
                        " and 3 different natural image intializations) "
                        "metamers, but got {len(rep_keys)} of them!")
    # # construct these in this particular order so that each seed_n shows up once
    # # in each column
    # met_comparisons = [(rep_keys[0], rep_keys[1]), (rep_keys[1], rep_keys[2]),
    #                    (rep_keys[2], rep_keys[0])]
    # if len(rep_keys) > 3:
    #     met_comparisons += [(rep_keys[3], rep_keys[4]), (rep_keys[4], rep_keys[5]),
                            # (rep_keys[5], rep_keys[3])]
    met_comparisons = itertools.combinations(rep_keys, 2)
    for im_1, im_2 in met_comparisons:
        dist = distance_func(reps[im_1], reps[im_2]).item()
        df.append(pd.DataFrame({'distance': dist, 'image_1': im_1,
                                'image_2': im_2}, index=[0]))
    df = pd.concat(df).reset_index(drop=True)
    df['synthesis_model'] = synth_model_name
    df['synthesis_scaling'] = scaling
    df['ref_image'] = ref_image_name.split('_')[0]
    df['image_1_seed'] = df.image_1.apply(_find_seed)
    df['image_2_seed'] = df.image_2.apply(_find_seed)
    df['image_1_seed_n'] = df.image_1_seed.apply(_grab_seed_n)
    df['image_2_seed_n'] = df.image_2_seed.apply(_grab_seed_n)
    df['image_1_init_type'] = df.image_1.apply(_find_init_type)
    df['image_2_init_type'] = df.image_2.apply(_find_init_type)
    df['image_1_init_supertype'] = df.image_1_init_type.apply(lambda x: {'white': 'noise', 'reference': 'reference'}.get(x, 'natural'))
    df['image_2_init_supertype'] = df.image_2_init_type.apply(lambda x: {'white': 'noise', 'reference': 'reference'}.get(x, 'natural'))

    def get_trial_type(x):
        trial_type = 'metamer_vs_'
        if 'reference' in x.image_1_init_supertype or 'reference' in x.image_2_init_supertype:
            trial_type += 'reference'
        else:
            trial_type += 'metamer'
        if 'natural' in x.image_1_init_supertype or 'natural' in x.image_2_init_supertype:
            trial_type += '-natural'
        # we did not do this type of trial
        if 'noise' in x.image_1_init_supertype and 'natural' in x.image_2_init_supertype:
            trial_type = 'not_run'
        return trial_type
    df['trial_type'] = df.apply(get_trial_type, 1)
    return df


def calculate_experiment_mse(stim, trial, bar_deg_size=2., screen_size_deg=73.45,
                             screen_size_pix=3840):
    """Calculate MSE for a single trial of the experiment.

    We calculate the MSE between the images as displayed in the experiment:
    gray bar down the center and only one side changing.

    Note that we don't do anything to rescale the values in the stim array, and
    the assumption is it contains the 8bit values going from 0 to 255

    Parameters
    ----------
    stim : np.ndarray
        Array of stimuli
    trial : np.ndarray
        2x2 array containing the indices presented in the trial, as you'd get
        from `idx[:, 0, :]`, where `idx` is the stimulus presentation index.
    bar_deg_size : float, optional
        Width of the bar, in degrees. Default matches experimental setup.
    screen_size_deg : float, optional
        Width of the screen, in degrees. Default matches experimental setup.
    screen_size_pix : float, optional
        Width of the screen, in pixels. Default matches experimental setup.

    Returns
    -------
    mse : float
        MSE between the two images presented in this trial.

    """
    bar_pix_size = int(bar_deg_size * (screen_size_pix / screen_size_deg))
    bar = _create_bar_mask(stim.shape[1], bar_pix_size)
    # unpack this to get the index of the stimulus on left and right, for first
    # and second image
    [[l1, l2], [r1, r2]] = trial

    # initialize the first and second image
    stim1 = np.empty_like(stim[0], dtype=float)
    stim2 = np.empty_like(stim[0], dtype=float)

    stim_half_width = stim.shape[-1] // 2
    # insert the contents into the two halves of the image
    stim1[:, :stim_half_width] = stim[l1, :, :stim_half_width]
    stim1[:, stim_half_width:] = stim[r1, :, stim_half_width:]
    stim2[:, :stim_half_width] = stim[l2, :, :stim_half_width]
    stim2[:, stim_half_width:] = stim[r2, :, stim_half_width:]

    # modulate the center by the masking bar
    stim1 = _add_bar(stim1, bar)
    stim2 = _add_bar(stim2, bar)

    # and return the MSE
    return np.square(stim1-stim2).mean()


def _get_seed_n(x):
    """Helper for expt_mse df."""
    try:
        # need to parse it as float first because int('0.0') will fail but
        # float('0.0') will not
        return int(float(x)) % 10
    except ValueError:
        return 'ref'

def _get_trial_structure(row):
    """Helper for expt_mse df."""
    l1 = _get_seed_n(row.image_left_1)
    l2 = _get_seed_n(row.image_left_2)
    r1 = _get_seed_n(row.image_right_1)
    r2 = _get_seed_n(row.image_right_2)
    if l1 == l2:
        change = 'R'
        second = r2
    elif r1 == r2:
        change = 'L'
        second = l2
    return f'{l1},{second},{change}'
