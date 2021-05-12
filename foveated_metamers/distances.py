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


def model_distance(model, synth_model_name, ref_image_name, scaling):
    """Calculate distances between images for a model.

    We want to reason about the model distance of our best model (the MSE in
    model space, same as used during synthesis except without the range
    penalty) between images synthesized by other models / scaling values. This
    is a step on the way towards getting a human perceptual metric, and should
    show us that the metamer-metamer distance, even for high scaling values, is
    still pretty small, while the metamer-reference distance is fairly large.

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
    met_natural_imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
    met_natural_scaling = config[synth_model_name.split('_')[0]]['scaling'][2:]
    met_natural_scaling += config[synth_model_name.split('_')[0]]['met_v_met_scaling'][:2]
    if (synth_model_name.startswith("V1") and
        any([ref_image_name.startswith(im) for im in met_natural_imgs]) and
        scaling in met_natural_scaling):
        paths += utils.generate_metamer_paths(synth_model_name,
                                              image_name=ref_image_name,
                                              comp='met-natural',
                                              scaling=scaling)
    synth_images = po.load_images(paths)
    ref_image = po.load_images(utils.get_ref_image_full_path(ref_image_name))
    ref_image_rep = model(ref_image)
    df = []
    reps = OrderedDict()
    for i, (im, p) in enumerate(zip(synth_images, paths)):
        image_name = op.splitext(op.basename(p))[0]
        reps[image_name] = model(im)
        dist = pop.optim.mse(reps[image_name], ref_image_rep).item()
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
        dist = pop.optim.mse(reps[im_1], reps[im_2]).item()
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
    metamer_vs_reference = np.logical_or((df.image_1_seed == 'reference').values,
                                         (df.image_2_seed == 'reference').values)
    df['trial_type'] = np.where(metamer_vs_reference, 'metamer_vs_reference',
                                'metamer_vs_metamer')
    return df
