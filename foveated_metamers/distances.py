#!/usr/bin/env python3
"""functions related to calculating distances
"""

import pandas as pd
import plenoptic as po
from . import utils
import os.path as op
import itertools
import re


def _find_seed(x):
    """Grabs seed from image name.

    if can't find seed, returns 'ref'

    """
    try:
        return re.findall('seed-(\d)_', x)[0]
    except IndexError:
        return 'ref'


def model_distance(model, synth_model_name, ref_image_name, scaling):
    """Calculate distances between images for a model.

    We want to reason about the distance

    """
    paths = utils.generate_metamer_paths(synth_model_name,
                                         image_name=ref_image_name,
                                         scaling=scaling)
    synth_images = po.load_images(paths)
    ref_image = po.load_images(utils.get_ref_image_full_path(ref_image_name))
    ref_image_rep = model(ref_image)
    df = []
    reps = {}
    for i, (im, p) in enumerate(zip(synth_images, paths)):
        image_name = op.splitext(op.basename(p))[0]
        reps[image_name] = model(im)
        dist = po.optim.l2_norm(reps[image_name], ref_image_rep).item()
        df.append(pd.DataFrame({'distance': dist, 'image_1': image_name,
                                'image_2': ref_image_name}, index=[0]))
    for im_1, im_2 in itertools.combinations(reps, 2):
        dist = po.optim.l2_norm(reps[im_1], reps[im_2]).item()
        df.append(pd.DataFrame({'distance': dist, 'image_1': im_1,
                                'image_2': im_2}, index=[0]))
    df = pd.concat(df).reset_index(drop=True)
    # df['distance_model'] = model
    df['synthesis_model'] = synth_model_name
    df['synthesis_scaling'] = scaling
    df['ref_image'] = ref_image_name.split('_')[0]
    df['image_1_seed'] = df.image_1.apply(_find_seed)
    df['image_2_seed'] = df.image_2.apply(_find_seed)
    return df
