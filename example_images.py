#!/usr/bin/env python3

import yaml
import subprocess
import os.path as op
import argparse
from glob import glob
from foveated_metamers import stimuli, utils


def main(model, subj_name, sess_num, comparison='ref'):
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)

    im_names = stimuli.get_images_for_session(subj_name, sess_num, 'downsample' in comparison)
    try:
        model_name = config[model]['model_name']
    except KeyError:
        model_name = model
    idx_paths = [op.join(config['DATA_DIR'], 'stimuli', model_name, f'task-split_comp-{comparison}', subj_name,
                         f'{subj_name}_task-split_comp-{comparison}_idx_sess-{sess_num:02d}_run-{r:02d}.npy')
                 for r in range(5)]
    for p in idx_paths:
        if not op.isfile(p):
            raise Exception(f"Index path {p} not found!")
    if comparison == 'ref':
        scaling = [config[model]['scaling'][0], config[model]['scaling'][-1]]
    elif comparison.startswith('met'):
        scaling = config[model]['scaling'] + config[model]['met_v_met_scaling']
        scaling = [scaling[-8], scaling[-1]]
    ref_images = []
    high_scaling_mets = []
    high_scaling_mets_2 = []
    low_scaling_mets = []
    low_scaling_mets_2 = []
    print(im_names)
    for im in im_names:
        ref_images.append(op.join(config["DATA_DIR"], 'ref_images_preproc', f"{im}.png"))
        # these will each have one image of each scaling, so grab the first and
        # last for low and high
        mets = utils.generate_metamer_paths(model_name, image_name=im, seed_n=0, comp=comparison)
        mets_2 = utils.generate_metamer_paths(model_name, image_name=im, seed_n=1, comp=comparison)
        high_scaling_mets.append(mets[0])
        low_scaling_mets.append(mets[-1])
        high_scaling_mets_2.append(mets_2[0])
        low_scaling_mets_2.append(mets_2[-1])
    # don't show natural images if comparison.startswith('met'), because they won't see them
    if comparison.startswith('ref'):
        subprocess.Popen(['eog', *ref_images], shell=False)
    subprocess.Popen(['eog', *low_scaling_mets], shell=False)
    subprocess.Popen(['eog', *high_scaling_mets], shell=False)
    if comparison.startswith('met'):
        subprocess.Popen(['eog', *low_scaling_mets_2], shell=False)
        subprocess.Popen(['eog', *high_scaling_mets_2], shell=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Open up some example metamer and reference images. See repo README for details")
    parser.add_argument('model',
                        help="{RGC, V1}. Which model to show examples from")
    parser.add_argument('subj_name', help="Name of the subject", type=str)
    parser.add_argument("sess_num", help="Number of the session", type=int)
    parser.add_argument("--comparison", '-c', default='ref',
                        help=("{ref, met, met-downsample-2, ref-natural, met-natural}."
                              " What comparison to show example images for"))
    args = vars(parser.parse_args())
    main(**args)
