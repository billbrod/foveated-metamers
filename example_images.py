#!/usr/bin/env python3

import yaml
import subprocess
import os.path as op
import argparse
from glob import glob
from foveated_metamers import stimuli


def main(model, subj_name, sess_num, comparison='ref'):
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)

    im_names = stimuli.get_images_for_session(subj_name, sess_num)
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
    elif comparison == 'met':
        scaling = config[model]['scaling'] + config[model]['met_v_met_scaling']
        scaling = [scaling[-8], scaling[-1]]
    ref_images = []
    high_scaling_mets = []
    low_scaling_mets = []
    print(im_names)
    for im in im_names:
        ref_images.append(op.join(config["DATA_DIR"], 'ref_images_preproc', f"{im}.png"))
        high_scaling_mets.append(glob(op.join(config['DATA_DIR'], 'metamers', model_name, f"{im}", f'scaling-{scaling[0]}', '*', '*',
                                              'seed-*_init-white_*metamer.png'))[0])
        low_scaling_mets.append(glob(op.join(config['DATA_DIR'], 'metamers', model_name, f"{im}", f'scaling-{scaling[-1]}', '*', '*',
                                             'seed-*_init-white_*metamer.png'))[0])
    # don't show natural images if comparison == met, because they won't see them
    if comparison == 'ref':
        subprocess.Popen(['eog', *ref_images], shell=False)
    subprocess.Popen(['eog', *low_scaling_mets], shell=False)
    subprocess.Popen(['eog', *high_scaling_mets], shell=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Open up some example metamer and reference images. See repo README for details")
    parser.add_argument('model',
                        help="{RGC, V1}. Which model to show examples from")
    parser.add_argument('subj_name', help="Name of the subject", type=str)
    parser.add_argument("sess_num", help="Number of the session", type=int)
    parser.add_argument("--comparison", '-c', default='ref',
                        help=("{ref, met}. Whether this run is comparing metamers against "
                              "reference images or other metamers."))
    args = vars(parser.parse_args())
    main(**args)
