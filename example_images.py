#!/usr/bin/env python3

import yaml
import subprocess
import os.path as op
import argparse
from glob import glob
from foveated_metamers import stimuli


def main(model, subj_name, sess_num):
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)

    im_names = stimuli.get_images_for_session(subj_name, sess_num)
    model_name = config[model]['model_name']
    scaling = [config[model]['scaling'][0], config[model]['scaling'][-1]]
    ref_images = []
    high_scaling_mets = []
    low_scaling_mets = []
    print(im_names)
    for im in im_names:
        ref_images.append(op.join(config["DATA_DIR"], 'ref_images_preproc', f"{im}_range-.05,.95_size-2048,2600.png"))
        low_scaling_mets.append(glob(op.join(config['DATA_DIR'], 'metamers', model_name, f"{im}*", f'scaling-{scaling[0]}', '*', '*',
                                 'seed-0_init-white_*metamer.png'))[0])
        low_scaling_mets.append(glob(op.join(config['DATA_DIR'], 'metamers', model_name, f"{im}*", f'scaling-{scaling[-1]}', '*', '*',
                                 'seed-0_init-white_*metamer.png'))[0])
    subprocess.Popen(['eog', ref_images], shell=False)
    subprocess.Popen(['eog', low_scaling_mets], shell=False)
    subprocess.Popen(['eog', high_scaling_mets], shell=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Open up some example metamer and reference images. See repo README for details")
    parser.add_argument('model',
                        help="{RGC, V1}. Which model to show examples from")
    parser.add_argument('subj_name', help="Name of the subject", type=str)
    parser.add_argument("sess_num", help="Number of the session", type=int)
    args = vars(parser.parse_args())
    main(**args)
