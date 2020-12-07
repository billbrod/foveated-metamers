#!/usr/bin/env python3

import yaml
import subprocess
import os.path as op
import argparse
from glob import glob


def main(model):
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)

    im_names = ['azulejos', 'tiles']
    model_name = config[model]['model_name']
    scaling = [config[model]['scaling'][0], config[model]['scaling'][-1]]
    for im in im_names:
        ref_images = op.join(config["DATA_DIR"], 'ref_images_preproc', f"{im}_range-.05,.95_size-2048,2600.png")
        metamers = [glob(op.join(config['DATA_DIR'], 'metamers', model_name, f"{im}*", f'scaling-{s}', '*', '*',
                                 'seed-0_init-white_*metamer.png'))[0] for s in scaling]
        subprocess.Popen(['eog', ref_images, *metamers, ], shell=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open up some example metamer and reference images")
    parser.add_argument('model',
                        help="{RGC, V1}. Which model to show examples from")
    args = vars(parser.parse_args())
    main(**args)
