#!/usr/bin/env python3

import argparse
import subprocess
import os
import re
import yaml
import hashlib
import json
import os.path as op
from glob import glob
from foveated_metamers import utils


CHECKSUM_PATH = op.join(op.dirname(op.realpath(__file__)), 'data', 'checksums.json')


# dictionary of download urls from the OSF
OSF_URL = {
    'synthesis_input': 'https://osf.io/sw4tb/download',
    'stimuli': {
        'stimuli_luminance_ref': 'https://osf.io/3d49e/download',
        'stimuli_luminance_met': 'https://osf.io/z35hf/download',
        'stimuli_energy_ref': 'https://osf.io/26w3s/download',
        'stimuli_energy_ref-nat': 'https://osf.io/cp8ru/download',
        'stimuli_energy_met': 'https://osf.io/n9trc/download',
        'stimuli_energy_met-nat': 'https://osf.io/rt3wk/download',
        'stimuli_energy_met_downsample': 'https://osf.io/qxf6k/download',
    },
    'behavioral_data': 'https://osf.io/hf72g/download',
    'mcmc_fits': {
        'mcmc_luminance_ref_partially-pooled.nc': 'https://osf.io/a8fxz/download',
        'mcmc_luminance_met_partially-pooled.nc': 'https://osf.io/snjqb/download',
        'mcmc_energy_ref_partially-pooled.nc': 'https://osf.io/r324n/download',
        'mcmc_energy_ref-nat_partially-pooled.nc': 'https://osf.io/4gtkh/download',
        'mcmc_energy_met_partially-pooled.nc': 'https://osf.io/n7wc3/download',
        'mcmc_energy_met-nat_partially-pooled.nc': 'https://osf.io/dx9ew/download',
        'mcmc_energy_met_downsample_partially-pooled.nc': 'https://osf.io/9wnvq/download',
    },
    'figure_input': 'https://osf.io/hvrs2/download',
    'freeman2011_check_input': "https://osf.io/e2zn8/download",
    'freeman2011_check_output': "https://osf.io/wa2zu/download",
}


def check_checksum(path, checksum):
    with open(path, 'rb') as f:
        test_checksum = hashlib.blake2b(f.read())
    return test_checksum.hexdigest() == checksum


def main(target_dataset):
    """Download dataset from OpenNeuro or OSF.

    Parameters
    ----------
    target_dataset : {'synthesis_input', 'stimuli',
                      'behavioral_data', 'mcmc_fits',
                      'figure_input',
                      'freeman2011_check_input',
                      'freeman2011_check_output'}
        Which dataset to download. See project README for more info.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)
    if op.split(config['DATA_DIR'])[-1].lower() != op.split(config['DATA_DIR'])[-1]:
        raise Exception(f"Name of your DATA_DIR must be all lowercase! But got {config['DATA_DIR']}")
    with open(CHECKSUM_PATH) as f:
        checksums = json.load(f)
    data_dir = config['DATA_DIR']
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using {data_dir} as data root directory.")
    targets = ['synthesis_input', 'stimuli', 'behavioral_data', 'mcmc_fits']
    check_dirs = ['ref_images_preproc', 'stimuli', 'behavioral', 'mcmc']
    yesno = 'y'
    for tar, check, size in zip(targets, check_dirs, ['176MB', '12GB', '2.6MB', '12GB']):
        if target_dataset == tar:
            if op.exists(op.join(data_dir, check)):
                yesno = input("Previous data found, do you wish to download the data anyway? [y/n] ")
                while yesno not in ['y', 'n']:
                    print("Please enter y or n")
                    yesno = input("Previous data found, do you wish to download the data anyway? [y/n] ")
            yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
            while yesno not in ['y', 'n']:
                print("Please enter y or n")
                yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
    if yesno == 'n':
        print("Exiting...")
        exit(0)
    # dictionary mapping between the names used in the upload vs those in the actual data directory
    model_name_map = {'energy': 'V1_norm_s6_gaussian', 'luminance': 'RGC_norm_gaussian'}
    comp_name_map = lambda x: x.replace('-nat', '-natural').replace('_downsample', '-downsample-2')
    if target_dataset == 'synthesis_input':
        print("Downloading synthesis input.")
        synth_checksum = False
        while not synth_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['synthesis_input']])
            synth_checksum = check_checksum('synthesis_input.tar.gz', checksums['synthesis_input.tar.gz'])
        subprocess.call(["tar", "xf", "synthesis_input.tar.gz"])
        # this file is unnecessary for the experiment
        subprocess.call(['rm', op.join('synthesis_input', 'metadata.json')])
        subprocess.call(["rsync", "-avPLuz", "synthesis_input/", f"{data_dir}/"])
        subprocess.call(["rm", "-r", "synthesis_input/"])
        subprocess.call(["rm", "synthesis_input.tar.gz"])
    elif target_dataset == 'stimuli':
        print("Downloading stimuli for all comparisons.")
        for name, url in OSF_URL['stimuli'].items():
            print(f"Downloading {name}")
            download_model = re.findall('stimuli_([a-z]+)_', name)[0]
            output_model = model_name_map[download_model]
            os.makedirs(op.join(data_dir, "stimuli", output_model), exist_ok=True)
            stim_checksum = False
            while not stim_checksum:
                subprocess.call(["curl", "-O", "-J", "-L", url])
                stim_checksum = check_checksum(f'{name}.tar.gz', checksums[f'{name}.tar.gz'])
            subprocess.call(["tar", "xf", f"{name}.tar.gz"])
            for f in glob(op.join(name, 'stimuli*')):
                subprocess.call(["mv", f, op.join(data_dir, 'stimuli', output_model) + '/'])
            subprocess.call(["rm", '-r', name])
            subprocess.call(["rm", f"{name}.tar.gz"])
    elif target_dataset == 'behavioral_data':
        print("Downloading behavioral data for all comparisons.")
        behav_checksum = False
        while not behav_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['behavioral_data']])
            behav_checksum = check_checksum('behavioral_data.tar.gz', checksums['behavioral_data.tar.gz'])
        subprocess.call(["tar", "xf", "behavioral_data.tar.gz"])
        for f in glob(op.join('behavioral_data', '*csv')):
            download_model, download_comp = re.findall(op.join('behavioral_data', '([a-z]+)_comp-([a-z-_]+)_data.csv'), f)[0]
            outp = op.join(data_dir, 'behavioral', model_name_map[download_model], f'task-split_comp-{comp_name_map(download_comp)}',
                           f'task-split_comp-{comp_name_map(download_comp)}_data.csv')
            os.makedirs(op.dirname(outp), exist_ok=True)
            subprocess.call(["cp", f, outp])
        subprocess.call(["rm", "-r", "behavioral_data/"])
        subprocess.call(["rm", "behavioral_data.tar.gz"])
    elif target_dataset == 'mcmc_fits':
        print("Downloading MCMC fits for all comparisons.")
        for name, url in OSF_URL['mcmc_fits'].items():
            print(f"Downloading {name}")
            download_model, download_comp = re.findall('mcmc_([a-z]+)_([a-z-_]+)_partially-pooled.nc', name)[0]
            outp_model = model_name_map[download_model]
            outp_comp = comp_name_map(download_comp)
            hyper = utils.get_mcmc_hyperparams({'mcmc_model': 'partially-pooled',
                                                'model_name': outp_model, 'comp': outp_comp})
            outp = op.join(data_dir, 'mcmc', outp_model, f'task-split_comp-{outp_comp}',
                           f'task-split_comp-{outp_comp}_mcmc_partially-pooled_{hyper}_scaling-extended.nc')
            mcmc_checksum = False
            while not mcmc_checksum:
                subprocess.call(["curl", "-O", "-J", "-L", url])
                mcmc_checksum = check_checksum(name, checksums[name])
            os.makedirs(op.dirname(outp), exist_ok=True)
            subprocess.call(["mv", name, outp])
    elif target_dataset == 'figure_input':
        print("Downloading figure input.")
        fig_checksum = False
        while not fig_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['figure_input']])
            fig_checksum = check_checksum('figure_input.tar.gz', checksums['figure_input.tar.gz'])
        subprocess.call(["tar", "xf", "figure_input.tar.gz"])
        subprocess.call(["rsync", "-avPLuz", "figure_input/", f"{data_dir}/"])
        subprocess.call(["rm", "-r", "figure_input/"])
        subprocess.call(["rm", "figure_input.tar.gz"])
    elif target_dataset == 'freeman2011_check_input':
        print("Downloading input for comparison against Freeman2011.")
        met_dir = op.join(data_dir, 'freeman_check')
        os.makedirs(met_dir, exist_ok=True)
        ref_dir = op.join(data_dir, 'ref_images')
        os.makedirs(ref_dir, exist_ok=True)
        freeman_checksum = False
        while not freeman_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['freeman2011_check_input']])
            freeman_checksum = check_checksum('freeman_check_inputs.tar.gz', checksums['freeman_check_inputs.tar.gz'])
        subprocess.call(["tar", "xf", "freeman_check_inputs.tar.gz"])
        subprocess.call(["mv", "freeman_check_inputs/metamer1.png", f"{met_dir}/"])
        subprocess.call(["mv", "freeman_check_inputs/metamer2.png", f"{met_dir}/"])
        subprocess.call(["mv", "freeman_check_inputs/fountain_size-512,512.png", f"{ref_dir}/"])
        subprocess.call(["rm", "freeman_check_inputs.tar.gz"])
        subprocess.call(["rmdir", "freeman_check_inputs"])
    elif target_dataset == 'freeman2011_check_output':
        print("Downloading output for comparison against Freeman2011.")
        met_dir = op.join(data_dir, 'metamers')
        os.makedirs(met_dir, exist_ok=True)
        windows_dir = op.join(data_dir, 'freeman_check', 'windows')
        os.makedirs(windows_dir, exist_ok=True)
        freeman_checksum = False
        while not freeman_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['freeman2011_check_output']])
            freeman_checksum = check_checksum('freeman_check.tar.gz', checksums['freeman_check.tar.gz'])
        subprocess.call(["tar", "xf", "freeman_check.tar.gz"])
        subprocess.call(["rm", "freeman_check.tar.gz"])
        subprocess.call(["cp", "-R", "metamers/V1_norm_s4_gaussian", f"{met_dir_name}/"])
        subprocess.call(["cp", "-R", "freeman_check/windows/*", f"{windows_dir_name}/"])
        subprocess.call(["rm", "-r", "metamers/V1_norm_s4_gaussian"])
        subprocess.call(["rmdir", "metamers"])
        subprocess.call(["rm", "-r", "freeman_check"])
    subprocess.call(['chmod', '-R', '777', data_dir])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Download data associated with the foveated metamers project, to reproduce the results.")
    )
    parser.add_argument("target_dataset", choices=['synthesis_input', 'stimuli',
                                                   'behavioral_data', 'mcmc_fits',
                                                   'figure_input',
                                                   'freeman2011_check_input'
                                                   'freeman2011_check_output'],
                        help="Which dataset to download, see project README for details.")
    args = vars(parser.parse_args())
    main(**args)
