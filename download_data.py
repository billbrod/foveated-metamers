#!/usr/bin/env python3

import argparse
import subprocess
import os
import os.path as op
import yaml
from glob import glob
from foveated_metamers import utils


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
        'mcmc_energy_ref_partially-pooled.nc-nat': 'https://osf.io/4gtkh/download',
        'mcmc_energy_met_partially-pooled.nc': 'https://osf.io/n7wc3/download',
        'mcmc_energy_met_partially-pooled.nc-nat': 'https://osf.io/dx9ew/download',
        'mcmc_energy_met_downsample_partially-pooled.nc': 'https://osf.io/9wnvq/download',
    },
}


def main(target_dataset):
    """Download dataset from OpenNeuro or OSF.

    Parameters
    ----------
    target_dataset : {'synthesis_input', 'stimuli',
                      'behavioral_data', 'mcmc_fits'}
        Which dataset to download. See project README for more info.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)
    if op.split(config['DATA_DIR'])[-1].lower() != op.split(config['DATA_DIR'])[-1]:
        raise Exception(f"Name of your DATA_DIR must be all lowercase! But got {config['DATA_DIR']}")
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
        subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['synthesis_input']])
        subprocess.call(["tar", "xf", "synthesis_input.tar.gz"])
        # this is unnecessary for the experiment
        subprocess.call(['rm', op.join('synthesis_input', 'metadata.json')])
        subprocess.call(["rsync", "-avPLuz", "synthesis_input/", f"{data_dir}/"])
        subprocess.call(["rm", "-r", "synthesis_input/"])
        subprocess.call(["rm", "synthesis_input.tar.gz"])
    elif target_dataset == 'stimuli':
        print("Downloading stimuli for all comparisons.")
        for name, url in OSF_URL['stimuli']:
            print(f"Downloading {name}")
            download_model = re.findall('stimuli_([a-z]+)_', name)[0]
            output_model = model_name_map[download_model]
            subprocess.call(["curl", "-O", "-J", "-L", url])
            subprocess.call(["tar", "xf", f"{name}.tar.gz"])
            subprocess.call(["rsync", '-avPLUz', f'{name}/', f"{data_dir}/stimuli/{output_model}/"])
            subprocess.call(["rm", '-r', name])
            subprocess.call(["rm", f"{name}.tar.gz"])
    elif target_dataset == 'behavioral_data':
        print("Downloading behavioral data for all comparisons.")
        subprocess.call(["curl", "-O", "-J", "-L", OSF_URL['behavioral_data']])
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
        for name, url in OSF_URL['mcmc_fits']:
            print(f"Downloading {name}")
            subprocess.call(["curl", "-O", "-J", "-L", url])
            download_model, download_comp = re.findall('mcmc_([a-z]+)_comp-([a-z-_]+)_partially-pooled.nc', name)[0]
            outp_model = model_name_map[download_model]
            outp_comp = comp_name_map(download_comp)
            hyper = utils.get_mcmc_hyperparams({'mcmc_model': 'partially-pooled',
                                                'model_name': outp_model, 'comp': outp_comp})
            outp = op.join(data_dir, 'mcmc', outp_model, f'task-split_comp-{outp_comp}',
                           f'task-split_comp-{outp_comp}_mcmc_partially-pooled_{hyper}_scaling-extended.nc')
            os.makedirs(op.dirname(outp), exist_ok=True)
            subprocess.call(["mv", f, outp])
    subprocess.call(['chmod', '-R', '777', data_dir])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Download data associated with the foveated metamers project, to reproduce the results.")
    )
    parser.add_argument("target_dataset", choices=['synthesis_input', 'stimuli',
                                                   'behavioral_data', 'mcmc_fits'],
                        help="Which dataset to download, see project README for details.")
    args = vars(parser.parse_args())
    main(**args)
