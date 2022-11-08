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


# dictionary of download urls
DOWNLOAD_URL = {
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
    'experiment_training': "https://osf.io/xy4ku/download",
    'mcmc_compare': "https://archive.nyu.edu/rest/bitstreams/{}/retrieve",
}

MCMC_COMPARE_FILES = ["mcmc_energy_met_compare_ic-loo.csv",
                      "mcmc_energy_met_downsample_compare_ic-loo.csv",
                      "mcmc_energy_met_downsample_partially-pooled-interactions.nc",
                      "mcmc_energy_met_downsample_unpooled.nc",
                      "mcmc_energy_met-nat_compare_ic-loo.csv",
                      "mcmc_energy_met-nat_partially-pooled-interactions.nc",
                      "mcmc_energy_met-nat_unpooled.nc",
                      "mcmc_energy_met_partially-pooled-interactions.nc",
                      "mcmc_energy_met_unpooled.nc",
                      "mcmc_energy_ref_compare_ic-loo.csv",
                      "mcmc_energy_ref-nat_compare_ic-loo.csv",
                      "mcmc_energy_ref-nat_partially-pooled-interactions.nc",
                      "mcmc_energy_ref-nat_unpooled.nc",
                      "mcmc_energy_ref_partially-pooled-interactions.nc",
                      "mcmc_energy_ref_unpooled.nc",
                      "mcmc_luminance_met_compare_ic-loo.csv",
                      "mcmc_luminance_met_partially-pooled-interactions.nc",
                      "mcmc_luminance_met_unpooled.nc",
                      "mcmc_luminance_ref_compare_ic-loo.csv",
                      "mcmc_luminance_ref_partially-pooled-interactions.nc",
                      "mcmc_luminance_ref_unpooled.nc"]


def check_checksum(path, checksum):
    with open(path, 'rb') as f:
        test_checksum = hashlib.blake2b(f.read())
    return test_checksum.hexdigest() == checksum


def main(target_dataset, skip_confirmation=False):
    """Download dataset from OpenNeuro or OSF.

    Parameters
    ----------
    target_dataset : {'synthesis_input', 'stimuli',
                      'behavioral_data', 'mcmc_fits',
                      'figure_input',
                      'freeman2011_check_input',
                      'freeman2011_check_output',
                      'mcmc_compare'}
        Which dataset to download (list of the above also allowable, in which
        case they'll be downloaded in the above order). See project README for
        more info.
    skip_confirmation : bool, optional
        If True, skip all confirmation checks and always download data.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)
    if op.split(config['DATA_DIR'])[-1].lower() != op.split(config['DATA_DIR'])[-1]:
        raise Exception(f"Name of your DATA_DIR must be all lowercase! But got {config['DATA_DIR']}")
    with open(CHECKSUM_PATH) as f:
        checksums = json.load(f)
    if not isinstance(target_dataset, list):
        target_dataset = [target_dataset]
    data_dir = config['DATA_DIR']
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using {data_dir} as data root directory.")
    targets = ['synthesis_input', 'stimuli', 'behavioral_data', 'mcmc_fits', 'figure_input',
               'freeman2011_check_input', 'freeman2011_check_output', 'experiment_training',
               'mcmc_compare']
    check_dirs = ['ref_images_preproc', 'stimuli', 'behavioral', 'mcmc', 'statistics',
                  'freeman_check/Freeman2011_metamers', 'freeman_check/windows', 'stimuli/training_noise',
                  # this isn't a dir, but a file
                  'mcmc/V1_norm_s6_gaussian/task-split_comp-met/task-split_comp-met_mcmc_compare_ic-loo.csv']
    sizes = ['176MB', '12GB', '2.6MB', '12GB', '580MB', '1MB', '60MB', '160MB', '24GB']
    if not skip_confirmation:
        for tar, check, size in zip(targets, check_dirs, sizes):
            yesno = 'y'
            if tar in target_dataset:
                if op.exists(op.join(data_dir, check)):
                    yesno = input(f"Previous data found for {tar}, do you wish to download that dataset anyway? [y/n] ")
                    while yesno not in ['y', 'n']:
                        print("Please enter y or n")
                        yesno = input(f"Previous data found for {tar}, do you wish to download that dataset anyway? [y/n] ")
                if yesno == 'n':
                    target_dataset.remove(tar)
                    continue
                yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
                while yesno not in ['y', 'n']:
                    print("Please enter y or n")
                    yesno = input(f"{tar} dataset will be approximately {size}, do you wish to download it? [y/n] ")
                if yesno == 'n':
                    target_dataset.remove(tar)
        if len(target_dataset) == 0:
            print("Exiting...")
            exit(0)
    else:
        print(f"Skipping all requests for confirmation and downloading {target_dataset} dataset(s)...")
    # dictionary mapping between the names used in the upload vs those in the actual data directory
    model_name_map = {'energy': 'V1_norm_s6_gaussian', 'luminance': 'RGC_norm_gaussian'}
    comp_name_map = lambda x: x.replace('-nat', '-natural').replace('_downsample', '-downsample-2')
    if 'synthesis_input' in target_dataset:
        print("Downloading synthesis input.")
        synth_checksum = False
        while not synth_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['synthesis_input']])
            synth_checksum = check_checksum('synthesis_input.tar.gz', checksums['synthesis_input.tar.gz'])
        subprocess.call(["tar", "xf", "synthesis_input.tar.gz"])
        # this file is unnecessary for the experiment
        subprocess.call(['rm', op.join('synthesis_input', 'metadata.json')])
        subprocess.call(["rsync", "-avPLuz", "synthesis_input/", f"{data_dir}/"])
        subprocess.call(["rm", "-r", "synthesis_input/"])
        subprocess.call(["rm", "synthesis_input.tar.gz"])
    if 'stimuli' in target_dataset:
        print("Downloading stimuli for all comparisons.")
        for name, url in DOWNLOAD_URL['stimuli'].items():
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
    if 'behavioral_data' in target_dataset:
        print("Downloading behavioral data for all comparisons.")
        behav_checksum = False
        while not behav_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['behavioral_data']])
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
    if 'mcmc_fits' in target_dataset:
        print("Downloading MCMC fits for all comparisons.")
        for name, url in DOWNLOAD_URL['mcmc_fits'].items():
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
    if 'figure_input' in target_dataset:
        print("Downloading figure input.")
        fig_checksum = False
        while not fig_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['figure_input']])
            fig_checksum = check_checksum('figure_input.tar.gz', checksums['figure_input.tar.gz'])
        subprocess.call(["tar", "xf", "figure_input.tar.gz"])
        for subdir in ['metamers', 'mad_images', 'synth_match_mse', 'statistics',
                       'windows_cache']:
            subprocess.call(["rsync", "-avPLuz", subdir, f"{data_dir}/"])
            subprocess.call(["rm", "-r", f"{subdir}/"])
        subprocess.call(["rm", "figure_input.tar.gz"])
    if 'freeman2011_check_input' in target_dataset:
        print("Downloading input for comparison against Freeman2011.")
        met_dir = op.join(data_dir, 'freeman_check', 'Freeman2011_metamers')
        os.makedirs(met_dir, exist_ok=True)
        ref_dir = op.join(data_dir, 'ref_images')
        os.makedirs(ref_dir, exist_ok=True)
        norm_stats_dir = op.join(data_dir, 'norm_stats')
        os.makedirs(norm_stats_dir, exist_ok=True)
        freeman_checksum = False
        while not freeman_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['freeman2011_check_input']])
            freeman_checksum = check_checksum('freeman_check_inputs.tar.gz', checksums['freeman_check_inputs.tar.gz'])
        subprocess.call(["tar", "xf", "freeman_check_inputs.tar.gz"])
        subprocess.call(["mv", "freeman_check_inputs/metamer1.png", f"{met_dir}/"])
        subprocess.call(["mv", "freeman_check_inputs/metamer2.png", f"{met_dir}/"])
        subprocess.call(["mv", "freeman_check_inputs/fountain_size-512,512.png", f"{ref_dir}/"])
        subprocess.call(["mv", "freeman_check_inputs/V1_texture_norm_stats.pt", f"{norm_stats_dir}/"])
        subprocess.call(["rm", "freeman_check_inputs.tar.gz"])
        subprocess.call(["rmdir", "freeman_check_inputs"])
    if 'freeman2011_check_output' in target_dataset:
        print("Downloading output for comparison against Freeman2011.")
        met_dir = op.join(data_dir, 'metamers')
        os.makedirs(met_dir, exist_ok=True)
        windows_dir = op.join(data_dir, 'freeman_check', 'windows')
        os.makedirs(windows_dir, exist_ok=True)
        freeman_checksum = False
        while not freeman_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['freeman2011_check_output']])
            freeman_checksum = check_checksum('freeman_check.tar.gz', checksums['freeman_check.tar.gz'])
        subprocess.call(["tar", "xf", "freeman_check.tar.gz"])
        subprocess.call(["rm", "freeman_check.tar.gz"])
        subprocess.call(["cp", "-R", "metamers/V1_norm_s4_gaussian", f"{met_dir}/"])
        subprocess.call(["cp", "-R", "freeman_check/windows/scaling-0.25/", f"{windows_dir}/"])
        subprocess.call(["cp", "-R", "freeman_check/windows/scaling-0.5/", f"{windows_dir}/"])
        subprocess.call(["rm", "-r", "metamers/V1_norm_s4_gaussian"])
        subprocess.call(["rmdir", "metamers"])
        subprocess.call(["rm", "-r", "freeman_check"])
    if 'experiment_training' in target_dataset:
        print("Downloading experiment training files.")
        training_checksum = False
        while not training_checksum:
            subprocess.call(["curl", "-O", "-J", "-L", DOWNLOAD_URL['experiment_training']])
            training_checksum = check_checksum('experiment_training.tar.gz',
                                               checksums['.tar.gz'])
        subprocess.call(["tar", "xf", "experiment_training.tar.gz"])
        subprocess.call(["rsync", "-avPLuz", 'stimuli', f"{data_dir}/"])
        subprocess.call(["rm", "-r", "stimuli/"])
        subprocess.call(["rm", "experiment_training.tar.gz"])
    if 'mcmc_compare' in target_dataset:
        print("Downloading files for MCMC model comparison.")
        for i, name in enumerate(MCMC_COMPARE_FILES):
            print(f'Downloading {name}')
            url = DOWNLOAD_URL['mcmc_compare'].format(129631+i)
            try:
                # if this regex works, then this is one of the .nc files
                # containing the mcmc fit
                download_model, download_comp, mcmc_model = re.findall('mcmc_([a-z]+)_([a-z-_]+)_([a-z-]+).nc', name)[0]
                outp_model = model_name_map[download_model]
                outp_comp = comp_name_map(download_comp)
                hyper = utils.get_mcmc_hyperparams({'mcmc_model': mcmc_model,
                                                    'model_name': outp_model, 'comp': outp_comp})
                outp = op.join(data_dir, 'mcmc', outp_model, f'task-split_comp-{outp_comp}',
                               f'task-split_comp-{outp_comp}_mcmc_{mcmc_model}_{hyper}_scaling-extended.nc')
            except IndexError:
                # then it's a mcmc compare csv, and we handle it differently
                download_model, download_comp, ic = re.findall('mcmc_([a-z]+)_([a-z-_]+)_compare_ic-([a-z]+).csv', name)[0]
                outp_model = model_name_map[download_model]
                outp_comp = comp_name_map(download_comp)
                outp = op.join(data_dir, 'mcmc', outp_model, f'task-split_comp-{outp_comp}',
                               f'task-split_comp-{outp_comp}_mcmc_compare_ic-{ic}.csv')
            mcmc_compare_checksum = False
            while not mcmc_compare_checksum:
                subprocess.call(["curl", "-k", "-L", url, '-o', name])
                mcmc_compare_checksum = check_checksum(name, checksums[name])
            os.makedirs(op.dirname(outp), exist_ok=True)
            subprocess.call(["mv", name, outp])
    # need to touch these files, in this order, to make sure that snakemake
    # doesn't get confused and thinks it needs to rerun things.
    paths_to_touch = []
    if op.exists(op.join(data_dir, 'metamers')):
        paths_to_touch.append('metamers')
    if op.exists(op.join(data_dir, 'stimuli')):
        paths_to_touch.append('stimuli')
    if op.exists(op.join(data_dir, 'synth_match_mse')):
        paths_to_touch.append('synth_match_mse')
    if op.exists(op.join(data_dir, 'mad_images')):
        paths_to_touch.append('mad_images')
    if op.exists(op.join(data_dir, 'behavioral')):
        paths_to_touch.append('behavioral')
    if op.exists(op.join(data_dir, 'mcmc')):
        paths_to_touch.append('mcmc')
    # The command we call just recursively touches everything in the specified
    # directory
    for path in paths_to_touch:
        subprocess.call(['find', f'{op.join(data_dir, path)}', '-type', 'f',
                         '-exec', 'touch', '{}', '+'])
    subprocess.call(['chmod', '-R', '777', data_dir])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Download data associated with the foveated metamers project, to reproduce the results.")
    )
    parser.add_argument("target_dataset", choices=['synthesis_input', 'stimuli',
                                                   'behavioral_data', 'mcmc_fits',
                                                   'figure_input',
                                                   'freeman2011_check_input',
                                                   'freeman2011_check_output',
                                                   'experiment_training',
                                                   'mcmc_compare'],
                        help="Which dataset to download, see project README for details.",
                        nargs='+')
    parser.add_argument('--skip-confirmation', '-s', action='store_true',
                        help="Skip all requests for confirmation and download data (intended for use in tests).")
    args = vars(parser.parse_args())
    main(**args)
