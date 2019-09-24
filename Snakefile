import os
import imageio
import time
import os.path as op
import numpy as np
from glob import glob
from plenoptic.simulate import pooling

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    ON_CLUSTER = True
    # need ffmpeg and our conda environment
    shell.prefix(". /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; conda activate metamers; "
                 "module load ffmpeg/intel/3.2.2; ")
else:
    ON_CLUSTER = False
wildcard_constraints:
    num="[0-9]+"

MODELS = ['RGC', 'V1', 'V1-norm', 'V1-norm-half_oct']
IMAGES = ['nuts', 'nuts_symmetric', 'nuts_constant', 'einstein', 'einstein_symmetric',
          'einstein_constant', 'AsianFusion-08', 'AirShow-12', 'ElFuenteDance-11',
          'Chimera1102347-03', 'CosmosLaundromat-08', 'checkerboard_period-64_size-256']
METAMER_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'metamers', '{model_name}', '{image_name}',
                                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-'
                                '{loss_fract}_cf-{coarse_to_fine}', 'seed-{seed}_init-{init_type}'
                                '_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_'
                                'thresh-{loss_thresh}_gpu-{gpu}_metamer.png')
REF_IMAGE_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'ref_images', '{image_name}.pgm')
SEEDS = {'sub-01': 0}


rule all_refs:
    input:
        [REF_IMAGE_TEMPLATE_PATH.format(image_name=i) for i in IMAGES]


rule yuv_to_mp4:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.yuv')
    output:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.mp4')
    log:
        op.join(config['NFLX_DIR'], 'logs', 'contents_org_yuv', '{video_name}.log')
    benchmark:
        op.join(config['NFLX_DIR'], 'logs', 'contents_org_yuv', '{video_name}_benchmark.txt')
    shell:
        # following this stackoverflow comment: https://stackoverflow.com/a/15780960/4659293
        "ffmpeg -f rawvideo -vcodec rawvideo -framerate 60 -s 1920x1080 -pixel_format yuv420p "
        "-i {input} -c:v libx264 -preset ultrafast -qp 0 {output} &> {log}"


rule mp4_to_pngs:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.mp4')
    output:
        [op.join(config['NFLX_DIR'], 'contents_org_yuv', '{{video_name}}', '{{video_name}}-{:02d}.png').format(i) for i in range(1, 13)]
    log:
        op.join(config['NFLX_DIR'], 'logs', 'contents_org_yuv', '{video_name}-png.log')
    benchmark:
        op.join(config['NFLX_DIR'], 'logs', 'contents_org_yuv', '{video_name}-png_benchmark.txt')
    params:
        out_name = lambda wildcards, output: output[0].replace('01', '%02d')
    shell:
        # following this stackoverlow: https://stackoverflow.com/a/10962408/4659293
        "ffmpeg -i {input} -r 1 {params.out_name} &> {log}"


rule png_to_pgm:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}', '{video_name}-{num}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{video_name}-{num}.pgm')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{video_name}-{num}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{video_name}-{num}_benchmark.txt')
    run:
        import imageio
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0], as_gray=True)
                imageio.imwrite(output[0], im)


rule pad_image:
    input:
        op.join(config["DATA_DIR"], 'ref_images', '{image_name}.{ext}')
    output:
        op.join(config["DATA_DIR"], 'ref_images', '{image_name}_{pad_mode}.{ext}')
    log:
        op.join(config["DATA_DIR"], 'logs', 'ref_images', '{image_name}_{pad_mode}-{ext}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'ref_images', '{image_name}_{pad_mode}-{ext}_benchmark.txt')
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.pad_image(input[0], wildcards.pad_mode, output[0])


rule generate_image:
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_type}_period-{period}_size-{size}.pgm')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}_benchmark.txt')
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.create_image(wildcards.image_type, int(wildcards.size), output[0],
                                         int(wildcards.period))


rule gen_norm_stats:
    input:
        config['TEXTURE_DIR']
    output:
        # here V1 and texture could be considered wildcards, but they're
        # the only we're doing this for now
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture_norm_stats-{num}.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture_norm_stats-{num}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture_norm_stats-{num}_benchmark.txt')
    params:
        index = lambda wildcards: (int(wildcards.num) * 100, (int(wildcards.num)+1) * 100)
    run:
        import plenoptic as po
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # scaling doesn't matter here
                v1 = po.simul.PrimaryVisualCortex(1, (512, 512), half_octave_pyramid=True)
                po.simul.non_linearities.generate_norm_stats(v1, input[0], output[0], (512, 512),
                                                             index=params.index)


# we need to generate the stats in blocks, and then want to re-combine them
rule combine_norm_stats:
    input:
        [op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture_norm_stats-{num}.pt').format(num=i)
         for i in range(9)]
    output:
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture_norm_stats.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture_norm_stats.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture_norm_stats_benchmark.txt')
    run:
        import torch
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                combined_stats = {}
                to_combine = [torch.load(i) for i in input]
                for k, v in to_combine[0].items():
                    if isinstance(v, dict):
                        d = {}
                        for l in v:
                            s = []
                            for i in to_combine:
                                s.append(i[k][l])
                            d[l] = torch.cat(s, 0)
                        combined_stats[k] = d
                    else:
                        s = []
                        for i in to_combine:
                            s.append(i[k])
                        combined_stats[k] = torch.cat(s, 0)
                torch.save(combined_stats, output[0])


rule cache_windows:
    output:
        op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_t-1.pt')
    log:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_t-1.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_t-1_benchmark.txt')
    run:
        import contextlib
        import plenoptic as po
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                img_size = [int(i) for i in wildcards.size.split(',')]
                po.simul.PoolingWindows(float(wildcards.scaling), img_size, float(wildcards.min_ecc),
                                        float(wildcards.max_ecc), cache_dir=op.dirname(output[0]),
                                        transition_region_width=1)


def get_norm_dict(wildcards):
    if 'norm' in wildcards.model_name and 'V1' in wildcards.model_name:
        return op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture_norm_stats.pt')
    else:
        return None


def get_windows(wildcards):
    r"""determine the cached window path for the specified model
    """
    window_template = op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}'
                              '_e0-{min_ecc:.03f}_em-{max_ecc:.01f}_t-1.pt')
    im = imageio.imread(REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.image_name))
    if wildcards.model_name == "RGC":
        size = ','.join([str(i) for i in im.shape])
        return window_template.format(scaling=wildcards.scaling, size=size,
                                      max_ecc=float(wildcards.max_ecc),
                                      min_ecc=float(wildcards.min_ecc))
    elif wildcards.model_name.startswith('V1'):
        windows = []
        # need them for every scale
        for i in range(4):
            output_size = ','.join([str(int(np.ceil(j / 2**i))) for j in im.shape])
            min_ecc, _ = pooling.calc_min_eccentricity(float(wildcards.scaling),
                                                       [np.ceil(j / 2**i) for j in im.shape],
                                                       float(wildcards.max_ecc))
            # don't do this for the lowest scale
            if i > 0 and min_ecc > float(wildcards.min_ecc):
                # this makes sure that whatever that third decimal place
                # is, we're always one above it. e.g., if min_ecc was
                # 1.3442, we want to use 1.345, and this will ensure
                # that
                min_ecc *= 1e3
                min_ecc -= min_ecc % 1
                min_ecc = (min_ecc+1) / 1e3
            else:
                min_ecc = float(wildcards.min_ecc)
            windows.append(window_template.format(scaling=wildcards.scaling, size=output_size,
                                                  max_ecc=float(wildcards.max_ecc),
                                                  min_ecc=min_ecc))
        return windows


def get_batches(wildcards):
    if len(wildcards.gpu.split(':')) > 1:
        return int(wildcards.gpu.split(':')[1])
    else:
        return 1


rule create_metamers:
    input:
        REF_IMAGE_TEMPLATE_PATH,
        get_windows,
    output:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        METAMER_TEMPLATE_PATH
    log:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}', 'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}'
                '_em-{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_gpu-{gpu}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}', 'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}'
                '_em-{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_benchmark.txt')
    resources:
        gpu = lambda wildcards: int(wildcards.gpu.split(':')[0]),
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        norm_dict = get_norm_dict,
        num_batches = get_batches,
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.create_metamers.main(wildcards.model_name, float(wildcards.scaling), input[0],
                                         int(wildcards.seed), float(wildcards.min_ecc),
                                         float(wildcards.max_ecc), float(wildcards.learning_rate),
                                         int(wildcards.max_iter), float(wildcards.loss_thresh),
                                         output[0], wildcards.init_type, resources.gpu>0,
                                         params.cache_dir, params.norm_dict, resources.gpu,
                                         wildcards.optimizer, float(wildcards.fract_removed),
                                         float(wildcards.loss_fract),
                                         float(wildcards.coarse_to_fine), int(params.num_batches))


# need to come up with a clever way to do this: either delete the ones
# we don't want or make this a function that only takes the ones we want
# or maybe grabs one each for max_iter, loss_thresh, learning_rate.
# Also need to think about how to handle max_ecc; it will be different
# if the images we use as inputs are different sizes. and init_type as
# well, V1/V2 will always be white, but RGC might also be gray or pink
def get_metamers_for_expt(wildcards):
    ims = ['nuts', 'einstein']
    images = [REF_IMAGE_TEMPLATE_PATH.format(image_name=i) for i in ims]
    return images+[METAMER_TEMPLATE_PATH.format(scaling=sc, seed=s, image_name=i, max_iter=1000,
                                                loss_thresh=1e-4, learning_rate=10, init_type='white',
                                                **wildcards)
                   for i in ims for sc in [.4, .5, .6] for s in [0, 1]]

rule collect_metamers:
    input:
        get_metamers_for_expt,
    output:
        # we collect across image_name and scaling, and don't care about
        # learning_rate, max_iter, loss_thresh
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}_'
                'stimuli.npy'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}_'
                'stimuli_description.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}'
                '_stimuli.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}'
                '_stimuli_benchmark.txt'),
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.collect_images(input, output[0])
                met.stimuli.create_metamer_df(input, [METAMER_TEMPLATE_PATH, REF_IMAGE_TEMPLATE_PATH],
                                              output[1])


rule generate_experiment_idx:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}_'
                'stimuli_description.csv'),
    output:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', '{subject}_e0-{min_ecc}_em-'
                '{max_ecc}_idx.npy'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', '{subject}_e0-{min_ecc}_em-'
                '{max_ecc}_idx.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', '{subject}_e0-{min_ecc}_em-'
                '{max_ecc}_idx_benchmark.txt'),
    params:
        seed = lambda wildcards: SEEDS[wildcards.subject]
    run:
        import foveated_metamers as met
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.generate_indices(pd.read_csv(input[0]), params.seed, output[0])
