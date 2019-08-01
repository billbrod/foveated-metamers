import os.path as op
from glob import glob

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

MODELS = ['RGC', 'V1']
IMAGES = ['nuts', 'nuts_symmetric', 'nuts_constant', 'einstein', 'einstein_symmetric',
          'einstein_constant', 'AsianFusion-08', 'AirShow-12', 'ElFuenteDance-11',
          'Chimera1102347-03', 'CosmosLaundromat-08']
METAMER_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'metamers', '{model_name}', '{image_name}',
                                'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_'
                                'em-{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_metamer.png')
REF_IMAGE_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'ref_images', '{image_name}.pgm')
SEEDS = {'sub-01': 0}

def initial_metamer_inputs(wildcards):
    path_template = METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt')
    # return [path_template.format(model_name=m, image_name=i, scaling=s, seed=0, learning_rate=lr,
    #                              min_ecc=.5, max_ecc=15, max_iter=20000, loss_thresh=1e-6) for
    #         m in MODELS for i in IMAGES for s in [.1, .2, .3, .4, .5, .6, .7, .8, .9] for lr in
    #         [.1, 1, 10]]
    metamers = [path_template.format(model_name='V1', image_name=i, scaling=s, seed=0,
                                     learning_rate=lr,min_ecc=.5, max_iter=5000, loss_thresh=1e-6,
                                     # want different max eccentricity
                                     # based on whether we've padded the
                                     # image (and thus doubled its
                                     # width) or not
                                     max_ecc={True: 30, False: 15}['_' in i])
                for i in IMAGES for s in [.4, .5, .6] for lr in [1, 10]]
    metamers.extend([path_template.format(model_name='RGC', image_name=i, scaling=s, seed=0,
                                     learning_rate=lr,min_ecc=.5, max_iter=5000, loss_thresh=1e-6,
                                     # want different max eccentricity
                                     # based on whether we've padded the
                                     # image (and thus doubled its
                                     # width) or not
                                     max_ecc={True: 30, False: 15}['_' in i])
                for i in IMAGES for s in [.2, .3, .4] for lr in [1, 10]])
    return metamers


rule initial_metamers:
    input:
        initial_metamer_inputs,


rule all_refs:
    input:
        [REF_IMAGE_TEMPLATE_PATH.format(image_name=i) for i in IMAGES]


rule yuv_to_mp4:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.yuv')
    output:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.mp4')
    shell:
        # following this stackoverflow comment: https://stackoverflow.com/a/15780960/4659293
        "ffmpeg -f rawvideo -vcodec rawvideo -framerate 60 -s 1920x1080 -pixel_format yuv420p "
        "-i {input} -c:v libx264 -preset ultrafast -qp 0 {output}"


rule mp4_to_pngs:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}.mp4')
    output:
        [op.join(config['NFLX_DIR'], 'contents_org_yuv', '{{video_name}}', '{{video_name}}-{:02d}.png').format(i) for i in range(1, 13)]
    params:
        out_name = lambda wildcards, output: output[0].replace('01', '%02d')
    shell:
        # following this stackoverlow: https://stackoverflow.com/a/10962408/4659293
        "ffmpeg -i {input} -r 1 {params.out_name}"


rule png_to_pgm:
    input:
        op.join(config['NFLX_DIR'], 'contents_org_yuv', '{video_name}', '{video_name}-{num}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{video_name}-{num}.pgm')
    run:
        import imageio
        im = imageio.imread(input[0], as_gray=True)
        imageio.imwrite(output[0], im)


rule pad_image:
    input:
        op.join(config["DATA_DIR"], 'ref_images', '{image_name}.{ext}')
    output:
        op.join(config["DATA_DIR"], 'ref_images', '{image_name}_{pad_mode}.{ext}')
    log:
        op.join(config["DATA_DIR"], 'logs', 'ref_images', '{image_name}_{pad_mode}-{ext}-%j.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'ref_images', '{image_name}_{pad_mode}-{ext}_benchmark.txt')
    run:
        import foveated_metamers as met
        met.stimuli.pad_image(input[0], wildcards.pad_mode, output[0])


rule create_metamers:
    input:
        REF_IMAGE_TEMPLATE_PATH
    output:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        METAMER_TEMPLATE_PATH
    log:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_'
                'iter-{max_iter}_thresh-{loss_thresh}-%j.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_'
                'iter-{max_iter}_thresh-{loss_thresh}_benchmark.txt')
    run:
        import foveated_metamers as met
        if ON_CLUSTER:
            log_file = None
        else:
            log_file = log[0]
        met.create_metamers.main(wildcards.model_name, float(wildcards.scaling), input[0],
                                 int(wildcards.seed), float(wildcards.min_ecc),
                                 float(wildcards.max_ecc), float(wildcards.learning_rate),
                                 int(wildcards.max_iter), float(wildcards.loss_thresh), log_file,
                                 output[0])


# need to come up with a clever way to do this: either delete the ones
# we don't want or make this a function that only takes the ones we want
# or maybe grabs one each for max_iter, loss_thresh, learning_rate.
# Also need to think about how to handle max_ecc; it will be different
# if the images we use as inputs are different sizes.
def get_metamers_for_expt(wildcards):
    ims = ['nuts', 'einstein']
    images = [REF_IMAGE_TEMPLATE_PATH.format(image_name=i) for i in ims]
    return images+[METAMER_TEMPLATE_PATH.format(scaling=sc, seed=s, image_name=i, max_iter=1000,
                                                loss_thresh=1e-4, learning_rate=10, **wildcards)
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
                '_stimuli-%j.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'e0-{min_ecc}_em-{max_ecc}'
                '_stimuli_benchmark.txt'),
    run:
        import foveated_metamers as met
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
                '{max_ecc}_idx-%j.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', '{subject}_e0-{min_ecc}_em-'
                '{max_ecc}_idx_benchmark.txt'),
    params:
        seed = lambda wildcards: SEEDS[wildcards.subject]
    run:
        import foveated_metamers as met
        import pandas as pd
        met.stimuli.generate_indices(pd.read_csv(input[0]), params.seed, output[0])
