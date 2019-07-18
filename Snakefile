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
          'einstein_constant']

def initial_metamer_inputs(wildcards):
    path_template = op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}',
                            'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-'
                            '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}.pt')
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


rule pad_image:
    input:
        op.join(config["DATA_DIR"], 'seed_images', '{image_name}.{ext}')
    output:
        op.join(config["DATA_DIR"], 'seed_images', '{image_name}_{pad_mode}.{ext}')
    log:
        op.join(config["DATA_DIR"], 'logs', 'seed_images', '{image_name}_{pad_mode}-{ext}-%j.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'seed_images', '{image_name}_{pad_mode}-{ext}_benchmark.txt')
    run:
        import imageio
        import warnings
        from skimage import util

        im = imageio.imread(input[0], as_gray=True)
        if im.max() > 1:
            warnings.warn("Assuming image range is (0, 255)")
            im /= 255
        pad_kwargs = {}
        if wildcards.pad_mode == 'constant':
            pad_kwargs['constant_values'] = .5
        im = util.pad(im, int(im.shape[0]/2), wildcards.pad_mode, **pad_kwargs)
        imageio.imwrite(output[0], im)


rule create_metamers:
    input:
        op.join(config["DATA_DIR"], 'seed_images', '{image_name}.pgm')
    output:
        op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}', 'scaling-{scaling}',
                'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-'
                '{loss_thresh}.pt'),
        op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}', 'scaling-{scaling}',
                'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-'
                '{loss_thresh}_metamer.png'),
        op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}', 'scaling-{scaling}',
                'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-'
                '{loss_thresh}_synthesis.mp4')
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


rule collect_metamers:
    input:
        # need to come up with a clever way to do this: either delete
        # the ones we don't want or make this a function that only takes
        # the ones we want or maybe grabs one each for max_iter,
        # loss_thresh, learning_rate
        op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}', 'scaling-{scaling}',
                'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-'
                '{loss_thresh}_metamer.png'),
    output:
        # we collect across image_name and scaling, and don't care about
        # learning_rate, max_iter, loss_thresh
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'seed-{seed}_e0-{min_ecc}_em-'
                '{max_ecc}_stimuli.npy'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'seed-{seed}_e0-{min_ecc}_'
                'em-{max_ecc}_stimuli-%j.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'seed-{seed}_e0-{min_ecc}_'
                'em-{max_ecc}_stimuli_benchmark.txt'),
    run:
        import imageio
        import numpy as np

        images = []
        for i in input:
            images.append(imageio.imread(i, as_gray=True))
        # want our images to be indexed along the first dimension
        images = np.einsum('ijk -> kij', np.dstack(images))
        imageio.imwrite(output[0], images)
