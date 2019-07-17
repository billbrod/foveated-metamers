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
IMAGES = ['nuts', 'einstein', 'einstein_symmetric', 'einstein_constant']

def initial_metamer_inputs(wildcards):
    path_template = op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}',
                            'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-'
                            '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}.pt')
    # return [path_template.format(model_name=m, image_name=i, scaling=s, seed=0, learning_rate=lr,
    #                              min_ecc=.5, max_ecc=15, max_iter=20000, loss_thresh=1e-6) for
    #         m in MODELS for i in IMAGES for s in [.1, .2, .3, .4, .5, .6, .7, .8, .9] for lr in
    #         [.001, .1, 1, 10, 100]]
    return [path_template.format(model_name=m, image_name=i, scaling=s, seed=0, learning_rate=lr,
                                 min_ecc=.5, max_ecc=15, max_iter=100, loss_thresh=1e-6) for
            m in MODELS for i in IMAGES for s in [.5] for lr in [10]]


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


rule initial_metamers:
    input:
        initial_metamer_inputs,


rule create_metamers:
    input:
        # we use glob like this so we don't need to know what the extension is
        lambda wildcards: glob(op.join(config["DATA_DIR"], 'seed_images', wildcards.image_name+'.*'))[0]
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
