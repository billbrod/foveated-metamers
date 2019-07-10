import os.path as op
from glob import glob

configfile:
    "config.yml"
if not os.path.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
    # then we're on the cluster
    ON_CLUSTER = True
    shell.prefix(". /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; conda activate metamers; ")
else:
    ON_CLUSTER = False

MODELS = ['RGC', 'V1']
IMAGES = ['nuts', 'einstein']

def initial_metamer_inputs(wildcards):
    path_template = op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}',
                            'scaling-{scaling}', 'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-'
                            '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}.pt')
    return [path_template.format(model_name=m, image_name=i, scaling=s, seed=0, learning_rate=lr,
                                 min_ecc=.5, max_ecc=15, max_iter=1000, loss_thresh=1e-4) for
            m in MODELS for i in IMAGES for s in [.1, .2, .3, .4, .5, .6, .7, .8, .9] for lr in
            [.001, .1, 1, 10, 100]]


rule initial_metamers:
    input:
        initial_metamer_inputs,


rule create_metamers:
    input:
        lambda wildcards: glob(op.join(config["DATA_DIR"], 'seed_images', wildcards.image_name+'.*'))[0]
    output:
        op.join(config["DATA_DIR"], 'metamers', '{model_name}', '{image_name}', 'scaling-{scaling}',
                'seed-{seed}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-'
                '{loss_thresh}.pt')
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
