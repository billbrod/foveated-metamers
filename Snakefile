# required to fix this strange problem:
# https://stackoverflow.com/questions/64797838/libgcc-s-so-1-must-be-installed-for-pthread-cancel-to-work
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
import math
import itertools
import re
import imageio
import time
import os.path as op
from glob import glob
import numpy as np
from foveated_metamers import utils
import numpyro
import multiprocessing


configfile:
    # config is in the same directory as this file
    op.join(op.dirname(op.realpath(workflow.snakefile)), 'config.yml')
if not op.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
# for some reason, I can't get os.system('module list') to work
# properly on NYU Greene (it always returns a non-zero exit
# code). However, they do have the CLUSTER environmental variable
# defined, so we can use that
if os.system("module list") == 0 or os.environ.get("CLUSTER", None):
    # then we're on the cluster
    ON_CLUSTER = True
    numpyro.set_host_device_count(multiprocessing.cpu_count())
else:
    ON_CLUSTER = False
    numpyro.set_host_device_count(4)
wildcard_constraints:
    num="[0-9]+",
    pad_mode="constant|symmetric",
    period="[0-9]+",
    size="[0-9,]+",
    bits="[0-9]+",
    img_preproc="full|degamma|range-[,.0-9]+|downsample-[0-9.]+_range-[,.0-9]+",
    # einstein for testing setup, fountain for comparing with Freeman and Simoncelli, 2011 metamers
    preproc_image_name="|".join([im+'_?[a-z]*' for im in config['IMAGE_NAME']['ref_image']])+"|einstein|fountain",
    preproc="|_degamma|degamma",
    gpu="0|1",
    sess_num="|".join([f'{i:02d}' for i in config['PSYCHOPHYSICS']['SESSIONS']]),
    run_num="|".join([f'{i:02d}' for i in config['PSYCHOPHYSICS']['RUNS']]),
    comp='met|ref|met-downsample-2|met-natural|ref-natural',
    save_all='|_saveall',
    gammacorrected='|_gamma-corrected',
    plot_focus='|_focus-subject|_focus-image',
    ecc_mask="|_eccmask-[0-9]+",
    logscale="log|linear",
    mcmc_model="partially-pooled|unpooled|partially-pooled-interactions-[.0-9]+|partially-pooled-interactions",
    fixation_cross="cross|nocross",
    cutout="cutout|nocutout|cutout_V1_natural-seed|cutout_RGC_natural-seed|nocutout_small|cutout_downsample",
    context="paper|poster",
    mcmc_plot_type="performance|params-(linear|log)-(none|lines|ci)",
    # don't capture experiment-mse for x, but capture anything else, because
    # experiment-mse is a different rule
    x="(?!experiment-mse)(.*)",
    mse="experiment_mse|full_image_mse",
    seed="[0-9]+",
    line="offset|nooffset",
    gif_direction="forward|reverse",
    scaling_extended="|_scaling-extended",
    osf_model_name='energy|luminance',
    # these are the only possible schematics
    schematic_name='|'.join([f.replace(op.join('reports', 'figures', ''), '').replace('.svg', '')
                             for f in glob(op.join('reports', 'figures', '*svg'))]),
    yaxis_dbl='single|double',
    t_width='[0-9.]+',
    window_type='cosine|gaussian',
ruleorder:
    collect_training_metamers > collect_training_noise > collect_metamers > demosaic_image > preproc_image > crop_image > generate_image > degamma_image > create_metamers > mcmc_compare_plot > mcmc_plots > embed_bitmaps_into_figure > compose_figures > copy_schematic

config['INKSCAPE_PREF_FILE'] = op.expanduser(config['INKSCAPE_PREF_FILE'])
config['FREEMAN_METAMER_PATH'] = op.expanduser(config['FREEMAN_METAMER_PATH'])
config['MATLABPYRTOOLS_PATH'] = op.expanduser(config['MATLABPYRTOOLS_PATH'])
LINEAR_IMAGES = config['IMAGE_NAME']['ref_image']
MODELS = [config[i]['model_name'] for i in ['RGC', 'V1']]
IMAGES = config['DEFAULT_METAMERS']['image_name']
# this is ugly, but it's easiest way to just replace the one format
# target while leaving the others alone
DATA_DIR = config['DATA_DIR']
if not DATA_DIR.endswith('/'):
    DATA_DIR += '/'
REF_IMAGE_TEMPLATE_PATH = config['REF_IMAGE_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR)
# the regex here removes all string formatting codes from the string,
# since Snakemake doesn't like them
METAMER_TEMPLATE_PATH = re.sub(":.*?}", "}", config['METAMER_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR))
MAD_TEMPLATE_PATH = re.sub(":.*?}", "}", config['MAD_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR))
METAMER_LOG_PATH = METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'logs/metamers/{model_name}').replace('_metamer.png', '.log')
MAD_LOG_PATH = MAD_TEMPLATE_PATH.replace('mad_images/{model_name}', 'logs/mad_images/{model_name}').replace('_mad.png', '.log')
CONTINUE_TEMPLATE_PATH = (METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'metamers_continue/{model_name}')
                          .replace("{clamp_each_iter}/", "{clamp_each_iter}/attempt-{num}_iter-{extra_iter}"))
CONTINUE_LOG_PATH = CONTINUE_TEMPLATE_PATH.replace('metamers_continue/{model_name}', 'logs/metamers_continue/{model_name}').replace('_metamer.png', '.log')
TEXTURE_DIR = config['TEXTURE_DIR']
if TEXTURE_DIR.endswith(os.sep) or TEXTURE_DIR.endswith('/'):
    TEXTURE_DIR = TEXTURE_DIR[:-1]
if op.exists(TEXTURE_DIR) and len(os.listdir(TEXTURE_DIR)) <= 800 and 'textures-subset-for-testing' not in TEXTURE_DIR:
    raise Exception(f"TEXTURE_DIR {TEXTURE_DIR} is incomplete!")

BEHAVIORAL_DATA_DATES = {
    'V1_norm_s6_gaussian': {
        'ref': {
            'sub-00': {'sess-00': '2021-Mar-23', 'sess-01': '2021-Mar-24', 'sess-02': '2021-Mar-24'},
            'sub-01': {'sess-00': '2021-Mar-30', 'sess-01': '2021-Mar-30', 'sess-02': '2021-Apr-01'},
            'sub-02': {'sess-00': '2021-Apr-14', 'sess-01': '2021-Apr-16', 'sess-02': '2021-Apr-21'},
            'sub-03': {'sess-00': '2021-Apr-02', 'sess-01': '2021-Apr-07', 'sess-02': '2021-Apr-09'},
            'sub-04': {'sess-00': '2021-Apr-05', 'sess-01': '2021-Apr-06', 'sess-02': '2021-Apr-12'},
            'sub-05': {'sess-00': '2021-Apr-30', 'sess-01': '2021-May-05', 'sess-02': '2021-May-11'},
            'sub-06': {'sess-00': '2021-May-14', 'sess-01': '2021-May-17', 'sess-02': '2021-May-18'},
            'sub-07': {'sess-00': '2021-Apr-23', 'sess-01': '2021-Apr-26', 'sess-02': '2021-Apr-28'},
        },
        'met': {
            'sub-00': {'sess-00': '2021-Apr-05', 'sess-01': '2021-Apr-07', 'sess-02': '2021-Apr-08'},
            'sub-01': {'sess-00': '2021-Apr-28', 'sess-01': '2021-Apr-28', 'sess-02': '2021-Apr-28'},
            'sub-02': {'sess-00': '2021-May-03', 'sess-01': '2021-May-04', 'sess-02': '2021-May-05'},
            'sub-03': {'sess-00': '2021-May-03', 'sess-01': '2021-May-04', 'sess-02': '2021-May-12'},
            'sub-04': {'sess-00': '2021-May-11', 'sess-01': '2021-May-12', 'sess-02': '2021-May-18'},
            'sub-05': {'sess-00': '2021-May-12', 'sess-01': '2021-May-17', 'sess-02': '2021-May-18'},
            'sub-06': {'sess-00': '2021-May-25', 'sess-01': '2021-May-26', 'sess-02': '2021-May-28'},
            'sub-07': {'sess-00': '2021-Apr-30', 'sess-01': '2021-Apr-30', 'sess-02': '2021-May-03'},
        },
        'met-downsample-2': {
            'sub-00': {'sess-00': '2021-May-04', 'sess-01': '2021-May-05', 'sess-02': '2021-May-06'},
        },
        'met-natural': {
            'sub-00': {'sess-00': '2021-Jul-14', 'sess-01': '2021-Jul-14', 'sess-02': '2021-Jul-15'},
        },
        'ref-natural': {
            'sub-00': {'sess-00': '2021-Jul-07', 'sess-01': '2021-Jul-13', 'sess-02': '2021-Jul-13'},
        },
    },
    'RGC_norm_gaussian': {
        'ref': {
            'sub-00': {'sess-00': '2021-Apr-02', 'sess-01': '2021-Apr-06', 'sess-02': '2021-Apr-06'},
            'sub-01': {'sess-00': '2021-Apr-16', 'sess-01': '2021-Apr-16', 'sess-02': '2021-Apr-16'},
            'sub-02': {'sess-00': '2021-Apr-07', 'sess-01': '2021-Apr-08', 'sess-02': '2021-Apr-12'},
            'sub-03': {'sess-00': '2021-Apr-27', 'sess-01': '2021-Apr-28', 'sess-02': '2021-Apr-30'},
            'sub-04': {'sess-00': '2021-Apr-21', 'sess-01': '2021-Apr-23', 'sess-02': '2021-Apr-27'},
            'sub-05': {'sess-00': '2021-Apr-14', 'sess-01': '2021-Apr-23', 'sess-02': '2021-Apr-27'},
            'sub-06': {'sess-00': '2021-Apr-14', 'sess-01': '2021-May-05', 'sess-02': '2021-May-12'},
            'sub-07': {'sess-00': '2021-Apr-14', 'sess-01': '2021-Apr-16', 'sess-02': '2021-Apr-21'},
        },
        'met': {
            'sub-00': {'sess-00': '2021-Apr-09'},
        }
    }
}

rule get_all_mcmc_plots:
    input:
        [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                 'task-split_comp-{comp}_mcmc_{mcmc_model}_{hyper}_{plot_name}.png').format(
                     mcmc_model=mc, model_name=m, comp=c, plot_name=p,
                     hyper=utils.get_mcmc_hyperparams({}, mcmc_model=mc, model_name=m, comp=c))
         for mc in ['unpooled', 'partially-pooled', 'partially-pooled-interactions']
         for m in MODELS
         for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'met-downsample-2', 'ref-natural']}.get(m, ['met', 'ref'])
         for p in ['diagnostics', 'performance', 'post-pred-check', 'psychophysical-params', 'grouplevel', 'param-corr_image',
                   'param-corr_subject'] + {'partially-pooled-interactions': ['metaparams', 'interaction-params', 'param-avg-method', 'params'],
                                            'partially-pooled': ['metaparams', 'param-avg-method', 'param-corr_a0', 'param-corr_s0', 'params']}.get(mc, [])
        ]


# quick rule to check that there are GPUs available and the environment
# has been set up correctly.
rule test_setup_all:
    input:
        [op.join(config['DATA_DIR'], 'test_setup', '{}_gpu-{}', 'einstein').format(m, g) for g in [0, 1]
         for m in MODELS],


rule test_setup:
    input:
        lambda wildcards: utils.generate_metamer_paths(model_name=wildcards.model_name,
                                                       image_name='einstein_degamma_size-256,256',
                                                       scaling=1.0,
                                                       gpu=wildcards.gpu_num,
                                                       max_iter=210,
                                                       seed=0)[0].replace('metamer.png', 'synthesis-1.png')
    output:
        directory(op.join(config['DATA_DIR'], 'test_setup', '{model_name}_gpu-{gpu_num}', 'einstein')),
    log:
        op.join(config['DATA_DIR'], 'logs', 'test_setup', '{model_name}_gpu-{gpu_num}', 'einstein.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'test_setup', '{model_name}_gpu-{gpu_num}', 'einstein_benchmark.txt')
    run:
        import contextlib
        import shutil
        import os.path as op
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print("Copying outputs from %s to %s" % (op.dirname(input[0]), output[0]))
                # Copy over only those files that were generated by the call
                # with the correct gpu_num wildcard, ignore the others
                # (wildcards.gpu_num can only take values 0 or 1)
                ignore_gpu = {0: '*_gpu-1*', 1: '*_gpu-0*'}[int(wildcards.gpu_num)]
                shutil.copytree(op.dirname(input[0]), output[0], ignore=shutil.ignore_patterns(ignore_gpu))


rule all_refs:
    input:
        [utils.get_ref_image_full_path(utils.get_gamma_corrected_ref_image(i))
         for i in IMAGES],
        [utils.get_ref_image_full_path(i) for i in IMAGES],


# for this project, our input images are linear images, but if you want
# to run this process on standard images, they have had a gamma
# correction applied to them. since we'll be displaying them on a linear
# display, we want to remove this correction (see
# https://www.cambridgeincolour.com/tutorials/gamma-correction.htm for
# an explanation)
rule degamma_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}-degamma-{bits}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}-degamma-{bits}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}-degamma-{bits}'
                '_benchmark.txt')
    run:
        import imageio
        import contextlib
        import foveated_metamers as fov
        from skimage import color
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                # when loaded in, the range of this will be 0 to 255, we
                # want to convert it to 0 to 1
                im = fov.utils.convert_im_to_float(im)
                # convert to grayscale
                im = color.rgb2gray(im)
                # 1/2.2 is the standard encoding gamma for jpegs, so we
                # raise this to its reciprocal, 2.2, in order to reverse
                # it
                im = im**2.2
                dtype = eval('np.uint%s' % wildcards.bits)
                imageio.imwrite(output[0], fov.utils.convert_im_to_int(im, dtype))


rule demosaic_image:
    input:
        op.join(config['DATA_DIR'], 'raw_images', '{image_name}.NEF')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}.tiff')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}_benchmark.txt')
    params:
        tiff_file = lambda wildcards, input: input[0].replace('NEF', 'tiff')
    shell:
        "dcraw -v -4 -q 3 -T {input}; "
        "mv {params.tiff_file} {output}"


rule crop_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}.tiff')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}_size-{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}_size-{size}_benchmark.txt')
    run:
        import imageio
        import contextlib
        from skimage import color
        import foveated_metamers as fov
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                curr_shape = np.array(im.shape)[:2]
                target_shape = [int(i) for i in wildcards.size.split(',')]
                print(curr_shape, target_shape)
                if len(target_shape) == 1:
                    target_shape = 2* target_shape
                target_shape = np.array(target_shape)
                crop_amt = curr_shape - target_shape
                # this is ugly, but I can't come up with an easier way to make
                # sure that we skip a dimension if crop_amt is 0 for it
                cropped_im = im
                for i, c in enumerate(crop_amt):
                    if c == 0:
                        continue
                    else:
                        if i == 0:
                            cropped_im = cropped_im[c//2:-c//2]
                        elif i == 1:
                            cropped_im = cropped_im[:, c//2:-c//2]
                        else:
                            raise Exception("Can only crop up to two dimensions!")
                cropped_im = color.rgb2gray(cropped_im)
                imageio.imwrite(output[0], fov.utils.convert_im_to_int(cropped_im, np.uint16))
                # tiffs can't be read in using the as_gray arg, so we
                # save it as a png, and then read it back in as_gray and
                # save it back out
                cropped_im = imageio.imread(output[0], as_gray=True)
                imageio.imwrite(output[0], cropped_im.astype(np.uint16))


rule preproc_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images', '{preproc_image_name}_size-{size}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images_preproc', '{preproc_image_name}_{img_preproc}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_image_preproc',
                '{preproc_image_name}_{img_preproc}_size-{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_image_preproc',
                '{preproc_image_name}_{img_preproc}_size-{size}_benchmark.txt')
    run:
        import imageio
        import contextlib
        import numpy as np
        from skimage import transform
        import foveated_metamers as fov
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                dtype = im.dtype
                im = np.array(im, dtype=np.float32)
                print("Original image has dtype %s" % dtype)
                if 'full' in wildcards.img_preproc:
                    print("Setting image to use full dynamic range")
                    # set the minimum value to 0
                    im = im - im.min()
                    # set the maximum value to 1
                    im = im / im.max()
                elif 'range' in wildcards.img_preproc:
                    a, b = re.findall('range-([.0-9]+),([.0-9]+)', wildcards.img_preproc)[0]
                    a, b = float(a), float(b)
                    print(f"Setting range to {a:02f}, {b:02f}")
                    if a > b:
                        raise Exception("For consistency, with range-a,b preprocessing, b must be"
                                        " greater than a, but got {a} > {b}!")
                    # set the minimum value to 0
                    im = im - im.min()
                    # set the maximum value to 1
                    im = im / im.max()
                    # and then rescale
                    im = im * (b - a) + a
                else:
                    print("Image will *not* use full dynamic range")
                    im = im / np.iinfo(dtype).max
                if 'downsample' in wildcards.img_preproc:
                    downscale = float(re.findall('downsample-([.0-9]+)_', wildcards.img_preproc)[0])
                    im = transform.pyramid_reduce(im, downscale)
                # always save it as 16 bit
                print("Saving as 16 bit")
                im = fov.utils.convert_im_to_int(im, np.uint16)
                imageio.imwrite(output[0], im)


rule gamma_correct_ref_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images{ref_suffix}', '{preproc_image_name}_{image_suffix}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images{ref_suffix}', '{preproc_image_name}_gamma-corrected_{image_suffix}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images{ref_suffix}', '{preproc_image_name}_gamma-corrected_{image_suffix}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images{ref_suffix}', '{preproc_image_name}_gamma-corrected_{image_suffix}_benchmark.txt')
    run:
        import imageio
        import contextlib
        import numpy as np
        import foveated_metamers as fov
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                dtype = im.dtype
                im = np.array(im, dtype=np.float32)
                print("Original image has dtype %s" % dtype)
                im = im / np.iinfo(dtype).max
                print("Raising image to 1/2.2, to gamma correct it")
                im = im ** (1/2.2)
                # always save it as 16 bit
                print("Saving as 16 bit")
                im = fov.utils.convert_im_to_int(im, np.uint16)
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
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.stimuli.pad_image(input[0], wildcards.pad_mode, output[0])


rule generate_image:
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_type}_period-{period}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.stimuli.create_image(wildcards.image_type, int(wildcards.size), output[0],
                                         int(wildcards.period))

rule preproc_textures:
    input:
        TEXTURE_DIR
    output:
        directory(TEXTURE_DIR + "_{preproc}")
    log:
        op.join(config['DATA_DIR'], 'logs', '{preproc}_textures.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', '{preproc}_textures_benchmark.txt')
    run:
        import imageio
        import contextlib
        from glob import glob
        import os.path as op
        import os
        from skimage import color
        import foveated_metamers as fov
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                os.makedirs(output[0])
                for i in glob(op.join(input[0], '*.jpg')):
                    im = imageio.imread(i)
                    im = fov.utils.convert_im_to_float(im)
                    if im.ndim == 3:
                        # then it's a color image, and we need to make it grayscale
                        im = color.rgb2gray(im)
                    if 'degamma' in wildcards.preproc:
                        # 1/2.2 is the standard encoding gamma for jpegs, so we
                        # raise this to its reciprocal, 2.2, in order to reverse
                        # it
                        im = im ** 2.2
                    # save as a 16 bit png
                    im = fov.utils.convert_im_to_int(im, np.uint16)
                    imageio.imwrite(op.join(output[0], op.split(i)[-1].replace('jpg', 'png')), im)


rule gen_norm_stats:
    input:
        TEXTURE_DIR + "{preproc}"
    output:
        # here V1 and texture could be considered wildcards, but they're
        # the only we're doing this for now
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture{preproc}_'
                'norm_stats-{num}.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats-{num}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats-{num}_benchmark.txt')
    params:
        index = lambda wildcards: (int(wildcards.num) * 100, (int(wildcards.num)+1) * 100)
    run:
        import contextlib
        import sys
        sys.path.append(op.join(op.dirname(op.realpath(__file__)), 'extra_packages'))
        import plenoptic_part as pop
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # scaling doesn't matter here
                v1 = pop.PooledV1(1, (512, 512), num_scales=6)
                pop.optim.generate_norm_stats(v1, input[0], output[0], (512, 512),
                                             index=params.index)


# we need to generate the stats in blocks, and then want to re-combine them
rule combine_norm_stats:
    input:
        lambda wildcards: [op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture'
                                   '{preproc}_norm_stats-{num}.pt').format(num=i, **wildcards)
                           for i in range(math.ceil(len(os.listdir(TEXTURE_DIR))/100))]
    output:
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture{preproc}_norm_stats.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats_benchmark.txt')
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


def get_mem_estimate(wildcards, partition=None):
    r"""estimate the amount of memory that this will need, in GB
    """
    try:
        if 'size-2048,2600' in wildcards.image_name:
            if 'gaussian' in wildcards.model_name or 'Obs' in wildcards.model_name:
                if 'V1' in wildcards.model_name or 'Obs' in wildcards.model_name:
                    if float(wildcards.scaling) < .1:
                        mem = 128
                    else:
                        mem = 64
                elif 'RGC' in wildcards.model_name:
                    # this is an approximation of the size of their windows,
                    # and if you have at least 3 times this memory, you're
                    # good. double-check this value -- the 1.36 is for
                    # converting form 2048,3528 (which the numbers came
                    # from) to 2048,2600 (which has 1.36x fewer pixels)
                    window_size = 1.17430726 / (1.36*float(wildcards.scaling))
                    mem = int(5 * window_size)
                    # running out of memory for larger scaling values, so let's
                    # never request less than 32 GB
                    mem = max(mem, 32)
            elif 'cosine' in wildcards.model_name:
                if 'V1' in wildcards.model_name:
                    # most it will need is 32 GB
                    mem = 32
                elif 'RGC' in wildcards.model_name:
                    # this is an approximation of the size of their windows,
                    # and if you have at least 3 times this memory, you're
                    # good
                    window_size = 0.49238059 / float(wildcards.scaling)
                    mem = int(5 * window_size)
            else:
                mem = 32
        else:
            # don't have a good estimate for these
            mem = 16
    except AttributeError:
        # then we don't have a image_name wildcard (and thus this is
        # being called by cache_windows)
        if wildcards.size == '2048,2600':
            if wildcards.window_type == 'gaussian':
                # this is an approximation of the size of their windows,
                # and if you have at least 3 times this memory, you're
                # good. double-check this value -- the 1.36 is for
                # converting form 2048,3528 (which the numbers came
                # from) to 2048,2600 (which has 1.36x fewer pixels)
                window_size = 1.17430726 / (1.36*float(wildcards.scaling))
                mem = int(5 * window_size)
            elif wildcards.window_type == 'cosine':
                # this is an approximation of the size of their windows,
                # and if you have at least 3 times this memory, you're
                # good
                window_size = 0.49238059 / float(wildcards.scaling)
                mem = int(5 * window_size)
        else:
            # don't have a good estimate here
            mem = 16
    try:
        if wildcards.save_all:
            # for this estimate, RGC with scaling .095 went from 36GB requested
            # to about 54GB used when stored iterations went from 100 to 1000.
            # that's 1.5x higher, and we add a bit of a buffer. also, don't
            # want to reduce memory estimate
            mem_factor = max((int(wildcards.max_iter) / 100) * (1.7/10), 1)
            mem *= mem_factor
    except AttributeError:
        # then we're missing either the save_all or max_iter wildcard, in which
        # case this is probably cache_windows and the above doesn't matter
        pass
    mem = int(np.ceil(mem))
    if partition == 'rusty':
        if int(wildcards.gpu) == 0:
            # in this case, we *do not* want to specify memory (we'll get the
            # whole node allocated but slurm could still kill the job if we go
            # over requested memory). setting mem=0 requests all memory on node
            mem = '0'
        else:
            # we'll be plugging this right into the mem request to slurm, so it
            # needs to be exactly correct
            mem = f"{mem}GB"
    return mem


rule cache_windows:
    output:
        op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}.pt')
    log:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.benchmark.txt')
    resources:
        mem = get_mem_estimate,
    run:
        import contextlib
        import plenoptic as po
        import sys
        sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages', 'pooling-windows'))
        import pooling
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                img_size = [int(i) for i in wildcards.size.split(',')]
                kwargs = {}
                if wildcards.window_type == 'cosine':
                    t_width = float(wildcards.t_width)
                    std_dev = None
                    min_ecc = float(wildcards.min_ecc)
                elif wildcards.window_type == 'gaussian':
                    std_dev = float(wildcards.t_width)
                    t_width = None
                    min_ecc = float(wildcards.min_ecc)
                pooling.PoolingWindows(float(wildcards.scaling), img_size, min_ecc,
                                       float(wildcards.max_ecc), cache_dir=op.dirname(output[0]),
                                       transition_region_width=t_width, std_dev=std_dev,
                                       window_type=wildcards.window_type, **kwargs)


def get_norm_dict(wildcards):
    if 'norm' in wildcards.model_name or wildcards.model_name.startswith('Obs'):
        preproc = ''
        # lienar images should also use the degamma'd textures
        if 'degamma' in wildcards.image_name or any([i in wildcards.image_name for i in LINEAR_IMAGES]):
            preproc += '_degamma'
        return op.join(config['DATA_DIR'], 'norm_stats', f'V1_texture{preproc}'
                       '_norm_stats.pt')
    else:
        return []


def get_windows(wildcards):
    r"""determine the cached window path for the specified model
    """
    wildcards = dict(wildcards)
    window_template = op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}'
                              '_e0-{min_ecc:.03f}_em-{max_ecc:.01f}_w-{t_width}_{window_type}.pt')
    try:
        if 'size-' in wildcards['image_name']:
            im_shape = wildcards['image_name'][wildcards['image_name'].index('size-') + len('size-'):]
            im_shape = im_shape.replace('.png', '')
            im_shape = [int(i) for i in im_shape.split(',')]
        else:
            try:
                im = imageio.imread(REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards['image_name']))
                im_shape = im.shape
            except FileNotFoundError:
                raise Exception("Can't find input image %s or infer its shape, so don't know what "
                                "windows to cache!" %
                                REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards['image_name']))
    except KeyError:
        # then there was no wildcards['image_name'], so grab the first one from
        # the DEFAULT_METAMERS list
        default_im = IMAGES[0]
        im_shape = default_im[default_im.index('size-') + len('size-'):]
        im_shape = im_shape.replace('.png', '')
        im_shape = [int(i) for i in im_shape.split(',')]
    try:
        max_ecc=float(wildcards['max_ecc'])
        min_ecc=float(wildcards['min_ecc'])
    except KeyError:
        # then there was no wildcards['max / min_ecc'], so grab the default values
        min_ecc = config['DEFAULT_METAMERS']['min_ecc']
        max_ecc = config['DEFAULT_METAMERS']['max_ecc']
    t_width = 1.0
    if 'cosine' in wildcards['model_name']:
        window_type = 'cosine'
    elif 'gaussian' in wildcards['model_name'] or wildcards['model_name'].startswith('Obs'):
        window_type = 'gaussian'
    scaling = wildcards['scaling']
    # make sure that scaling is 0-padded, if appropriate. e.g., 0.01, not .01
    if isinstance(scaling, str) and scaling[0] == '.':
        scaling = '0' + scaling
    if wildcards['model_name'].startswith("RGC"):
        # RGC model only needs a single scale of PoolingWindows.
        size = ','.join([str(i) for i in im_shape])
        return window_template.format(scaling=scaling, size=size,
                                      max_ecc=max_ecc, t_width=t_width,
                                      min_ecc=min_ecc, window_type=window_type,)
    elif wildcards['model_name'].startswith('V1') or wildcards['model_name'].startswith('Obs'):
        windows = []
        # need them for every scale
        try:
            num_scales = int(re.findall('s([0-9]+)', wildcards['model_name'])[0])
        except (IndexError, ValueError):
            num_scales = 4
        for i in range(num_scales):
            output_size = ','.join([str(int(np.ceil(j / 2**i))) for j in im_shape])
            windows.append(window_template.format(scaling=scaling, size=output_size,
                                                  max_ecc=max_ecc,
                                                  min_ecc=min_ecc,
                                                  t_width=t_width, window_type=window_type))
        return windows

def get_partition(wildcards, cluster):
    # if our V1 scaling value is small enough, we need a V100 and must specify
    # it. otherwise, we can use any GPU, because they'll all have enough
    # memory. The partition name depends on the cluster (greene or rusty), so
    # we have two different params, one for each, and the cluster config grabs
    # the right one. For now, greene doesn't require setting partition.
    if cluster not in ['greene', 'rusty']:
        raise Exception(f"Don't know how to handle cluster {cluster}")
    if int(wildcards.gpu) == 0:
        if cluster == 'rusty':
            return 'ccn'
        elif cluster == 'greene':
            return None
    else:
        if cluster == 'rusty':
            return 'gpu'
        elif cluster == 'greene':
            return None

def get_constraint(wildcards, cluster):
    if int(wildcards.gpu) > 0 and cluster == 'rusty':
        return 'v100-32gb'
    else:
        return ''

def get_cpu_num(wildcards):
    if int(wildcards.gpu) > 0:
        # then we're using the GPU and so don't really need CPUs
        cpus = 1
    else:
        # these are all based on estimates from rusty (which automatically
        # gives each job 28 nodes), and checking seff to see CPU usage
        try:
            if float(wildcards.scaling) > .06:
                cpus = 21
            elif float(wildcards.scaling) > .03:
                cpus = 26
            else:
                cpus = 28
        except AttributeError:
            # then we don't have a scaling attribute
            cpus = 10
    return cpus


def get_init_image(wildcards):
    if wildcards.init_type in ['white', 'gray', 'pink', 'blue']:
        return []
    else:
        try:
            # then this is just a noise level, and there is no input required
            float(wildcards.init_type)
            return []
        except ValueError:
            return utils.get_ref_image_full_path(wildcards.init_type)

rule create_metamers:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
        init_image = get_init_image,
    output:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'history.csv'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'history.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'window_check.svg'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'metamer-16.png'),
        METAMER_TEMPLATE_PATH.replace('.png', '.npy'),
        report(METAMER_TEMPLATE_PATH),
    log:
        METAMER_LOG_PATH,
    benchmark:
        METAMER_LOG_PATH.replace('.log', '_benchmark.txt'),
    resources:
        gpu = lambda wildcards: int(wildcards.gpu),
        cpus_per_task = get_cpu_num,
        mem = get_mem_estimate,
        # this seems to be the best, anymore doesn't help and will eventually hurt
        num_threads = 9,
    params:
        rusty_mem = lambda wildcards: get_mem_estimate(wildcards, 'rusty'),
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        # if we can use a GPU, synthesis doesn't take very long. If we can't,
        # it takes forever (7 days is probably not enough, but it's the most I
        # can request on the cluster -- will then need to manually ask for more
        # time).
        time = lambda wildcards: {1: '12:00:00', 0: '7-00:00:00'}[int(wildcards.gpu)],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        rusty_constraint = lambda wildcards: get_constraint(wildcards, 'rusty'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.clamp_each_iter == 'True':
                    clamp_each_iter = True
                elif wildcards.clamp_each_iter == 'False':
                    clamp_each_iter = False
                if wildcards.coarse_to_fine == 'False':
                    coarse_to_fine = False
                else:
                    coarse_to_fine = wildcards.coarse_to_fine
                if wildcards.init_type not in ['white', 'blue', 'pink', 'gray']:
                    init_type = fov.utils.get_ref_image_full_path(wildcards.init_type)
                else:
                    init_type = wildcards.init_type
                if resources.gpu == 1:
                    get_gid = True
                elif resources.gpu == 0:
                    get_gid = False
                else:
                    raise Exception("Multiple gpus are not supported!")
                if wildcards.save_all:
                    save_all = True
                else:
                    save_all = False
                with fov.utils.get_gpu_id(get_gid, on_cluster=ON_CLUSTER) as gpu_id:
                    fov.create_metamers.main(wildcards.model_name, float(wildcards.scaling),
                                             input.ref_image, int(wildcards.seed), float(wildcards.min_ecc),
                                             float(wildcards.max_ecc), float(wildcards.learning_rate),
                                             int(wildcards.max_iter), float(wildcards.loss_thresh),
                                             int(wildcards.loss_change_iter), output[0],
                                             init_type, gpu_id, params.cache_dir, input.norm_dict,
                                             wildcards.optimizer, float(wildcards.fract_removed),
                                             float(wildcards.loss_fract),
                                             float(wildcards.loss_change_thresh), coarse_to_fine,
                                             wildcards.clamp, clamp_each_iter, wildcards.loss,
                                             save_all=save_all, num_threads=resources.num_threads)


rule continue_metamers:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        norm_dict = get_norm_dict,
        continue_path = lambda wildcards: utils.find_attempts(dict(wildcards)).replace('_metamer.png', '.pt'),
        init_image = get_init_image,
    output:
        CONTINUE_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'history.csv'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'history.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'synthesis.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'window_check.svg'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'metamer-16.png'),
        CONTINUE_TEMPLATE_PATH.replace('.png', '.npy'),
        report(CONTINUE_TEMPLATE_PATH),
    log:
        CONTINUE_LOG_PATH,
    benchmark:
        CONTINUE_LOG_PATH.replace('.log', '_benchmark.txt'),
    resources:
        gpu = lambda wildcards: int(wildcards.gpu),
        mem = get_mem_estimate,
        cpus_per_task = get_cpu_num,
        # this seems to be the best, anymore doesn't help and will eventually hurt
        num_threads = 12,
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        time = lambda wildcards: {'V1': '12:00:00', 'RGC': '7-00:00:00'}[wildcards.model_name.split('_')[0]],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        rusty_constraint = lambda wildcards: get_constraint(wildcards, 'rusty'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.clamp_each_iter == 'True':
                    clamp_each_iter = True
                elif wildcards.clamp_each_iter == 'False':
                    clamp_each_iter = False
                if wildcards.coarse_to_fine == 'False':
                    coarse_to_fine = False
                else:
                    coarse_to_fine = wildcards.coarse_to_fine
                if wildcards.init_type not in ['white', 'blue', 'pink', 'gray']:
                    init_type = fov.utils.get_ref_image_full_path(wildcards.init_type)
                else:
                    init_type = wildcards.init_type
                if resources.gpu == 1:
                    get_gid = True
                elif resources.gpu == 0:
                    get_gid = False
                else:
                    raise Exception("Multiple gpus are not supported!")
                with fov.utils.get_gpu_id(get_gid, on_cluster=ON_CLUSTER) as gpu_id:
                    # this is the same as the original call in the
                    # create_metamers rule, except we replace max_iter with
                    # extra_iter, set learning_rate to None, and add the
                    # input continue_path at the end
                    fov.create_metamers.main(wildcards.model_name, float(wildcards.scaling),
                                             input.ref_image, int(wildcards.seed), float(wildcards.min_ecc),
                                             float(wildcards.max_ecc), None,
                                             int(wildcards.extra_iter), float(wildcards.loss_thresh),
                                             int(wildcards.loss_change_iter), output[0],
                                             init_type, gpu_id, params.cache_dir, input.norm_dict,
                                             wildcards.optimizer, float(wildcards.fract_removed),
                                             float(wildcards.loss_fract),
                                             float(wildcards.loss_change_thresh), coarse_to_fine,
                                             wildcards.clamp, clamp_each_iter, wildcards.loss,
                                             input.continue_path, num_threads=resources.num_threads)


rule gamma_correct_metamer:
    input:
        lambda wildcards: [m.replace('metamer.png', 'metamer.npy') for m in
                           utils.generate_metamer_paths(**wildcards)]
    output:
        report(METAMER_TEMPLATE_PATH.replace('metamer.png', 'metamer_gamma-corrected.png'))
    log:
        METAMER_LOG_PATH.replace('.log', '_gamma-corrected.log')
    benchmark:
        METAMER_LOG_PATH.replace('.log', '_gamma-corrected_benchmark.txt')
    run:
        import foveated_metamers as fov
        import contextlib
        import numpy as np
        import shutil
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if output[0].endswith('metamer_gamma-corrected.png'):
                    if ('degamma' in wildcards.image_name or
                        any([i in wildcards.image_name for i in LINEAR_IMAGES])):
                        print(f"Saving gamma-corrected image {output[0]} as np.uint8")
                        im = np.load(input[0])
                        im = im ** (1/2.2)
                        im = fov.utils.convert_im_to_int(im, np.uint8)
                        imageio.imwrite(output[0], im)
                    else:
                        print("Image already gamma-corrected, copying to {output[0]}")
                        shutil.copy(output[0].replace('_gamma-corrected', ''), output[0])


# for subject to learn task structure: comparing noise and reference images
rule collect_training_noise:
    input:
        # this duplication is *on purpose*. it's the easiest way of making sure
        # each of them show up twice in our stimuli array and description
        # dataframe
        op.join(config['DATA_DIR'], 'ref_images', 'uniform-noise_size-2048,2600.png'),
        op.join(config['DATA_DIR'], 'ref_images', 'pink-noise_size-2048,2600.png'),
        op.join(config['DATA_DIR'], 'ref_images', 'uniform-noise_size-2048,2600.png'),
        op.join(config['DATA_DIR'], 'ref_images', 'pink-noise_size-2048,2600.png'),
        [utils.get_ref_image_full_path(i) for i in IMAGES[:2]],
    output:
        op.join(config["DATA_DIR"], 'stimuli', 'training_noise', 'stimuli_comp-{comp}.npy'),
        report(op.join(config["DATA_DIR"], 'stimuli', 'training_noise', 'stimuli_description_comp-{comp}.csv')),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_noise', 'stimuli_comp-{comp}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_noise', 'stimuli_comp-{comp}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.stimuli.collect_images(input[:6], output[0])
                df = []
                for i, p in enumerate(input):
                    if i < 4:
                        image_name = op.basename(input[-2:][i//2]).replace('.pgm', '').replace('.png', '')
                        # dummy scaling value
                        scaling = 1
                        model = 'training'
                        seed = i % 2
                    else:
                        image_name = op.basename(p).replace('.pgm', '').replace('.png', '')
                        scaling = None
                        model = None
                        seed = None
                    df.append(pd.DataFrame({'base_signal': p, 'image_name': image_name, 'model': model,
                                            'scaling': scaling, 'seed': seed}, index=[0]))
                pd.concat(df).to_csv(output[1], index=False)


def get_training_metamers(wildcards):
    if wildcards.comp == 'ref':
        scaling = [config[wildcards.model_name.split('_')[0]]['scaling'][0],
                   config[wildcards.model_name.split('_')[0]]['scaling'][-1]]
        seed_n = [0]
    else:
        all_scaling = (config[wildcards.model_name.split('_')[0]]['scaling'] +
                       config[wildcards.model_name.split('_')[0]]['met_v_met_scaling'])
        scaling = [all_scaling[-8], all_scaling[-1]]
        seed_n = [0, 1]
    # we don't generate all the metamers for this comparison, so specify
    # exactly which images to grab
    if wildcards.model_name.startswith('RGC') and wildcards.comp == 'met':
        images = ['gnarled_range-.05,.95_size-2048,2600',
                  'boats_range-.05,.95_size-2048,2600']
    else:
        images = IMAGES[:2]
    mets = utils.generate_metamer_paths(scaling=scaling, image_name=images,
                                        seed_n=seed_n, **wildcards)
    return [m.replace('metamer.png', 'metamer.npy') for m in mets]
                    

# for subjects to get a sense for how this is done with metamers
rule collect_training_metamers:
    input:
        get_training_metamers,
        lambda wildcards: [utils.get_ref_image_full_path(i) for i in IMAGES[:2]]
    output:
        op.join(config["DATA_DIR"], 'stimuli', 'training_{model_name}', 'stimuli_comp-{comp}.npy'),
        report(op.join(config["DATA_DIR"], 'stimuli', 'training_{model_name}', 'stimuli_description_comp-{comp}.csv')),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_{model_name}', 'stimuli_comp-{comp}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_{model_name}', 'stimuli_comp-{comp}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.stimuli.collect_images(input, output[0])
                fov.stimuli.create_metamer_df(input, output[1])


rule collect_metamers:
    input:
        lambda wildcards: [m.replace('metamer.png', 'metamer.npy') for m in
                           utils.generate_metamer_paths(**wildcards)],
        lambda wildcards: [utils.get_ref_image_full_path(i, downsample='downsample' in wildcards.comp)
                           for i in IMAGES]
    output:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_comp-{comp}.npy'),
        report(op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv')),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'stimuli_comp-{comp}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'stimuli_comp-{comp}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.stimuli.collect_images(input, output[0])
                fov.stimuli.create_metamer_df(input, output[1])


rule create_masks:
    input:
        op.join(config["DATA_DIR"], 'stimuli', MODELS[0], 'stimuli_comp-ref.npy'),
        op.join(config["DATA_DIR"], 'stimuli', MODELS[0], 'stimuli_description_comp-ref.csv'),
    output:
        op.join(config['DATA_DIR'], 'stimuli', 'log-ecc-masks.npy'),
        op.join(config['DATA_DIR'], 'stimuli', 'log-ecc-masks_info.csv'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'stimuli', 'log-ecc-masks.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'stimuli', 'log-ecc-masks_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import numpy as np
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim = np.load(input[0])
                stim_df = pd.read_csv(input[1])
                max_ecc = stim_df.max_ecc.dropna().unique()
                assert len(max_ecc) == 1, "Found multiple max_ecc!"
                masks, mask_df = fov.stimuli.create_eccentricity_masks(stim.shape[-2:],
                                                                       max_ecc[0])
                np.save(output[0], masks)
                mask_df.to_csv(output[1], index=False)


def get_experiment_seed(wildcards):
    # the number from subject will be a number from 1 to 30, which we multiply
    # by 10 in order to get the tens/hundreds place, and the run number
    # will be between 0 and 5, which we use for the ones place. we use the same
    # seed for different model stimuli, since those will be completely
    # different sets of images.
    try:
        seed = 10*int(wildcards.subject.replace('sub-', '')) + int(wildcards.run_num)
    except ValueError:
        # then this is the training subject and seed doesn't really matter
        seed = int(wildcards.run_num)
    return seed


rule generate_experiment_idx:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
    output:
        report(op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                       '{subject}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}.npy')),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{subject}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{subject}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}_benchmark.txt'),
    params:
        seed = get_experiment_seed,
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim_df = pd.read_csv(input[0])
                try:
                    ref_image_to_include = fov.stimuli.get_images_for_session(wildcards.subject,
                                                                              int(wildcards.sess_num),
                                                                              'downsample' in wildcards.comp)
                    if 'training' in wildcards.model_name:
                        raise Exception("training models only allowed for sub-training!")
                    # in a session, we show 5 images. each run has 3 of those
                    # images, and we rotate so that each image shows up in 3
                    # runs.
                    idx = list(range(len(config['PSYCHOPHYSICS']['RUNS'])))
                    r = int(wildcards.run_num)
                    # we're 0-indexed, so r==4 is the 5th run
                    if r > 4 or len(idx) != 5:
                        raise Exception("This only works for 5 runs per session!")
                    idx = idx[r:r+3] + idx[:max(0, r-2)]
                    ref_image_to_include = ref_image_to_include[idx]
                except ValueError:
                    # then this is the training subject
                    if int(wildcards.sess_num) > 0 or int(wildcards.run_num):
                        raise Exception("only session 0 and run 0 allowed for sub-training!")
                    if 'training' not in wildcards.model_name:
                        raise Exception("only training models allowed for sub-training!")
                    else:
                        # if it is the traning model, then the stimuli description
                        # has already been restricted to only the values we want
                        ref_image_idx = [0, 1]
                    ref_image_to_include = stim_df.image_name.unique()[ref_image_idx]
                stim_df = stim_df.query("image_name in @ref_image_to_include")
                # we might have something after the - (like downsample-2), which
                # we don't want to include
                comp = 'met_v_' + wildcards.comp.split('-')[0]
                idx = fov.stimuli.generate_indices_split(stim_df, params.seed, comp, n_repeats=12)
                np.save(output[0], idx)


# for training, we want an array of correct responses so we can give feedback.
rule training_correct_responses:
    input:
        op.join(config["DATA_DIR"], 'stimuli', 'training_{model_name}', 'stimuli_description_comp-{comp}.csv'),
        op.join(config["DATA_DIR"], 'stimuli', 'training_{model_name}', 'task-split_comp-{comp}', 'sub-training',
                'sub-training_task-split_comp-{comp}_idx_sess-00_run-00.npy'),
    output:
        op.join(config["DATA_DIR"], 'stimuli', 'training_{model_name}', 'task-split_comp-{comp}', 'sub-training',
                'sub-training_task-split_comp-{comp}_sess-00_run-00_correct-responses.npy'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_{model_name}', 'task-split_comp-{comp}', 'sub-training',
                'sub-training_task-split_comp-{comp}_sess-00_run-00_correct-responses.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'training_{model_name}', 'task-split_comp-{comp}', 'sub-training',
                'sub-training_task-split_comp-{comp}_sess-00_run-00_correct-responses_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import numpy as np
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[0])
                idx = np.load(input[1])
                expt_df = fov.analysis.create_experiment_df_split(df, idx)
                np.save(output[0], expt_df.correct_response.values)


def get_all_idx(wildcards):
    if (wildcards.model_name == 'RGC_norm_gaussian' and wildcards.comp == 'met'):
        sessions = [0]
    else:
        sessions = config['PSYCHOPHYSICS']['SESSIONS']
    return [op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                    '{subject}_task-split_comp-{comp}_idx_sess-%02d_run-%02d.npy') % (s, r)
            for s in sessions for r in config['PSYCHOPHYSICS']['RUNS']]


rule generate_all_idx:
    input:
        get_all_idx,
    output:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{subject}_task-split_comp-{comp}_idx_tmp.txt')
    shell:
        'echo "This is a temporary file used by Snakemake to create all run index .npy files. It is otherwise unused." > {output}'


rule create_experiment_df:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{subject}_task-split_comp-{comp}_idx_sess-{sess_num}_run-{run_num}.npy'),
        op.join(config["DATA_DIR"], 'raw_behavioral', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}{ecc_mask}.hdf5'),
        op.join(config["DATA_DIR"], 'stimuli', 'log-ecc-masks_info.csv'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}', '{subject}',
                '{date}_{subject}_task-split_comp-{comp}{ecc_mask}_sess-{sess_num}_run-{run_num}_expt.csv'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}', '{subject}',
                '{date}_{subject}_task-split_comp-{comp}{ecc_mask}_sess-{sess_num}_run-{run_num}_trials.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                '{subject}', '{date}_{subject}_task-split_comp-{comp}{ecc_mask}_sess-{sess_num}_run-{run_num}_expt.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                '{subject}', '{date}_{subject}_task-split_comp-{comp}{ecc_mask}_sess-{sess_num}_run-{run_num}_expt_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import numpy as np
        import pandas as pd
        import re
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim_df = pd.read_csv(input[0])
                idx = np.load(input[1])
                trials = fov.analysis.summarize_trials(input[2])
                fig = fov.analysis.plot_timing_info(trials, wildcards.subject,
                                                    wildcards.sess_num,
                                                    wildcards.run_num)
                fig.savefig(output[1], bbox_inches='tight')
                df = fov.analysis.create_experiment_df_split(stim_df, idx)
                if len(wildcards.comp.split('-')) > 1:
                    df['trial_type'] = df.trial_type.apply(lambda x: x+'-'+wildcards.comp.split('-')[1])
                df = fov.analysis.add_response_info(df, trials, wildcards.subject,
                                                    wildcards.sess_num, wildcards.run_num)
                if wildcards.ecc_mask:
                    mask_idx = int(wildcards.ecc_mask.split('-')[1])
                    ecc_mask_df = pd.read_csv(input[3]).set_index('window_n')
                    ecc_mask_df = ecc_mask_df.loc[mask_idx]
                    df['min_ecc'] = ecc_mask_df.min_eccentricity
                    # the outer-most mask in ecc_mask_df will have
                    # max_eccentricity larger than the actual image (which is
                    # equivalent to saying that the mask hasn't "turnd off" by
                    # the edge of the image). for ease, we don't change max_ecc
                    # in that case.
                    if ecc_mask_df.max_eccentricity < df.max_ecc.unique()[0]:
                        df['max_ecc'] = ecc_mask_df.max_eccentricity
                df.to_csv(output[0], index=False)


rule combine_all_behavior:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'behavioral', '{{model_name}}', 'task-split_comp-{{comp}}{{ecc_mask}}', '{subject}',
                                   '{date}_{subject}_task-split_comp-{{comp}}{{ecc_mask}}_{sess}_run-{i:02d}_expt.csv').format(
                                       i=i, sess=ses, date=date, subject=subj)
                           for i in range(5)
                           for subj, subj_dict in BEHAVIORAL_DATA_DATES[wildcards.model_name][wildcards.comp+wildcards.ecc_mask].items()
                           for ses, date in subj_dict.items()],
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                'task-split_comp-{comp}{ecc_mask}_data.csv'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                'task-split_comp-{comp}{ecc_mask}_performance.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                'task-split_comp-{comp}{ecc_mask}_run_lengths.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                'task-split_comp-{comp}{ecc_mask}_plots.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}{ecc_mask}',
                'task-split_comp-{comp}{ecc_mask}_plots_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                expt_df = pd.concat([pd.read_csv(i) for i in input])
                expt_df.to_csv(output[0], index=False)
                g = fov.figures.performance_plot(expt_df, hue='subject_name',
                                                 height=2.5, curve_fit=True,
                                                 style='trial_type')
                g.fig.savefig(output[1], bbox_inches='tight')
                g = fov.figures.run_length_plot(expt_df, hue='subject_name')
                g.fig.savefig(output[2], bbox_inches='tight')


# only make this plot for the ref comparison, see the comments of the function for why
rule plot_loss_performance_comparison:
    input:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_data.csv'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-ref.csv'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_{x}_comparison.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_{x}_comparison_subjects.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_{x}_comparison_line.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_{x}_comparison.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_{x}_comparison_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim_df = pd.read_csv(input[-1])
                expt_df = pd.read_csv(input[0])
                g = fov.figures.compare_loss_and_performance_plot(expt_df, stim_df, x=wildcards.x)
                g.fig.savefig(output[0], bbox_inches='tight')
                g = fov.figures.compare_loss_and_performance_plot(expt_df, stim_df, x=wildcards.x, col_wrap=None, row='subject_name')
                g.fig.savefig(output[1], bbox_inches='tight')
                g = fov.figures.compare_loss_and_performance_plot(expt_df, stim_df, x=wildcards.x, col=None, plot_kind='line',
                                                                  height=5, logscale_xaxis=True if wildcards.x=='loss' else False)
                g.fig.savefig(output[2], bbox_inches='tight')


rule simulate_dataset:
    output:
        op.join(config["DATA_DIR"], 'behavioral', 'simulated_{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_data.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', 'simulated_{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_data.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', 'simulated_{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_data_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if 'V1' in wildcards.model_name:
                    scaling = config['V1']['scaling']
                    s0 = .08
                elif 'RGC' in wildcards.model_name:
                    scaling = config['RGC']['scaling']
                    s0 = .01
                simul = fov.mcmc.simulate_dataset(s0, 5, num_subjects=4,
                                                  trial_types=1, num_images=20,
                                                  num_trials=36)
                obs = simul.observed_responses.copy().astype(np.float32)
                # block out simulated data like our actual data is blocked out
                obs[:, :, :len(simul.subject_name):2, 10:15] = np.nan
                obs[:, :, 1:len(simul.subject_name):2, 15:] = np.nan
                simul['observed_responses'] = obs
                simul = simul.to_dataframe().reset_index()
                simul = simul.rename(columns={'observed_responses': 'hit_or_miss_numeric'})
                simul['model'] = f'simulated_{wildcards.model_name}'
                simul.to_csv(output[0], index=False)


rule mcmc:
    input:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_data.csv'),
    output:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_'
                'c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.nc'),
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_'
                'c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_scaling-extended.nc'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_'
                'c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_'
                'c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_benchmark.txt'),
    run:
        import contextlib
        import foveated_metamers as fov
        import pandas as pd
        import jax
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print(f"Running on {jax.device_count()} cpus!")
                dataset = fov.mcmc.assemble_dataset_from_expt_df(pd.read_csv(input[0]))
                mcmc_model = wildcards.mcmc_model
                interact_sd = None
                if 'interactions' in mcmc_model:
                    try:
                        interact_sd = float(re.findall('partially-pooled-interactions-([.0-9]+)', mcmc_model)[0])
                    except IndexError:
                        interact_sd = .1
                    mcmc_model = 'partially-pooled-interactions'
                mcmc = fov.mcmc.run_inference(dataset, mcmc_model,
                                              float(wildcards.step_size),
                                              int(wildcards.num_draws),
                                              int(wildcards.num_chains),
                                              int(wildcards.num_warmup),
                                              int(wildcards.seed),
                                              float(wildcards.accept_prob),
                                              int(wildcards.tree_depth),
                                              interact_sd=interact_sd)
                # want to have a different seed for constructing the inference
                # data object than we did for inference itself
                inf_data = fov.mcmc.assemble_inf_data(mcmc, dataset,
                                                      mcmc_model,
                                                      int(wildcards.seed)+1,
                                                      interact_sd=interact_sd)
                inf_data.to_netcdf(output[0])
                # want to have a different seed for constructing the inference
                # data object than we did for inference itself
                inf_data_extended = fov.mcmc.assemble_inf_data(mcmc, dataset,
                                                               mcmc_model,
                                                               int(wildcards.seed)+10,
                                                               extend_scaling=True,
                                                               interact_sd=interact_sd)
                inf_data_extended.to_netcdf(output[1])
                

rule mcmc_plots:
    input:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.nc'),
    output:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}.png'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}_'
                'c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_{mcmc_model}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}_benchmark.txt'),
    resources:
        mem = lambda wildcards: {'post-pred-check': 15, 'param-corr': 10}.get(wildcards.plot_type.split('_')[0], 5)
    run:
        import foveated_metamers as fov
        import arviz as az
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                inf_data = az.from_netcdf(input[0])
                if wildcards.plot_type == 'post-pred-check':
                    print("Creating posterior predictive check.")
                    fig = fov.figures.posterior_predictive_check(inf_data, col='subject_name', row='image_name', height=1.5,
                                                                 style='trial_type', hue='distribution')
                elif wildcards.plot_type == 'performance':
                    print("Creating performance plot.")
                    fig = fov.figures.posterior_predictive_check(inf_data,
                                                                 hue='subject_name',
                                                                 col='image_name',
                                                                 height=2.5,
                                                                 col_wrap=5,
                                                                 style='trial_type')
                elif wildcards.plot_type == 'diagnostics':
                    print("Creating MCMC diagnostics plot.")
                    fig = fov.figures.mcmc_diagnostics_plot(inf_data)
                elif 'param-corr' in wildcards.plot_type:
                    print("Creating MCMC parameter correlations plot.")
                    fig = fov.figures.mcmc_parameter_correlation(inf_data,
                                                                 wildcards.plot_type.split('_')[1])
                elif wildcards.plot_type == "param-avg-method":
                    print("Creating MCMC parameter average comparison plot.")
                    df = fov.mcmc.inf_data_to_df(inf_data, 'parameter grouplevel means',
                                                 query_str="distribution=='posterior'", hdi=.95)
                    df['avg_method'] = 'individual curves + avg'
                    tmp = fov.mcmc.inf_data_to_df(inf_data, 'parameter grouplevel means Heiko method',
                                                  query_str="distribution=='posterior'", hdi=.95)
                    tmp['avg_method'] = 'overall effects + combine'
                    df = pd.concat([df, tmp]).reset_index(drop=True)
                    fig = fov.figures.psychophysical_grouplevel_means(df, style=['avg_method', 'trial_type'])
                elif wildcards.plot_type == 'psychophysical-params':
                    print("Creating psychophysical parameters plot.")
                    fig = fov.figures.psychophysical_curve_parameters(inf_data,
                                                                      rotate_xticklabels=True,
                                                                      aspect=3,
                                                                      height=5,
                                                                      style='trial_type')
                elif wildcards.plot_type == 'pairplot':
                    print("Creating parameter pairplot.")
                    fig = fov.figures.parameter_pairplot(inf_data, hue='subject_name')
                elif wildcards.plot_type == 'params':
                    print("Creating parameter distribution plot.")
                    fig = fov.figures.partially_pooled_parameters(inf_data, height=4, aspect=2.5,
                                                                  rotate_xticklabels=True)
                elif wildcards.plot_type == 'interaction-params':
                    print("Creating interaction parameter distribution plot.")
                    fig = fov.figures.interaction_parameters(inf_data)
                elif wildcards.plot_type == 'metaparams':
                    print("Creating metaparameter distribution plot.")
                    fig = fov.figures.partially_pooled_metaparameters(inf_data, height=5)
                elif wildcards.plot_type == 'grouplevel':
                    print("Creating parameter grouplevel means distribution plot.")
                    fig = fov.figures.psychophysical_grouplevel_means(inf_data)
                else:
                    raise Exception(f"Don't know how to handle plot_type {wildcards.plot_type}!")
                fig.savefig(output[0], bbox_inches='tight')


rule mcmc_arviz_compare:
    # unlike the rule after this one, this uses arviz's built-in compare
    # functionality
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{{model_name}}', 'task-split_comp-{{comp}}',
                                   'task-split_comp-{{comp}}_mcmc_{mcmc_model}_{hyper}.nc').format(mcmc_model=m,
                                                                                                   hyper=utils.get_mcmc_hyperparams(wildcards, mcmc_model=m))
                           for m in ['unpooled', 'partially-pooled', 'partially-pooled-interactions']]
    output:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_ic-{ic}.csv'),
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_ic-{ic}_arviz.png')
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_ic-{ic}_arviz.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_ic-{ic}_arviz_benchmark.txt')
    run:
        import foveated_metamers as fov
        import arviz as az
        import re
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                models = {}
                for i in input:
                    inf = az.from_netcdf(i)
                    name = re.findall('mcmc_([a-z-]+)_step', i)[0]
                    models[name] = inf
                comp_df = az.compare(models, ic=wildcards.ic)
                ax = az.plot_compare(comp_df)
                ax.figure.savefig(output[1], bbox_inches='tight')
                # model label is found in the index of the df, so need to grab it back
                comp_df.reset_index().to_csv(output[0], index=False)


rule mcmc_compare_plot:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{{model_name}}', 'task-split_comp-{{comp}}',
                                   'task-split_comp-{{comp}}_mcmc_{mcmc_model}_{hyper}.nc').format(mcmc_model=m,
                                                                                                   hyper=utils.get_mcmc_hyperparams(wildcards, mcmc_model=m))
                           for m in ['unpooled', 'partially-pooled', 'partially-pooled-interactions',
                                     'partially-pooled-interactions-1']]
    output:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_{plot_type}.png'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_{plot_type}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_{plot_type}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import arviz as az
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if wildcards.plot_type == 'psychophysical-params':
                    df_type = 'psychophysical curve parameters'
                elif wildcards.plot_type == 'psychophysical-grouplevel':
                    df_type = 'parameter grouplevel means'
                df = []
                for i in input:
                    inf = az.from_netcdf(i)
                    df.append(fov.mcmc.inf_data_to_df(inf, df_type,
                                                      query_str="distribution=='posterior'", hdi=.95))
                df = pd.concat(df)
                if wildcards.plot_type == 'psychophysical-params':
                    fig = fov.figures.psychophysical_curve_parameters(df, style=['mcmc_model_type',
                                                                                 'trial_type'],
                                                                      row='trial_type',
                                                                      height=5, aspect=3,
                                                                      rotate_xticklabels=True)
                elif wildcards.plot_type == 'psychophysical-grouplevel':
                    fig = fov.figures.psychophysical_grouplevel_means(df, style=['mcmc_model_type', 'trial_type'])
                fig.savefig(output[0], bbox_inches='tight')


rule calculate_heterogeneity:
    input:
        [op.join(config["DATA_DIR"], 'ref_images_preproc', '{img}{{gammacorrected}}_range-.05,.95_size-2048,2600.png').format(img=img)
         for img in LINEAR_IMAGES]
    output:
        op.join(config['DATA_DIR'], 'figures', 'image_select', 'heterogeneity', 'heterogeneity{gammacorrected}.csv'),
        [op.join(config["DATA_DIR"], 'figures', 'image_select', 'heterogeneity', '{img}{{gammacorrected}}_map.png').format(img=img)
         for img in LINEAR_IMAGES]
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'image_select', 'heterogeneity', 'heterogeneity{gammacorrected}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'image_select', 'heterogeneity', 'heterogeneity{gammacorrected}_benchmark.txt')
    run:
        import plenoptic as po
        import foveated_metamers as fov
        import contextlib
        import pyrtools as pt
        import os.path as op
        import pandas as pd
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                images = po.to_numpy(po.load_images(input)).squeeze()
                df = []
                for n, im, out in zip(input, images, output[1:]):
                    # 7th pyramid scale is dominated by the edge of the picture
                    hg, tmp = fov.statistics.heterogeneity(im, pyramid_height=6)
                    n = op.split(n)[-1].split('_')
                    if 'symmetric' in n:
                        n = '_'.join(n[:2])
                    else:
                        n = n[0]
                    tmp['image'] = n
                    df.append(tmp)
                    print(n, out)
                    # this pads out the arrays so we can plot them all on
                    # imshow
                    pads = []
                    for i, h in enumerate(hg[::-1]):
                        ideal_shape = 2**i * np.array(hg[-1].shape)
                        pad = []
                        for x in ideal_shape - h.shape:
                            pad.append((np.ceil(x/2).astype(int),
                                        np.floor(x/2).astype(int)))
                        pads.append(pad)
                    # need to reverse the order, since we construct it backwards
                    pads = pads[::-1]
                    hg = [np.pad(h, pads[i]) for i, h in enumerate(hg)]
                    fig = pt.imshow(hg, zoom=.125,
                                    title=[f'heterogeneity {n}\nscale {i}' for i in range(len(hg))])
                    fig.savefig(out, bbox_inches='tight')
                df = pd.concat(df).reset_index(drop=True)
                df.to_csv(output[0], index=False)


rule compute_amplitude_spectra:
    input:
        [op.join(config["DATA_DIR"], 'ref_images_preproc', '{img}_range-.05,.95_size-2048,2600.png').format(img=img)
         for img in LINEAR_IMAGES],
        lambda wildcards: utils.generate_metamer_paths(**wildcards),
    output:
        op.join(config['DATA_DIR'], 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_amplitude-spectra.nc')
    log:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_amplitude-spectra.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_amplitude-spectra_benchmark.txt')
    run:
        import foveated_metamers as fov
        from collections import OrderedDict
        import xarray
        import contextlib
        import os
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if wildcards.model_name.startswith('RGC') and wildcards.comp.startswith('met'):
                    met_ref_imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
                else:
                    met_ref_imgs = LINEAR_IMAGES
                # make sure this is in the same order as LINEAR_IMAGES, whatever it is
                met_ref_imgs = sorted(met_ref_imgs, key=lambda x: LINEAR_IMAGES.index(x))
                seeds = set([int(re.findall('seed-(\d+)_', i)[0][-1]) for i in input if 'seed' in i])
                scalings = set([float(re.findall('scaling-([\d.]+)', i)[0])
                                for i in input if 'scaling' in i])
                # grab spectra for reference images
                ims = [i for i in input if 'scaling' not in i and 'seed' not in i]
                metadata = OrderedDict(model=wildcards.model_name, trial_type=f'met_v_{wildcards.comp}')
                ims = sorted(ims, key=lambda x: LINEAR_IMAGES.index([i for i in LINEAR_IMAGES if i in x][0]))
                assert len(ims) == len(LINEAR_IMAGES), f"Have too many images! Expected {len(LINEAR_IMAGES)}, but got {ims}"
                ref_image_spectra = fov.statistics.image_set_amplitude_spectra(ims, LINEAR_IMAGES, metadata)
                ref_image_spectra = ref_image_spectra.rename({'sf_amplitude': 'ref_image_sf_amplitude',
                                                              'orientation_amplitude': 'ref_image_orientation_amplitude'})
                spectra = []
                for scaling in scalings:
                    tmp_ims = [i for i in input if len(re.findall(f'scaling-{scaling}{os.sep}', i)) == 1]
                    tmp_spectra = []
                    for seed in seeds:
                        # grab spectra for all images with matching seed_n and scaling.
                        metadata = OrderedDict(model=wildcards.model_name, trial_type=f'met_v_{wildcards.comp}',
                                               scaling=scaling, seed_n=seed)
                        ims = [i for i in tmp_ims if len(re.findall(f'seed-\d*{seed}_', i)) == 1]
                        ims = sorted(ims, key=lambda x: LINEAR_IMAGES.index([i for i in LINEAR_IMAGES if i in x][0]))
                        assert len(ims) == len(met_ref_imgs), f"Have too many images! Expected {len(met_ref_imgs)}, but got {ims}"
                        tmp_spectra.append(fov.statistics.image_set_amplitude_spectra(ims, met_ref_imgs, metadata))
                    spectra.append(xarray.concat(tmp_spectra, 'seed_n'))
                spectra = xarray.concat(spectra, 'scaling')
                spectra = xarray.merge([spectra.rename({'sf_amplitude': 'metamer_sf_amplitude',
                                                        'orientation_amplitude': 'metamer_orientation_amplitude'}),
                                        ref_image_spectra])
                spectra.to_netcdf(output[0])


rule plot_amplitude_spectra:
    input:
        op.join(config['DATA_DIR'], 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_amplitude-spectra.nc')
    output:
        op.join(config['DATA_DIR'], 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{amplitude_type}-spectra.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{amplitude_type}-spectra_plot.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'amplitude_spectra', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{amplitude_type}-spectra_plot_benchmark.txt')
    run:
        import foveated_metamers as fov
        import xarray
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                ds = xarray.load_dataset(input[0])
                if wildcards.amplitude_type == 'sf':
                    g = fov.figures.amplitude_spectra(ds)
                elif wildcards.amplitude_type == 'orientation':
                    g = fov.figures.amplitude_orientation(ds)
                elif wildcards.amplitude_type == 'orientation-demeaned':
                    g = fov.figures.amplitude_orientation(ds, demean=True)
                g.savefig(output[0], bbox_inches='tight')


rule simulate_optimization:
    output:
        op.join(config['DATA_DIR'], 'simulate', 'optimization', 'a0-{a0}_s0-{s0}_seeds-{n_seeds}_iter-{max_iter}.svg'),
        op.join(config['DATA_DIR'], 'simulate', 'optimization', 'a0-{a0}_s0-{s0}_seeds-{n_seeds}_iter-{max_iter}_params.csv'),
        op.join(config['DATA_DIR'], 'simulate', 'optimization', 'a0-{a0}_s0-{s0}_seeds-{n_seeds}_iter-{max_iter}_data.csv'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'simulate', 'optimization', 'a0-{a0}_s0-{s0}_seeds-{n_seeds}_iter-{max_iter}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'simulate', 'optimization', 'a0-{a0}_s0-{s0}_seeds-{n_seeds}_iter-{max_iter}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fig, param, data = fov.simulate.test_optimization(float(wildcards.a0),
                                                                  float(wildcards.s0),
                                                                  n_opt=int(wildcards.n_seeds),
                                                                  max_iter=int(wildcards.max_iter))
                fig.savefig(output[0], bbox_inches='tight')
                param.to_csv(output[1], index=False)
                data.to_csv(output[2], index=False)


rule simulate_num_trials:
    output:
        op.join(config['DATA_DIR'], 'simulate', 'num_trials', 'trials-{n_trials}_a0-{a0}_s0-{s0}_boots-{n_boots}_iter-{max_iter}.svg'),
        op.join(config['DATA_DIR'], 'simulate', 'num_trials', 'trials-{n_trials}_a0-{a0}_s0-{s0}_boots-{n_boots}_iter-{max_iter}_params.csv'),
        op.join(config['DATA_DIR'], 'simulate', 'num_trials', 'trials-{n_trials}_a0-{a0}_s0-{s0}_boots-{n_boots}_iter-{max_iter}_data.csv'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'simulate', 'num_trials', 'trials-{n_trials}_a0-{a0}_s0-{s0}_boots-{n_boots}_iter-{max_iter}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'simulate', 'num_trials', 'trials-{n_trials}_a0-{a0}_s0-{s0}_boots-{n_boots}_iter-{max_iter}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fig, param, data = fov.simulate.test_num_trials(int(wildcards.n_trials), int(wildcards.n_boots),
                                                                float(wildcards.a0), float(wildcards.s0),
                                                                max_iter=int(wildcards.max_iter))
                fig.savefig(output[0], bbox_inches='tight')
                param.to_csv(output[1], index=False)
                data.to_csv(output[2], index=False)


rule simulate_num_trials_figure:
    input:
        [op.join(config['DATA_DIR'], 'simulate', 'num_trials', 'trials-{n_trials}_a0-5_s0-{s0}_boots-100_iter-10000_params.csv').format(n_trials=n, s0=s0)
         for s0 in [.1, .2] for n in [10, 20, 30, 40, 50]]
    output:
        report(op.join(config['DATA_DIR'], 'figures', '{context}', 'simulate', 'num_trials.svg'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'simulate', 'num_trials.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'simulate', 'num_trials.log')
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.concat([pd.read_csv(f) for f in input])
                font_scale = {'poster': 1.7}.get(wildcards.context, 1)
                with sns.plotting_context(wildcards.context, font_scale=font_scale):
                    g = fov.figures.simulate_num_trials(df)
                    g.fig.savefig(output[0], bbox_inches='tight')


rule scaling_comparison_figure:
    input:
        lambda wildcards: [m.replace('metamer.png', 'metamer_gamma-corrected.png') for m in
                           utils.generate_metamer_paths(**wildcards)],
        lambda wildcards: utils.get_ref_image_full_path(utils.get_gamma_corrected_ref_image(wildcards.image_name))
    output:
        report(op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                       '{image_name}_seed-{seed}_scaling.svg'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_seed-{seed}_scaling.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_seed-{seed}_scaling_benchmark.txt')
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import contextlib
        import re
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = {'poster': 1.7}.get(wildcards.context, 1)
                max_ecc = float(re.findall('em-([0-9.]+)_', input[0])[0])
                with sns.plotting_context(wildcards.context, font_scale=font_scale):
                    scaling = {MODELS[0]: config['RGC']['scaling'],
                               MODELS[1]: config['V1']['scaling']}[wildcards.model_name]
                    fig = fov.figures.scaling_comparison_figure(wildcards.model_name,
                        wildcards.image_name, scaling, wildcards.seed, max_ecc=max_ecc)
                    fig.savefig(output[0], bbox_inches='tight')


rule window_area_figure:
    input:
        windows = get_windows,
    output:
        report(op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                       'scaling-{scaling}_window_area.svg'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'scaling-{scaling}_window_area.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'scaling-{scaling}_window_area_benchmark.txt')
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import contextlib
        import torch
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = {'poster': 1.7}.get(wildcards.context, 1)
                min_ecc = config['DEFAULT_METAMERS']['min_ecc']
                max_ecc = config['DEFAULT_METAMERS']['max_ecc']
                size = [int(i) for i in config['IMAGE_NAME']['size'].split(',')]
                image = torch.rand((1, 1, *size))
                with sns.plotting_context(wildcards.context, font_scale=font_scale):
                    # remove the normalizing aspect, since we don't need it here
                    model, _, _, _ = fov.create_metamers.setup_model(wildcards.model_name.replace('_norm', ''),
                                                                     float(wildcards.scaling),
                                                                     image, min_ecc, max_ecc, params.cache_dir)
                    fig = fov.figures.pooling_window_area(model.PoolingWindows)
                    fig.savefig(output[0], bbox_inches='tight')


rule create_single_window:
    input:
        op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}.pt')
    output:
        op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}_single.pt')
    log:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}_single.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}_single_benchmark.txt')
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    resources:
        mem = get_mem_estimate,
    run:
        import foveated_metamers as fov
        import torch
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                size = [int(i) for i in wildcards.size.split(',')]
                image = torch.rand((1, 1, *size))
                model, _, _, _ = fov.create_metamers.setup_model(f'RGC_{wildcards.window_type}',
                                                                 float(wildcards.scaling),
                                                                 image,
                                                                 float(wildcards.min_ecc),
                                                                 float(wildcards.max_ecc),
                                                                 params.cache_dir)
                window = fov.utils.grab_single_window(model.PoolingWindows)
                torch.save(window, output[0])


def _get_image_for_window_example(wildcards):
    if wildcards.comp.startswith('met-downsample'):
        # if we set image_name when calling generate_metamer_paths, it will
        # ignore comp-met-downsample, so we need to manually update it
        wildcards.image_name = wildcards.image_name.replace('_range', '_' + wildcards.comp.replace('met-', '') + '_range')
    try:
        return utils.generate_metamer_paths(gamma_corrected=True, **wildcards)
    except Exception as e:
        if 'please specify the seed argument' not in str(e):
            raise e
        else:
            wildcards = dict(wildcards)
            wildcards['seed'] = wildcards.pop('seed_n')
            return utils.generate_metamer_paths(gamma_corrected=True, **wildcards)


def _get_single_window(wildcards):
    default_img_size = utils._find_img_size(config['DEFAULT_METAMERS']['image_name'][0])
    pix_to_deg = float(config['DEFAULT_METAMERS']['max_ecc']) / default_img_size.max()
    img_size = utils._find_img_size(wildcards.image_name)
    wildcards = dict(wildcards)
    wildcards['max_ecc'] = img_size.max() * pix_to_deg
    wildcards['min_ecc'] = config['DEFAULT_METAMERS']['min_ecc']
    window = get_windows(wildcards)
    if isinstance(window, list):
        return window[0].replace('.pt', '_single.pt')
    else:
        return window.replace('.pt', '_single.pt')


rule window_example_figure:
    input:
        _get_image_for_window_example,
        _get_single_window,
    output:
        report(op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                       '{image_name}_scaling-{scaling}_seed-{seed_n}_comp-{comp}_gpu-{gpu}_linewidth-{lw}_window.png'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_scaling-{scaling}_seed-{seed_n}_comp-{comp}_gpu-{gpu}_linewidth-{lw}_window.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_scaling-{scaling}_seed-{seed_n}_comp-{comp}_gpu-{gpu}_linewidth-{lw}_window_benchmark.txt')
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    run:
        import foveated_metamers as fov
        import contextlib
        import torch
        import imageio
        import matplotlib.pyplot as plt
        sys.path.append('extra_packages/pooling-windows/')
        import pooling
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, _ = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                image = fov.utils.convert_im_to_float(imageio.imread(input[0]))
                window = torch.load(input[1])
                # this is the same computation to get the window_max_amplitude
                # from the creation of the PoolingWindows attribute (but
                # asserting that the gaussian width is 1, which we guarantee
                # elsewhere). this way we don't need to load in the
                # corresponding PoolingWindows object (which may be huge) to
                # retrieve this value
                if 'gaussian' in wildcards.model_name:
                    window_max_amplitude = (1 / (1 * pooling.pooling.GAUSSIAN_SUM)) ** 2
                elif 'cosine' in wildcards.model_name:
                    window_max_amplitude = 1
                fig = fov.figures.pooling_window_example(window, image, vrange=(0, 1),
                                                         linewidths=float(wildcards.lw)*style['lines.linewidth'],
                                                         target_amp=window_max_amplitude / 2)
                fig.savefig(output[0])


rule pixelwise_diff_figure:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_comp-{comp}.npy'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', 'paper', '{model_name}',
                'scaling-{scaling}_comp-{comp}_pixelwise_errors.png'),
        op.join(config['DATA_DIR'], 'errors', '{model_name}',
                'scaling-{scaling}_comp-{comp}_pixelwise_errors.npy'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', '{model_name}',
                'scaling-{scaling}_comp-{comp}_pixelwise_errors.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', '{model_name}',
                'scaling-{scaling}_comp-{comp}_pixelwise_errors_benchmark.txt')
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import contextlib
        import numpy as np
        import pandas as pd
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = 1
                stim = np.load(input[0])
                stim_df = pd.read_csv(input[1])
                with sns.plotting_context('paper', font_scale=font_scale):
                    fig, errors = fov.figures.synthesis_pixel_diff(stim, stim_df, float(wildcards.scaling))
                    fig.savefig(output[0], bbox_inches='tight')
                    np.save(output[1], errors)


rule all_pixelwise_diff_figure:
    input:
        lambda wildcards: [op.join(config['DATA_DIR'], 'errors', '{{model_name}}',
                                   'scaling-{}_comp-{{comp}}_pixelwise_errors.npy').format(sc)
                           for sc in config[wildcards.model_name.split('_')[0]]['scaling']],
    output:
        op.join(config['DATA_DIR'], 'figures', 'paper', '{model_name}',
                'comp-{comp}_all_pixelwise_errors.png'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', '{model_name}',
                'comp-{comp}_all_pixelwise_errors.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', '{model_name}',
                'comp-{comp}_all_pixelwise_errors_benchmark.txt'),
    params:
        model_dict = lambda wildcards: config[wildcards.model_name.split('_')[0]]
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import contextlib
        import numpy as np
        import pyrtools as pt
        import pandas as pd
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = 1
                scaling = params.model_dict['scaling']
                errors = np.zeros((len(scaling),
                                   *[int(i) for i in config['IMAGE_NAME']['size'].split(',')]),
                                  dtype=np.float32)
                for i, f in enumerate(input):
                    errors[i] = np.load(f).mean(0)
                with sns.plotting_context('paper', font_scale=font_scale):
                    fig = pt.imshow([e for e in errors], zoom=.5, col_wrap=4,
                                    title=[f'scaling {sc}' for sc in scaling])
                    fig.suptitle('Pixelwise squared errors, averaged across images\n',
                                 va='bottom', fontsize=fig.axes[0].title.get_fontsize()*1.25)
                    fig.savefig(output[0], bbox_inches='tight')


rule performance_figure:
    input:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_data.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance{plot_focus}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance{plot_focus}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance{plot_focus}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        import matplotlib.pyplot as plt
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                expt_df = pd.read_csv(input[0])
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                col = 'image_name'
                hue = 'subject_name'
                height = fig_width / 6
                if wildcards.plot_focus == '_focus-image':
                    hue = 'model'
                elif wildcards.plot_focus == '_focus-subject':
                    col = None
                    height = fig_width / 3
                expt_df.model = expt_df.model.map(lambda x: {'RGC': 'Retina'}.get(x.split('_')[0],
                                                                                  x.split('_')[0]))
                df['model'] = df['model'].map(fov.plotting.MODEL_PLOT)
                df['trial_type'] = df['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                g = fov.figures.performance_plot(expt_df, hue=hue,
                                                 height=height, col=col,
                                                 curve_fit=True,
                                                 style='trial_type')
                if wildcards.context == 'paper':
                    g.fig.suptitle('')
                g.fig.savefig(output[0], bbox_inches='tight')


rule mcmc_figure:
    input:
        lambda wildcards: op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                                  f'task-split_comp-{{comp}}_mcmc_{{mcmc_model}}_{utils.get_mcmc_hyperparams(wildcards)}{{scaling_extended}}.nc'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc{scaling_extended}_{mcmc_model}_{plot_type}.{ext}'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc{scaling_extended}_{mcmc_model}_{plot_type}_{ext}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc{scaling_extended}_{mcmc_model}_{plot_type}_{ext}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import matplotlib.pyplot as plt
        import arviz as az
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                inf_data = az.from_netcdf(input[0])
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                if wildcards.plot_type == 'params-grouplevel':
                    inf_data = fov.mcmc.inf_data_to_df(inf_data, 'parameter grouplevel means',
                                                       query_str="distribution=='posterior'", hdi=.95)
                    inf_data['trial_type'] = inf_data['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                    col = 'level'
                    fig = fov.figures.psychophysical_grouplevel_means(inf_data,
                                                                      height=fig_width/4,
                                                                      col=col)
                    for ax in fig.axes:
                        ax.set_title(ax.get_title().replace('a0', 'gain').replace('s0', 'critical scaling'))
                    fig.suptitle(fig._suptitle.get_text(), y=1.05)
                elif 'performance' in wildcards.plot_type:
                    col = 'image_name'
                    hue = 'subject_name'
                    style = 'trial_type'
                    height = fig_width / 6
                    kwargs = {}
                    if 'focus' in wildcards.plot_type:
                        inf_data = fov.mcmc.inf_data_to_df(inf_data, 'predictive grouplevel means', hdi=.95)
                        if 'focus-image' in wildcards.plot_type:
                            if 'one-ax' in wildcards.plot_type:
                                assert inf_data.model.nunique() == 1, "focus-image_one-ax currently only works with one model!"
                                pal = {k: fov.plotting.get_palette('model', inf_data.model.unique())[inf_data.model.unique()[0]]
                                       for k in fov.plotting.get_palette('image_name')}
                                hue = 'image_name'
                                col = None
                                height = fig_width / 2
                                kwargs['palette'] = pal
                            else:
                                hue = 'model'
                            inf_data = inf_data.query("level=='image_name'").rename(
                                columns={'dependent_var': 'image_name'})
                            inf_data['subject_name'] = 'all subjects'
                        elif 'focus-outlier' in wildcards.plot_type:
                            # want to highlight nyc and llama by changing their color ...
                            pal = fov.plotting.get_palette('image_name_focus-outlier', inf_data.model.unique())
                            # and plotting them on top of other lines
                            kwargs['hue_order'] = sorted(fov.plotting.get_palette('image_name').keys())
                            zorder = [2 if k in ['nyc', 'llama'] else 1 for k in kwargs['hue_order']]
                            hue = 'image_name'
                            col = None
                            height = fig_width / 2
                            inf_data = inf_data.query("level=='image_name'").rename(
                                columns={'dependent_var': 'image_name'})
                            inf_data['subject_name'] = 'all subjects'
                            kwargs['palette'] = pal
                            kwargs['hue_kws'] = {'zorder': zorder}
                        elif 'focus-subject' in wildcards.plot_type:
                            col = None
                            if 'one-ax' in wildcards.plot_type:
                                height = fig_width / 2
                                assert inf_data.model.nunique() == 1, "focus-image_one-ax currently only works with one model!"
                                pal = {k: fov.plotting.get_palette('model', inf_data.model.unique())[inf_data.model.unique()[0]]
                                       for k in fov.plotting.get_palette('subject_name')}
                                kwargs['palette'] = pal
                            else:
                                height = fig_width / 3
                            inf_data = inf_data.query("level=='subject_name'").rename(
                                columns={'dependent_var': 'subject_name'})
                            inf_data['image_name'] = 'all images'
                    elif 'all' in wildcards.plot_type:
                        kwargs['col_wrap'] = 5
                        kwargs['hdi'] = 0
                        kwargs['markersize'] = .5 * plt.rcParams['lines.markersize']
                        # get rid of those lines that we don't have observations for
                        inf_data = fov.mcmc._drop_no_observations(inf_data, fov.mcmc.inf_data_to_df(inf_data, 'predictive', hdi=0))

                    inf_data['model'] = inf_data['model'].map(fov.plotting.MODEL_PLOT)
                    inf_data['trial_type'] = inf_data['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                    g = fov.figures.posterior_predictive_check(inf_data,
                                                               col=col,
                                                               hue=hue,
                                                               style=style,
                                                               height=height,
                                                               increase_size=False,
                                                               **kwargs)
                    fig = g.fig
                else:
                    raise Exception(f"Don't know how to handle plot type {wildcards.plot_type}!")
                if 'focus-outlier' in wildcards.plot_type or 'one-ax' in wildcards.plot_type:
                    # don't need the legend here, it's not doing much
                    warnings.warn("Removing legend, because it's not doing much.")
                    fig.legends[0].remove()
                    if 'V1' in wildcards.model_name:
                        warnings.warn("Removing ylabel so we don't have redundant labels when composing figure")
                        fig.axes[0].set_ylabel('')
                    if wildcards.comp == 'ref':
                        warnings.warn("Removing xlabel so we don't have redundant labels when composing figure")
                        fig.axes[0].set_xlabel('')
                elif wildcards.plot_type == 'performance-all':
                    fig.subplots_adjust(right=.95)
                    # this only needs to be done on the first axis, because
                    # their ticks are yoked together
                    fig.axes[0].set_xticks(fig.axes[0].get_xticks()[::2])
                if wildcards.context == 'paper':
                    fig.suptitle('')
                    for i, ax in enumerate(fig.axes):
                        # also need to move the titles down a bit
                        if wildcards.plot_type == 'performance-all':
                            # titles on this one need to be a bit higher than the rest
                            ax.set_title(ax.get_title(), y=.91)
                        else:
                            ax.set_title(ax.get_title(), y=.85)
                        # still running into this issue
                        # https://github.com/mwaskom/seaborn/issues/2293 with
                        # things about this size, so we manually set the
                        # xticklabels invisible
                        if col == 'image_name' and i <= 14:
                            [xticklab.set_visible(False) for xticklab in ax.get_xticklabels()]
                fig.savefig(output[0], bbox_inches='tight', transparent=True)


rule mcmc_compare_figure:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{{model_name}}', 'task-split_comp-{{comp}}',
                                   'task-split_comp-{{comp}}_mcmc_{mcmc_model}_{hyper}_scaling-extended.nc').format(mcmc_model=m,
                                                                                                                    hyper=utils.get_mcmc_hyperparams(wildcards, mcmc_model=m))
                           for m in ['unpooled', 'partially-pooled', 'partially-pooled-interactions']]
    output:
        op.join(config["DATA_DIR"], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_psychophysical-grouplevel_{axis_scale}_yax-{yaxis_dbl}.svg')
    log:
        op.join(config["DATA_DIR"], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_psychophysical-grouplevel_{axis_scale}_yax-{yaxis_dbl}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_psychophysical-grouplevel_{axis_scale}_yax-{yaxis_dbl}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import matplotlib.pyplot as plt
        import arviz as az
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                df = []
                for i in input:
                    inf = az.from_netcdf(i)
                    df.append(fov.mcmc.inf_data_to_df(inf, 'parameter grouplevel means',
                                                      query_str="distribution=='posterior'", hdi=.95))
                df = pd.concat(df)
                df.model = df.model.map(fov.plotting.MODEL_PLOT)
                df.trial_type = df.trial_type.map(fov.plotting.TRIAL_TYPE_PLOT)
                fig = fov.figures.psychophysical_grouplevel_means(df, style=['mcmc_model_type', 'trial_type'],
                                                                  height=fig_width/4, increase_size=False,
                                                                  row_order=['s0', 'a0'])
                for i, ax in enumerate(fig.axes):
                    if 'linear' == wildcards.axis_scale:
                        ax.set(yscale='linear')
                    elif 'log' == wildcards.axis_scale:
                        ax.set(yscale='log')
                    title = ax.get_title().replace('a0', "Max $d'$").replace('s0', 'Critical Scaling')
                    title = title.split('|')[0]
                    # remove title
                    ax.set_title('')
                    ax.set_xlabel(ax.get_xlabel().split('_')[0].capitalize())
                    if i % 2 == 0:
                        ax.set_ylabel(title)
                if wildcards.yaxis_dbl == 'double':
                    # need to draw so that the tick locations are set.
                    fig.canvas.draw()
                    # because the ylabels are longer for this one
                    if wildcards.model_name == 'RGC_norm_gaussian' and wildcards.comp == 'ref':
                        pos = -.20
                    else:
                        pos = -.15
                    fov.plotting.add_asymptotic_performance_yaxis(fig.axes[2], position=pos)
                fig.suptitle('')
                # change title of this to be clearer. this is an
                # exceedingly hacky way of doing this.
                leg = fig.legends[0]
                leg.texts = [t.set_text(t.get_text().replace('Trial type', 'Comparison').replace('Mcmc', 'MCMC'))
                             for t in leg.texts]
                fig.savefig(output[0], bbox_inches='tight', transparent=True)


rule mcmc_compare_psychophysical_params_figure:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{{model_name}}', 'task-split_comp-{{comp}}',
                                   'task-split_comp-{{comp}}_mcmc_{mcmc_model}_{hyper}.nc').format(mcmc_model=m,
                                                                                                   hyper=utils.get_mcmc_hyperparams(wildcards, mcmc_model=m))
                           for m in ['unpooled', 'partially-pooled', 'partially-pooled-interactions']]
    output:
        op.join(config["DATA_DIR"], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_unp-pp_psychophysical-params_{comp_method}.svg'),
        op.join(config["DATA_DIR"], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_unp-ppi_psychophysical-params_{comp_method}.svg'),
        op.join(config["DATA_DIR"], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_ppi-pp_psychophysical-params_{comp_method}.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_all_psychophysical-params_{comp_method}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_mcmc_compare_all_psychophysical-params_{comp_method}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        import arviz as az
        import matplotlib.pyplot as plt
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                df = []
                for i in input:
                    inf = az.from_netcdf(i)
                    tmp = fov.mcmc.inf_data_to_df(inf, 'psychophysical curve parameters',
                                                  query_str="distribution=='posterior'",
                                                  hdi=0)
                    df.append(fov.mcmc._drop_no_observations(inf, tmp))
                df = pd.concat(df)
                df.model = df.model.map(fov.plotting.MODEL_PLOT)
                df.trial_type = df.trial_type.map(fov.plotting.TRIAL_TYPE_PLOT)
                comps = [('unpooled', 'partially-pooled'), ('unpooled', 'partially-pooled-interactions'),
                         ('partially-pooled-interactions', 'partially-pooled')]
                for i, c in enumerate(comps):
                    g = fov.figures.compare_mcmc_psychophysical_params(df, c, height=fig_width/6,
                                                                       compare_method=wildcards.comp_method)
                    g.fig.subplots_adjust(right=.95, hspace=.27)
                    # change title of this to be clearer. this is an
                    # exceedingly hacky way of doing this.
                    leg = g.fig.legends[0]
                    leg.texts = [t.set_text(t.get_text().replace('Trial type', 'Comparison'))
                                 for t in leg.texts]
                    if wildcards.context == 'paper':
                        for j, ax in enumerate(g.fig.axes):
                            # still running into this issue
                            # https://github.com/mwaskom/seaborn/issues/2293 with
                            # things about this size, so we manually set the
                            # xticklabels invisible
                            if j <= 14:
                                [xticklab.set_visible(False) for xticklab in ax.get_xticklabels()]
                            ax.set_title(ax.get_title(), y=.91)
                    g.savefig(output[i], bbox_inches='tight', transparent=True)


rule mcmc_arviz_compare_figure:
    input:
        [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_compare_ic-{{ic}}.csv').format(comp=c, model_name=m)
         for m in MODELS
         for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'met-downsample-2', 'ref-natural'], 'RGC_norm_gaussian': ['ref', 'met']}[m]]
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'mcmc_compare_ic-{ic}.{ext}'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mcmc_compare_ic-{ic}_{ext}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mcmc_compare_ic-{ic}_{ext}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                _, fig_width = fov.style.plotting_style(wildcards.context)
                comp_df = []
                for p in input:
                    model, comp, ic = re.findall('mcmc/([A-Za-z0-9_]+)/task-split_comp-([a-z-0-9]+)/task.*_ic-([a-z]+).csv', p)[0]
                    tmp = pd.read_csv(p)
                    tmp['model'] = model
                    tmp['trial_type'] = f"metamer_vs_{comp.replace('ref', 'reference').replace('met', 'metamer').replace('-2', '')}"
                    comp_df.append(tmp)
                comp_df = pd.concat(comp_df)
                comp_df.model = comp_df.model.map(fov.plotting.MODEL_PLOT)
                comp_df.trial_type = comp_df.trial_type.map(fov.plotting.TRIAL_TYPE_PLOT)
                # the downsample title has an extra newline, which we'll remove here
                comp_df.trial_type = comp_df.trial_type.map(lambda x: x.replace('\n(', ' ('))
                aspect = 2
                height = (fig_width/comp_df.model.nunique()) / aspect
                row_order = comp_df.trial_type.unique()
                row_order = row_order[[0, 1, 4, 2, 3]]
                g = fov.figures.mcmc_arviz_compare(comp_df, row='trial_type',
                                                   col='model', aspect=aspect,
                                                   row_order=row_order,
                                                   height=height, sharex=False)
                g.savefig(output[0], bbox_inches='tight', transparent=True)


rule mcmc_performance_comparison_figure:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                                   'task-split_comp-{comp}_mcmc_{{mcmc_model}}_{hyper}{{scaling_extended}}.nc').format(comp=c, model_name=m,
                                                                                                                       hyper=utils.get_mcmc_hyperparams(wildcards, model_name=m, comp=c))
                           for m in MODELS
                           for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'ref-natural', 'met-downsample-2'], 'RGC_norm_gaussian': ['ref', 'met']}[m]],
        op.join(config['DATA_DIR'], 'dacey_data',
                'Dacey1992_mcmc_line-nooffset_step-.1_prob-.8_depth-10_c-4_d-1000_w-1000_s-10.nc'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'mcmc{scaling_extended}_{mcmc_model}_{mcmc_plot_type}_{focus}.{ext}'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}',
                'mcmc{scaling_extended}_{mcmc_model}_{mcmc_plot_type}_{focus}_{ext}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}',
                'mcmc{scaling_extended}_{mcmc_model}_{mcmc_plot_type}_{focus}_{ext}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import matplotlib.pyplot as plt
        import arviz as az
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                if wildcards.mcmc_plot_type == 'performance':
                    df_kind = 'predictive grouplevel means'
                    query_str = None
                elif 'params' in wildcards.mcmc_plot_type:
                    df_kind = 'parameter grouplevel means'
                    query_str = "distribution=='posterior'"
                df = []
                # use this because we may drop some of the x-values, but we
                # still want the same axes
                x_order = set()
                for f in input[:-1]:
                    tmp = az.from_netcdf(f)
                    if wildcards.focus.startswith('sub'):
                        subject_name = wildcards.focus.split('_')[0]
                        # each subject only sees 15 images, but we'll have
                        # parameters for all 20 for the partially pooled model
                        # because we modeled the image-level and subject-level
                        # effects separately and thus have predictions for the
                        # unseen (subject, image) pairs
                        if 'RGC_norm_gaussian' in f and 'comp-met' in f:
                            # this comparison only had one session
                            images = np.concatenate([fov.stimuli.get_images_for_session(subject_name, i,
                                                                                        'downsample' in f)
                                                     for i in range(1)])
                        else:
                            images = np.concatenate([fov.stimuli.get_images_for_session(subject_name, i,
                                                                                        'downsample' in f)
                                                     for i in range(3)])
                        x_order = x_order.union(set(tmp.posterior.subject_name.values))
                        tmp = tmp.sel(subject_name=subject_name, image_name=images)
                        # need to have subject_name as a dimension still
                        tmp.posterior = tmp.posterior.expand_dims('subject_name')
                        tmp.prior = tmp.prior.expand_dims('subject_name')
                        tmp.posterior_predictive = tmp.posterior_predictive.expand_dims('subject_name')
                        tmp.prior_predictive = tmp.prior_predictive.expand_dims('subject_name')
                        tmp.observed_data = tmp.observed_data.expand_dims('subject_name')
                    df.append(fov.mcmc.inf_data_to_df(tmp,
                                                      df_kind, query_str,
                                                      hdi=.95))
                df = pd.concat(df)
                # if x_order is empty, we want it to be None
                if not x_order:
                    x_order = None
                # else, make it a list and sort it
                else:
                    x_order = sorted(list(x_order))
                query_str = None
                perf_query_str = None
                if wildcards.focus.startswith('sub'):
                    focus = re.findall('(sub-[0-9]+)', wildcards.focus)[0]
                    if 'comp-natural' in wildcards.focus:
                        query_str = "trial_type in ['metamer_vs_metamer', 'metamer_vs_reference', 'metamer_vs_metamer-natural', 'metamer_vs_reference-natural']"
                    elif 'comp-downsample' in wildcards.focus:
                        query_str = "trial_type in ['metamer_vs_metamer', 'metamer_vs_reference', 'metamer_vs_metamer-downsample']"
                    perf_query_str = f"level=='subject_name' & dependent_var=='{focus}'"
                elif wildcards.focus == 'comp-all':
                    perf_query_str = "level=='all'"
                elif wildcards.focus == 'comp-base':
                    query_str = 'trial_type in ["metamer_vs_metamer", "metamer_vs_reference"]'
                    perf_query_str = 'level == "all"'
                elif wildcards.focus == 'comp-ref':
                    query_str = 'trial_type in ["metamer_vs_reference"] '
                    perf_query_str = 'level == "all"'
                else:
                    raise Exception(f"Don't know how to handle focus {wildcards.focus}!")
                if query_str is not None:
                    df = df.query(query_str)
                df['model'] = df['model'].map(fov.plotting.MODEL_PLOT)
                if wildcards.mcmc_plot_type == 'performance':
                    df['trial_type'] = df['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                    if perf_query_str is not None:
                        df = df.query(perf_query_str)
                    if not wildcards.focus.startswith('sub'):
                        df['subject_name'] = 'all subjects'
                    else:
                        df = df.rename(columns={'dependent_var': 'subject_name'})
                    df['image_name'] = 'all images'
                    g = fov.figures.posterior_predictive_check(df, col=None,
                                                               hue='model',
                                                               style='trial_type',
                                                               height=fig_width/2.5,
                                                               aspect=2,
                                                               logscale_xaxis=True,
                                                               increase_size=False)
                    g.fig.canvas.draw()
                    fov.plotting.add_physiological_scaling_bars(g.ax, az.from_netcdf(input[-1]))
                    fig = g.fig
                    if 'line' in wildcards.focus:
                        sc = re.findall('line-scaling-([.0-9]+)', wildcards.focus)[0]
                        g.ax.axvline(float(sc), color='k')
                elif 'params' in wildcards.mcmc_plot_type:
                    mean_line = {'none': False, 'lines': 'lines-only', 'ci': True}[wildcards.mcmc_plot_type.split('-')[-1]]
                    warnings.warn("Removing luminance model, metamer_vs_metamer for params plot (it's not informative)")
                    df = df.query("trial_type != 'metamer_vs_metamer' or model != 'Luminance model'")
                    df['trial_type'] = df['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                    fig = fov.figures.psychophysical_grouplevel_means(df,
                                                                      height=fig_width/4,
                                                                      mean_line=mean_line,
                                                                      x_order=x_order,
                                                                      increase_size=False,
                                                                      row_order=['s0', 'a0'])
                    for i, ax in enumerate(fig.axes):
                        if 'linear' in wildcards.mcmc_plot_type:
                            if 'a0' in ax.get_title():
                                ylim = (0, 10)
                            elif 's0' in ax.get_title():
                                ylim = (0, .5)
                            ax.set(yscale='linear', ylim=ylim)
                        elif 'log' in wildcards.mcmc_plot_type:
                            if 'sub-00' in  wildcards.focus:
                                if 'a0' in ax.get_title():
                                    ylim = (9e-1, 10)
                                elif 's0' in ax.get_title():
                                    ylim = (1e-2, 1e0)
                            elif 'comp-ref' in wildcards.focus:
                                if 'a0' in ax.get_title():
                                    ylim = (8e-1, 10)
                                elif 's0' in ax.get_title():
                                    ylim = (1e-2, 2e-1)
                            else:
                                if 'a0' in ax.get_title():
                                    ylim = (2e-1, 10)
                                elif 's0' in ax.get_title():
                                    ylim = (1e-2, 1e0)
                            ax.set(yscale='log', ylim=ylim)
                        title = ax.get_title().replace('a0', "Max $d'$").replace('s0', 'Critical Scaling')
                        title = title.split('|')[0]
                        # remove title
                        ax.set_title('')
                        ax.set_xlabel(ax.get_xlabel().split('_')[0].capitalize())
                        if i % 2 == 0:
                            ax.set_ylabel(title)
                    if wildcards.context == 'paper':
                        warnings.warn("Removing legend, because other panel will have it.")
                        fig.legends[0].remove()
                    fig.suptitle(fig._suptitle.get_text(), y=1.05)
                if wildcards.context == 'paper':
                    fig.suptitle('')
                    # change title of this to be clearer. this is an
                    # exceedingly hacky way of doing this.
                    if len(fig.legends) > 0:
                        leg = fig.legends[0]
                        leg.texts = [t.set_text(t.get_text().replace('Trial type', 'Comparison'))
                                     for t in leg.texts]
                fig.savefig(output[0], bbox_inches='tight', transparent=True)


rule mcmc_parameter_correlation_figure:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                                   'task-split_comp-{comp}_mcmc_{{mcmc_model}}_{hyper}.nc').format(comp=c, model_name=m,
                                                                                                   hyper=utils.get_mcmc_hyperparams(wildcards, model_name=m, comp=c))
                           for m in MODELS
                           for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'met-downsample-2', 'ref-natural'], 'RGC_norm_gaussian': ['ref', 'met']}[m]],
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'mcmc_{mcmc_model}_param-correlation.{ext}'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mcmc_{mcmc_model}_param-correlation_{ext}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mcmc_{mcmc_model}_param-correlation_{ext}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import arviz as az
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                df_kind = 'psychophysical curve parameters'
                query_str = "distribution=='posterior' & hdi==50"
                df = []
                for f in input:
                    df.append(fov.mcmc.inf_data_to_df(az.from_netcdf(f), df_kind, query_str, hdi=.95))
                df = pd.concat(df)
                index_cols = [col for col in df.columns if col not in ['parameter', 'value']]
                df = pd.pivot_table(df, 'value', index_cols, 'parameter').reset_index()
                df = df.rename(columns={'a0': 'Gain', 's0': 'Critical scaling'})
                g = sns.relplot(data=df, x='Critical scaling', y='Gain', col='model', row='trial_type',
                                facet_kws={'sharex': 'col', 'sharey': 'col'})
                g.savefig(output[0], bbox_inches='tight')


rule performance_comparison_figure:
    input:
        [op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                 'task-split_comp-{comp}_data.csv').format(comp=c, model_name=m)
         for m in MODELS
         for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'met-downsample-2', 'ref-natural']}.get(m, ['met', 'ref'])],
        op.join(config['DATA_DIR'], 'dacey_data',
                'Dacey1992_mcmc_line-offset_step-.1_prob-.8_depth-10_c-4_d-1000_w-1000_s-10.nc'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'performance_{focus}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'performance_{focus}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'performance_{focus}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        import matplotlib.pyplot as plt
        import arviz as az
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                col = None
                logscale_xaxis = True
                curve_fit = 'to_chance'
                if wildcards.focus.startswith('sub'):
                    query_str = f"subject_name=='{wildcards.focus}'"
                elif wildcards.focus.startswith('comp'):
                    if wildcards.focus == 'comp-base':
                        query_str = f'trial_type in ["metamer_vs_metamer", "metamer_vs_reference"]'
                    elif wildcards.focus == 'comp-ref':
                        query_str = f'trial_type == "metamer_vs_reference"'
                    elif wildcards.focus == 'comp-ref-natural':
                        query_str = (f'trial_type in ["metamer_vs_reference-natural", "metamer_vs_reference"] & '
                                     'subject_name=="sub-00" & model == "V1_norm_s6_gaussian"')
                        col = 'image_name'
                        logscale_xaxis = False
                    elif wildcards.focus == 'comp-met-natural':
                        query_str = (f'trial_type in ["metamer_vs_metamer-natural", "metamer_vs_metamer"] & '
                                     'subject_name=="sub-00" & model == "V1_norm_s6_gaussian"')
                        col = 'image_name'
                        curve_fit = True
                    elif wildcards.focus == 'comp-all':
                        query_str = ''
                    else:
                        raise Exception(f"Don't know how to handle focus {wildcards.focus}")
                else:
                    # then assume this is an image
                    query_str = f"image_name.str.startswith('{wildcards.focus}')"
                if query_str:
                    df = pd.concat([pd.read_csv(f).query(query_str) for f in input[:-1]])
                else:
                    df = pd.concat([pd.read_csv(f) for f in input[:-1]])
                df.model = df.model.map(lambda x: {'RGC': 'Retina'}.get(x.split('_')[0],
                                                                        x.split('_')[0]))
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                height = fig_width / 3 if col is None else fig_width / 6
                df['model'] = df['model'].map(fov.plotting.MODEL_PLOT)
                df['trial_type'] = df['trial_type'].map(fov.plotting.TRIAL_TYPE_PLOT)
                g = fov.figures.performance_plot(df, col=col,
                                                 curve_fit=curve_fit,
                                                 hue='model',
                                                 height=height,
                                                 style='trial_type',
                                                 aspect=2 if col is None else 1,
                                                 logscale_xaxis=logscale_xaxis)
                if col is None:
                    # need to draw so that the following code can check text size
                    g.fig.canvas.draw()
                    fov.plotting.add_physiological_scaling_bars(g.ax, az.from_netcdf(input[-1]))
                if wildcards.context == 'paper':
                    g.fig.suptitle('')
                g.fig.savefig(output[0], bbox_inches='tight')


rule ref_image_figure:
    input:
        op.join(config["DATA_DIR"], 'stimuli', MODELS[1], 'stimuli_comp-ref.npy'),
        op.join(config["DATA_DIR"], 'stimuli', MODELS[1], 'stimuli_description_comp-ref.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', 'poster', 'ref_images.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'poster', 'ref_images.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'poster', 'ref_images_benchmark.txt')
    run:
        import foveated_metamers as fov
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim = np.load(input[0])
                stim_df = pd.read_csv(input[1])
                style, fig_width = fov.style.plotting_style('poster')
                plt.style.use(style)
                fig = fov.figures.ref_image_summary(stim, stim_df)
                fig.savefig(output[0], bbox_inches='tight', pad_inches=0)


def get_ref_images(wildcards, gamma_corrected=True):
    img_sets = config['PSYCHOPHYSICS']['IMAGE_SETS']
    images = [op.join(config['DATA_DIR'], 'ref_images_preproc', im + '.png') for im in
              sorted(img_sets['all']) + sorted(img_sets['A']) + sorted(img_sets['B'])]
    if gamma_corrected:
        images = [im.replace('_range-', '_gamma-corrected_range-') for im in images]
    return images


# do this in a different way
rule ref_image_figure_paper:
    input:
        op.join('reports', 'figures', 'ref_images.svg'),
        get_ref_images,
    output:
        op.join(config['DATA_DIR'], 'figures', 'paper', 'ref_images.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', 'ref_images.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', 'paper', 'ref_images_benchmark.txt')
    run:
        import subprocess
        import shutil
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                shutil.copy(input[0], output[0])
                for i, im in enumerate(input[1:]):
                    # we add the trailing " to make sure we only replace IMAGE1, not IMAGE10
                    subprocess.call(['sed', '-i', f's|IMAGE{i+1}"|{im}"|', output[0]])


rule max_dprime_asymp_performance_figure:
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'max_dprime_asymp_perf.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'max_dprime_asymp_perf.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'max_dprime_asymp_perf_benchmark.txt')
    run:
        import foveated_metamers as fov
        import matplotlib.pyplot as plt
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context, figsize='half')
                plt.style.use(style)
                fig = fov.figures.max_dprime_vs_asymptotic_performance(figsize=(fig_width, fig_width))
                fig.savefig(output[0], bbox_inches='tight', transparent=True)


rule synthesis_video:
    input:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
    output:
        [METAMER_TEMPLATE_PATH.replace('metamer.png', f'synthesis-{i}.{f}') for
         i, f in enumerate(['png', 'png', 'png', 'mp4', 'png', 'mp4'])]
    log:
        METAMER_LOG_PATH.replace('.log', '_synthesis_video.log')
    benchmark:
        METAMER_LOG_PATH.replace('.log', '_synthesis_video_benchmark.txt')
    resources:
        mem = get_mem_estimate,
    run:
        import foveated_metamers as fov
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fov.figures.synthesis_video(input[0], wildcards.model_name)


def get_all_synth_images(wildcards):
    synth_imgs = utils.generate_metamer_paths(wildcards.synth_model_name,
                                               image_name=wildcards.image_name,
                                               comp='ref')
    # this has a reduced set of metamers that we test
    met_imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
    if not wildcards.synth_model_name.startswith("RGC") or any([wildcards.image_name.startswith(im) for im in met_imgs]):
        synth_imgs += utils.generate_metamer_paths(wildcards.synth_model_name,
                                                   image_name=wildcards.image_name,
                                                   comp='met')
    if wildcards.synth_model_name.startswith("V1"):
        synth_imgs += utils.generate_metamer_paths(wildcards.synth_model_name,
                                                   image_name=wildcards.image_name,
                                                   comp='met-natural')
        synth_imgs += utils.generate_metamer_paths(wildcards.synth_model_name,
                                                   image_name=wildcards.image_name,
                                                   comp='ref-natural')
    return synth_imgs



rule compute_distances:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
        synth_images = get_all_synth_images,
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'scaling-{scaling}',
                'synth-{synth_model_name}', '{image_name}_e0-{min_ecc}_em-{max_ecc}_distances.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'scaling-{scaling}',
                'synth-{synth_model_name}', '{image_name}_e0-{min_ecc}_em-{max_ecc}_distances.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'scaling-{scaling}',
                'synth-{synth_model_name}', '{image_name}_e0-{min_ecc}_em-{max_ecc}_distances_benchmark.txt')
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    resources:
        mem = get_mem_estimate,
    run:
        import foveated_metamers as fov
        import plenoptic as po
        import torch
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                ref_image = po.load_images(input.ref_image)
                if input.norm_dict:
                    norm_dict = torch.load(input.norm_dict)
                else:
                    norm_dict = None
                if wildcards.model_name.startswith('Obs'):
                    if wildcards.model_name == 'Obs_sfp':
                        # these values come from my spatial frequency
                        # preferences experiment, using fMRI to measure
                        # spatial frequency tuning in human V1
                        sf_params = {'sf_weighting_sigma': 2.2,
                                     'sf_weighting_slope': .12,
                                     'sf_weighting_intercept': 3.5,
                                     'sf_weighting_amplitude': 1,
                                     # this value was unmeasured in our
                                     # experiment, so I don't know what to do
                                     # with it
                                     'sf_weighting_mean_lum': 1}
                    elif wildcards.model_name == 'Obs_null':
                        # these values should be exactly equivalent to the V1
                        # model, not reweighting the values at all
                        sf_params = {'sf_weighting_sigma': 1e10,
                                     'sf_weighting_slope': 0,
                                     'sf_weighting_intercept': 1,
                                     'sf_weighting_amplitude': 1,
                                     'sf_weighting_mean_lum': 1}
                    else:
                        raise Exception("Don't know how to handle observer models without using sfp parameters!")
                    model = fov.ObserverModel(float(wildcards.scaling), ref_image.shape[-2:],
                                              6, 3, normalize_dict=norm_dict,
                                              cache_dir=params.cache_dir,
                                              min_eccentricity=float(wildcards.min_ecc),
                                              max_eccentricity=float(wildcards.max_ecc),
                                              **sf_params)
                else:
                    model = fov.create_metamers.setup_model(wildcards.model_name, float(wildcards.scaling),
                                                            ref_image, float(wildcards.min_ecc),
                                                            float(wildcards.max_ecc), params.cache_dir,
                                                            norm_dict)[0]
                synth_scaling = config[wildcards.synth_model_name.split('_')[0]]['scaling']
                met_imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
                if not wildcards.synth_model_name.startswith('RGC') or any([wildcards.image_name.startswith(im) for im in met_imgs]):
                    synth_scaling += config[wildcards.synth_model_name.split('_')[0]]['met_v_met_scaling']
                df = []
                for sc in synth_scaling:
                    df.append(fov.distances.model_distance(model, wildcards.synth_model_name,
                                                           wildcards.image_name, sc))
                df = pd.concat(df).reset_index(drop=True)
                df['distance_model'] = wildcards.model_name
                df['distance_scaling'] = float(wildcards.scaling)
                df.to_csv(output[0], index=False)


rule distance_plot:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'distances', '{{model_name}}',
                                   'scaling-{{scaling}}', 'synth-{synth_model_name}',
                                   '{image_name}_e0-{{min_ecc}}_em-{{max_ecc}}_distances.csv').format(synth_model_name=s, image_name=i)
                          for s in MODELS for i in IMAGES],
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'scaling-{scaling}',
                'e0-{min_ecc}_em-{max_ecc}_all_distances.csv'),
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'scaling-{scaling}',
                'e0-{min_ecc}_em-{max_ecc}_all_distances.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'scaling-{scaling}',
                'e0-{min_ecc}_em-{max_ecc}_all_distances.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'scaling-{scaling}',
                'e0-{min_ecc}_em-{max_ecc}_all_distances_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        import seaborn as sns
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.concat([pd.read_csv(f) for f in input]).reset_index(drop=True)
                df.to_csv(output[0], index=False)
                df['synthesis_model'] = df['synthesis_model'].apply(lambda x: x.split('_')[0])
                df['distance_model'] = df['distance_model'].apply(lambda x: x.split('_')[0])
                hue_order = fov.plotting.get_order('image_name')
                g = sns.catplot('synthesis_scaling', 'distance', 'ref_image', data=df,
                                col='trial_type', sharey=True, row='synthesis_model', kind='point',
                                sharex=False, col_order=['metamer_vs_reference', 'metamer_vs_metamer'],
                                hue_order=hue_order)
                for ijk, d in g.facet_data():
                    ax = g.facet_axis(*ijk[:2])
                    ax.set_xticklabels([f'{s:.03f}' for s in d.synthesis_scaling.unique()])
                g.set(yscale='log')
                g.savefig(output[1], bbox_inches='tight')


rule synthesis_distance_plot:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'distances', '{{model_name}}',
                                   'scaling-{scaling}', 'synth-{{model_name}}',
                                   '{image_name}_e0-{{min_ecc}}_em-{{max_ecc}}_distances.csv').format(image_name=i, scaling=s)
                           for s in {'V1': [.063, .27, 1.5], 'RGC': [.01, .058, 1.5]}[wildcards.model_name.split('_')[0]]
                           for i in IMAGES],
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}',
                'e0-{min_ecc}_em-{max_ecc}_synthesis_distances.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}',
                'e0-{min_ecc}_em-{max_ecc}_synthesis_distances.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}',
                'e0-{min_ecc}_em-{max_ecc}_synthesis_distances_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.concat([pd.read_csv(f) for f in input]).reset_index(drop=True)
                g = fov.figures.synthesis_distance_plot(df, x='metamer_vs_reference')
                g.savefig(output[0], bbox_inches='tight')


rule calculate_radial_squared_error:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_comp-{comp}.npy'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'radial_se_comp-{comp}.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'radial_se_comp-{comp}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'radial_se_comp-{comp}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import numpy as np
        import scipy
        import pyrtools as pt
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim = np.load(input[0])
                stim_df = pd.read_csv(input[1])
                if wildcards.model_name == 'RGC_norm_gaussian' and wildcards.comp == 'met':
                    # for this model and comparison, we only had 5 images
                    names = stim_df.image_name.unique()[:5].tolist()
                    stim_df = stim_df.query("image_name in @names")
                # create a dummy idx, which is not randomized (that's what setting seed=None does)
                idx = fov.stimuli.generate_indices_split(stim_df, None,
                                                         f'met_v_{wildcards.comp.split("-")[0]}',
                                                         12)
                # this contains all the relevant metadata we want for this comparison
                dist_df = fov.analysis.create_experiment_df_split(stim_df, idx)
                # remove duplicates, so we only have a single row for each
                # (target image, scaling, seed). this means we're throwing out
                # 3/4 of our rows
                if wildcards.comp.startswith('ref'):
                    dist_df = dist_df.drop_duplicates(['image_name', 'scaling', 'unique_seed'])
                elif wildcards.comp.startswith('met'):
                    # slightly more complicated to do this for the metamer comparisons
                    tmp = dist_df.apply(lambda x: sorted(set([x.image_left_1, x.image_left_2, x.image_right_1, x.image_right_2])), 1)
                    dist_df['seeds'] = tmp.apply(lambda x: f'{int(x[0])},{int(x[1])}')
                    dist_df = dist_df.drop_duplicates(['image_name', 'scaling', 'seeds'])
                    dist_df = dist_df.drop(columns=['seeds'])
                # initialize the images we'll use to compute radial means,
                # modified from fov.statistics.amplitude_spectra
                rbin = pt.synthetic_images.polar_radius(stim.shape[-2:]).astype(np.int)
                # only look at the circle that can fit in the image (i.e.,
                # throw away the corners, because there are fewer pixels out
                # there and so the averages won't be comparable)
                disk = pt.synthetic_images.polar_radius(stim.shape[-2:])
                thresh = min(stim.shape[-2:])//2
                disk = disk < thresh
                rbin[~disk] = rbin.max()+1
                # now iterate through all trials and compute the mse on each of
                # them
                df = []
                for _, row in dist_df.iterrows():
                    # unpack this to get the index of the stimulus on left and right, for first
                    # and second image
                    i = row.trial_number
                    [[l1, l2], [r1, r2]] = idx[:, i]
                    # need to cast these as floats else the MSE will be *way*
                    # off (like 100 instead of 8600). don't do the whole stim
                    # array at once because that would *drastically* increase
                    # memory use.
                    img1 = stim[l1].astype(float)
                    if l1 == l2:
                        img2 = stim[r2].astype(float)
                    else:
                        img2 = stim[l2].astype(float)
                    diff = np.square(img1-img2)
                    diff = scipy.ndimage.mean(diff, labels=rbin, index=np.arange(thresh-1))
                    row = row.to_dict()
                    row.update({'mse': diff, 'distance_pixels': np.arange(len(diff))})
                    tmp = pd.DataFrame(row)
                    tmp['distance_degrees'] = tmp.distance_pixels * ((2*tmp.max_ecc) / max(img1.shape[-2:]))
                    df.append(tmp)
                df = pd.concat(df)
                df.to_csv(output[0], index=False)


rule radial_squared_error_figure:
    input:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'radial_se_comp-{comp}.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}', 'radial_se_comp-{comp}_ecc-{ecc}.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}', 'radial_se_comp-{comp}_ecc-{ecc}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}', 'radial_se_comp-{comp}_ecc-{ecc}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import matplotlib.pyplot as plt
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[0])
                # sometimes the values have floating point precision issues,
                # e.g., 0.058 becoming 0.579999999999. this prevents that, so the legend is prettier
                df.scaling = np.round(df.scaling, 3)
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                if wildcards.ecc == 'None':
                    ecc_range = None
                else:
                    ecc_range = [float(e) for e in wildcards.ecc.split(',')]
                g = fov.figures.radial_mse(df, height=fig_width/6, ecc_range=ecc_range)
                for i, ax in enumerate(g.axes.flatten()):
                    # still running into this issue
                    # https://github.com/mwaskom/seaborn/issues/2293 with
                    # things about this size, so we manually set the
                    # xticklabels invisible
                    if i <= 14:
                        [xticklab.set_visible(False) for xticklab in ax.get_xticklabels()]
                g.savefig(output[0], bbox_inches='tight', transparent=True)


rule calculate_mse:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_comp-{comp}.npy'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'expt_mse_comp-{comp}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'expt_mse_comp-{comp}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import numpy as np
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim = np.load(input[0])
                stim_df = pd.read_csv(input[1])
                if wildcards.model_name == 'RGC_norm_gaussian' and wildcards.comp == 'met':
                    # for this model and comparison, we only had 5 images
                    names = stim_df.image_name.unique()[:5].tolist()
                    stim_df = stim_df.query("image_name in @names")
                # create a dummy idx, which is not randomized (that's what setting seed=None does)
                idx = fov.stimuli.generate_indices_split(stim_df, None,
                                                         f'met_v_{wildcards.comp.split("-")[0]}',
                                                         12)
                # this contains all the relevant metadata we want for this comparison
                dist_df = fov.analysis.create_experiment_df_split(stim_df, idx)
                expt_mse = np.empty(len(dist_df))
                met_mse = np.empty(len(dist_df))
                # now iterate through all trials and compute the mse on each of
                # them
                for i in range(len(dist_df)):
                    # unpack this to get the index of the stimulus on left and right, for first
                    # and second image
                    [[l1, l2], [r1, r2]] = idx[:, i]
                    # need to cast these as floats else the MSE will be *way*
                    # off (like 100 instead of 8600). don't do the whole stim
                    # array at once because that would *drastically* increase
                    # memory use. and calculate_experiment_mse function handles
                    # this internally
                    img1 = stim[l1].astype(float)
                    if l1 == l2:
                        img2 = stim[r2].astype(float)
                    else:
                        img2 = stim[l2].astype(float)
                    met_mse[i] = np.square(img1-img2).mean()
                    expt_mse[i] = fov.distances.calculate_experiment_mse(stim, idx[:, i])
                dist_df['experiment_mse'] = expt_mse
                dist_df['full_image_mse'] = met_mse
                dist_df['trial_structure'] = dist_df.apply(fov.distances._get_trial_structure, 1)
                dist_df['changed_side'] = dist_df.trial_structure.apply(lambda x: x[-1])
                dist_df.to_csv(output[0], index=False)


rule mse_plot:
    input:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}.csv'),
    output:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}_{mse}_heatmap.svg'),
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}_{mse}_plot.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'expt_mse_comp-{comp}_{mse}_plots.log'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'distances', '{model_name}', 'expt_mse_comp-{comp}_{mse}_plots_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        import seaborn as sns
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[0])
                g = fov.figures.mse_heatmap(df, wildcards.mse)
                g.savefig(output[0])
                df = fov.plotting._remap_image_names(df)
                style = 'changed_side' if wildcards.mse == 'experiment_mse' else None
                g = sns.relplot(data=df, x='scaling', y=wildcards.mse,
                                hue='image_name', style=style, kind='line',
                                palette=fov.plotting.get_palette('image_name'),
                                hue_order=fov.plotting.get_order('image_name'))
                g.savefig(output[1])


rule mse_performance_comparison:
    input:
        op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}.csv'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_data.csv'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{mse}_comparison.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{mse}_comparison_subjects.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{mse}_comparison_line.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{mse}_comparison.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_{mse}_comparison_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                mse_df = pd.read_csv(input[0])
                expt_df = pd.read_csv(input[1])
                g = fov.figures.compare_loss_and_performance_plot(expt_df,
                                                                  mse_df, x=wildcards.mse)
                g.fig.savefig(output[0], bbox_inches='tight')
                g = fov.figures.compare_loss_and_performance_plot(expt_df,
                                                                  mse_df, x=wildcards.mse,
                                                                  col_wrap=None, row='subject_name')
                g.fig.savefig(output[1], bbox_inches='tight')
                g = fov.figures.compare_loss_and_performance_plot(expt_df,
                                                                  mse_df,
                                                                  x=wildcards.mse,
                                                                  col=None,
                                                                  plot_kind='line',
                                                                  height=5)
                g.fig.savefig(output[2], bbox_inches='tight')


rule experiment_mse_example_img:
    input:
        lambda wildcards: utils.get_ref_image_full_path(wildcards.ref_image),
        lambda wildcards: utils.generate_metamer_paths(wildcards.model_name,
                                                       comp=wildcards.comp,
                                                       seed_n=int(wildcards.seed),
                                                       image_name=wildcards.ref_image,
                                                       scaling=float(wildcards.scaling),
                                                       init_type=wildcards.init_type)
    output:
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', 'expt_img1_{ref_image}_{init_type}_scaling-{scaling}_dir-{direction}_seed-{seed}.png'),
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', 'expt_img2_{ref_image}_{init_type}_scaling-{scaling}_dir-{direction}_seed-{seed}.png'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}', 'expt_{ref_image}_{init_type}_scaling-{scaling}_dir-{direction}_seed-{seed}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}', 'expt_{ref_image}_{init_type}_scaling-{scaling}_dir-{direction}_seed-{seed}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import numpy as np
        import imageio
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                bar_deg_size = 2
                screen_size_deg = 73.45
                screen_size_pix = 3840
                ref_image = imageio.imread(input[0])
                ref_image = ref_image / np.iinfo(ref_image.dtype).max
                metamer = imageio.imread(input[1])
                metamer = metamer / np.iinfo(metamer.dtype).max
                bar_pix_size = int(bar_deg_size * (screen_size_pix / screen_size_deg))
                bar = fov.distances._create_bar_mask(ref_image.shape[0], bar_pix_size)
                if wildcards.comp.startswith('ref'):
                    orig = ref_image.copy()
                elif wildcards.comp.startswith('met'):
                    orig = metamer.copy()
                orig = fov.distances._add_bar(orig, bar)
                imageio.imwrite(output[0], orig)
                half_width = ref_image.shape[-1] // 2
                if wildcards.direction == 'L':
                    ref_image[..., :half_width] = metamer[..., :half_width]
                elif wildcards.direction == 'R':
                    ref_image[..., half_width:] = metamer[..., half_width:]
                ref_image = fov.distances._add_bar(ref_image, bar)
                imageio.imwrite(output[1], ref_image)


rule mix_images_match_mse:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.ref_image),
        init_image = get_init_image,
        mse = op.join(config["DATA_DIR"], 'distances', '{model_name}', 'expt_mse_comp-{comp}.csv'),
    output:
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}.png'),
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}.npy'),
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}_synth.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}',
                '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}',
                '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                mse = pd.read_csv(input.mse)
                # sometimes the values have floating point precision issues,
                # e.g., 0.058 becoming 0.579999999999. this prevents that, so
                # they match the string we're passing in
                mse.scaling = np.round(mse.scaling, 3)
                mse = mse.query(f"image_name=='{wildcards.ref_image}' & scaling=={wildcards.scaling}")
                if wildcards.direction != 'None':
                    mse = mse.query(f"changed_side=='{wildcards.direction}'")
                    target_err = mse.experiment_mse.min()
                    direction = wildcards.direction
                else:
                    direction = None
                    target_err = mse.full_image_mse.min()
                if mse.empty:
                    raise Exception(f"No comparisons match image {wildcards.ref_image}, scaling {wildcards.scaling},"
                                    f" and direction {wildcards.direction}")
                # if the init_type corresponds to another image, we need its
                # full path. else, just the string identifying the type of
                # noise suffices
                if input.init_image:
                    other_img = input.init_image
                else:
                    other_img = wildcards.init_type
                fov.create_other_synth.main(input.ref_image, other_img,
                                            target_err, float(wildcards.lr),
                                            int(wildcards.max_iter),
                                            direction,
                                            int(wildcards.seed),
                                            output[0])


rule gamma_correct_match_mse:
    input:
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}.npy'),
    output:
        op.join(config['DATA_DIR'], 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}_gamma-corrected.png'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}_gamma-corrected.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'synth_match_mse', '{model_name}_comp-{comp}', '{ref_image}_init-{init_type}_scaling-{scaling}_dir-{direction}_lr-{lr}_max-iter-{max_iter}_seed-{seed}_gamma-corrected_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import numpy as np
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    print(f"Saving gamma-corrected image {output[0]} as np.uint8")
                    im = np.load(input[0])
                    # we know this lies between 0 and 255. we make this 256 to allow for 255.001 and similar
                    if im.max() < 1 or im.max() > 256:
                        raise Exception(f"Expected image to lie within (0, 255), but found max {im.max()}!")
                    im = (im / 255) ** (1/2.2)
                    im = fov.utils.convert_im_to_int(im, np.uint8)
                    imageio.imwrite(output[0], im)


rule create_mad_images:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        # we only support dir-None (that is, the full image synthesis). to
        # support the half-image, would need to modify the metrics so they only
        # compute loss on the target half (so gradient for other half is 0) and
        # then reset that other half at end of synthesis (pytorch seems to
        # randomly change pixels with 0 gradient)
        init_image = op.join(config['DATA_DIR'], 'synth_match_mse', '{met_model_name}_comp-{comp}', '{image_name}_init-{synth_init_type}_scaling-{scaling}_dir-None_lr-1e-9_max-iter-200_seed-0.png'),
    output:
        MAD_TEMPLATE_PATH.replace('_mad.png', '.pt'),
        MAD_TEMPLATE_PATH.replace('mad.png', 'synthesis.png'),
        MAD_TEMPLATE_PATH.replace('mad.png', 'image-diff.png'),
        MAD_TEMPLATE_PATH.replace('.png', '.npy'),
        report(MAD_TEMPLATE_PATH),
    log:
        MAD_LOG_PATH,
    benchmark:
        MAD_LOG_PATH.replace('.log', '_benchmark.txt'),
    resources:
        gpu = lambda wildcards: int(wildcards.gpu),
        cpus_per_task = get_cpu_num,
        mem = get_mem_estimate,
        # this seems to be the best, anymore doesn't help and will eventually hurt
        num_threads = 9,
    params:
        rusty_mem = lambda wildcards: get_mem_estimate(wildcards, 'rusty'),
        # if we can use a GPU, synthesis doesn't take very long. If we can't,
        # it takes forever (7 days is probably not enough, but it's the most I
        # can request on the cluster -- will then need to manually ask for more
        # time).
        time = lambda wildcards: {1: '12:00:00', 0: '7-00:00:00'}[int(wildcards.gpu)],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        rusty_constraint = lambda wildcards: get_constraint(wildcards, 'rusty'),
    run:
        import foveated_metamers as fov
        import contextlib
        import matplotlib as mpl
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # having issues with the default matplotlib backend causing
                # core dumps
                mpl.use('svg')
                if resources.gpu == 1:
                    get_gid = True
                elif resources.gpu == 0:
                    get_gid = False
                else:
                    raise Exception("Multiple gpus are not supported!")
                # tradeoff_lambda can be a float or None
                try:
                    tradeoff_lambda = float(wildcards.tradeoff_lambda)
                except ValueError:
                    tradeoff_lambda = None
                with fov.utils.get_gpu_id(get_gid, on_cluster=ON_CLUSTER) as gpu_id:
                    fov.create_mad_images.main('mse',
                                               wildcards.model_name,
                                               input.ref_image,
                                               wildcards.synth_target,
                                               int(wildcards.seed),
                                               float(wildcards.learning_rate),
                                               int(wildcards.max_iter),
                                               float(wildcards.stop_criterion),
                                               int(wildcards.stop_iters),
                                               output[0],
                                               input.init_image,
                                               gpu_id,
                                               wildcards.optimizer,
                                               tradeoff_lambda,
                                               float(wildcards.range_lambda),
                                               num_threads=resources.num_threads)


rule gamma_correct_mad_images:
    input:
        MAD_TEMPLATE_PATH.replace('.png', '.npy'),
    output:
        MAD_TEMPLATE_PATH.replace('.png', '_gamma-corrected.png')
    log:
        MAD_LOG_PATH.replace('.log', '_gamma-corrected.log'),
    benchmark:
        MAD_LOG_PATH.replace('.log', '_gamma-corrected_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import numpy as np
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    print(f"Saving gamma-corrected image {output[0]} as np.uint8")
                    im = np.load(input[0])
                    # we know this lies between 0 and 255. we make this 256 to allow for 255.001 and similar
                    if im.max() < 1 or im.max() > 256:
                        raise Exception(f"Expected image to lie within (0, 255), but found max {im.max()}!")
                    im = (im / 255) ** (1/2.2)
                    im = fov.utils.convert_im_to_int(im, np.uint8)
                    imageio.imwrite(output[0], im)


rule distance_vs_performance_plot:
    input:
        op.join(config["DATA_DIR"], 'distances', '{distance_model}', 'scaling-{scaling}', 'e0-{min_ecc}_em-{max_ecc}_all_distances.csv'),
        lambda wildcards: [op.join(config["DATA_DIR"], 'behavioral', '{{synthesis_model}}', 'task-split_comp-{comp}',
                                   'task-split_comp-{comp}_data.csv').format(comp=c)
                           for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'ref-natural']}.get(wildcards.synthesis_model, ['met', 'ref'])],
    output:
        op.join(config['DATA_DIR'], 'distances', '{distance_model}', 'scaling-{scaling}',
                '{synthesis_model}_e0-{min_ecc}_em-{max_ecc}_distance_vs_performance.{ext}')
    log:
        op.join(config['DATA_DIR'], 'logs', 'distances', '{distance_model}', 'scaling-{scaling}',
                '{synthesis_model}_e0-{min_ecc}_em-{max_ecc}_distance_vs_performance_{ext}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'distances', '{distance_model}', 'scaling-{scaling}',
                '{synthesis_model}_e0-{min_ecc}_em-{max_ecc}_distance_vs_performance_{ext}_benchmark.txt')
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                dist_df = pd.read_csv(input[0])
                expt_df = pd.concat([pd.read_csv(f) for f in input[1:]])
                logscale_xaxis = True if wildcards.synthesis_model.startswith('V1') else False
                g = fov.figures.compare_distance_and_performance(expt_df, dist_df,
                                                                 logscale_xaxis=logscale_xaxis)
                g.savefig(output[0])


rule scaling_gif:
    input:
        lambda wildcards: [f for comp in ['met', 'ref'] for f in
                           utils.generate_metamer_paths(wildcards.model_name, gamma_corrected=bool(wildcards.gammacorrected),
                                                        image_name=wildcards.image_name, seed_n=int(wildcards.seed),
                                                        comp=comp + {'natural': '-natural', 'white': ''}[wildcards.comp_type])]
    output:
        op.join(config['DATA_DIR'], 'scaling_gif', '{model_name}', '{image_name}_seed-{seed}_f-{fr}_{gif_direction}{gammacorrected}_comp-{comp_type}.gif')
    log:
        op.join(config['DATA_DIR'], 'logs', 'scaling_gif', '{model_name}', '{image_name}_seed-{seed}_f-{fr}_{gif_direction}{gammacorrected}_comp-{comp_type}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'scaling_gif', '{model_name}', '{image_name}_seed-{seed}_f-{fr}_{gif_direction}{gammacorrected}_comp-{comp_type}_benchmark.txt')
    run:
        import subprocess
        import contextlib
        import tempfile
        import shutil
        import os.path as op
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # remove duplicate files
                files = set(input)
                # sort by scaling
                if wildcards.gif_direction == 'reverse':
                    sort_files = list(reversed(sorted(files, key=lambda x: float(re.findall('scaling-([0-9.]+)', x)[0]))))
                elif wildcards.gif_direction == 'forward':
                    sort_files = sorted(files, key=lambda x: float(re.findall('scaling-([0-9.]+)', x)[0]))
                # assert that there's only one per scaling
                scaling_vals = [float(re.findall('scaling-([0-9.]+)', f)[0]) for f in sort_files]
                if len(scaling_vals) != len(set(scaling_vals)):
                    raise Exception("Each scaling value should show up exactly once!")
                tmpdir = tempfile.mkdtemp()
                for i, f in enumerate(sort_files):
                    shutil.copy(f, op.join(tmpdir, f'frame-{i:02d}.png'))
                print('\n'.join(sort_files))
                subprocess.call(['ffmpeg', '-f', 'image2', '-framerate', wildcards.fr, '-i', f'{tmpdir}/frame-%02d.png', output[0]])


rule freeman_windows:
    output:
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-{scaling}', 'plotwindows.mat'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-{scaling}', 'masks.mat'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'freeman_check', 'windows', 'scaling-{scaling}', 'windows.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'freeman_check', 'windows', 'scaling-{scaling}', 'windows_benchmark.txt'),
    params:
        met_path = config['FREEMAN_METAMER_PATH'],
        matlabPyrTools_path = config['MATLABPYRTOOLS_PATH'],
        output_dir = lambda wildcards, output: op.dirname(output[0]),
    shell:
        "cd matlab; "
        "matlab -nodesktop -nodisplay -r \"addpath(genpath('{params.met_path}')); "
        "addpath(genpath('{params.matlabPyrTools_path}')); "
        "freeman_windows({wildcards.scaling}, '{params.output_dir}'); quit;\""


rule freeman_check:
    input:
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.5', 'plotwindows.mat'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.25', 'plotwindows.mat'),
        # unclear from paper what exact scaling was used.
        utils.generate_metamer_paths(model_name='V1_norm_s4_gaussian',
                                     image_name='fountain_size-512,512',
                                     scaling=[.3, .4, .5],
                                     max_ecc=13,
                                     gpu=1,
                                     seed=0),

rule freeman_compare_windows_figure:
    input:
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.5', 'masks.mat'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.25', 'masks.mat'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'freeman_windows_comparison.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'freeman_windows_comparison.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'freeman_windows_comparison_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import h5py
        import contextlib
        import plenoptic as po
        import pyrtools as pt
        import sys
        sys.path.append('extra_packages/pooling-windows/')
        import pooling
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                to_plot = []
                titles = []
                for inp, sc in zip(input, [.5, .25]):
                    mat_idx = {.5: -85, .25: -355}[sc]
                    with h5py.File(inp, 'r') as f:
                        freeman_win = f[f['scale'][1, 0]]['maskMat'][:][..., mat_idx]
                    gauss_idx = {.25: (25, 25), .5: (11, 12)}[sc]
                    gauss_pw = pooling.PoolingWindows(sc, (512, 512),
                                                      max_eccentricity=13,
                                                      num_scales=5,
                                                      window_type='gaussian',
                                                      std_dev=1)
                    gauss_win = po.to_numpy(gauss_pw.ecc_windows[1][gauss_idx[0]]) * po.to_numpy(gauss_pw.angle_windows[1][gauss_idx[1]])
                    # normalize so that max value is 1, to match matlab implementation
                    gauss_win /= gauss_win.max()
                    to_plot.extend([freeman_win, gauss_win, freeman_win - gauss_win])
                    titles.extend([f'{paper}, scaling {sc}' for paper in ['Freeman 2011', 'This paper', 'Difference']])
                fig = pt.imshow(to_plot, vrange='auto0', col_wrap=3)
                for ax, title in zip(fig.axes, titles):
                    ax.set_title(title)
                    # adjust the position to get rid of extra spacing where
                    # subtitle used to be.
                    bb = ax.get_position()
                    if bb.y0 != 0:
                        bb.y0 = bb.y0 - bb.y0/4
                        ax.set_position(bb)
                    fov.figures.add_fixation_cross(ax, cross_size=25)
                fig.savefig(output[0], bbox_inches='tight', transparent=True)


rule dacey_figure:
    input:
        op.join('data/Dacey1992_RGC.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'Dacey1992.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'Dacey1992.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'Dacey1992_benchmark.txt')
    run:
        import contextlib
        import pandas as pd
        import seaborn as sns
        import foveated_metamers as fov
        import matplotlib.pyplot as plt
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[0])
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                pal = fov.plotting.get_palette('cell_type', df.cell_type.unique())
                g = sns.relplot(data=df, x='eccentricity_deg', y='dendritic_field_diameter_min',
                                # this aspect is approximately that of the
                                # figure in the paper
                                hue='cell_type', aspect=1080/725, palette=pal)
                g.set(xscale='log', yscale='log', xlim=(.1, 100), ylim=(1, 1000),
                      xlabel='Eccentricity (degrees)',
                      ylabel='Dendritic field diameter (min of arc)')
                g.savefig(output[0])


rule dacey_mcmc:
    input:
        op.join('data/Dacey1992_RGC.csv'),
    output:
        op.join(config['DATA_DIR'], 'dacey_data',
                'Dacey1992_mcmc_line-{line}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.nc'),
        op.join(config['DATA_DIR'], 'dacey_data',
                'Dacey1992_mcmc_line-{line}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_diagnostics.png'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'dacey_data',
                'Dacey1992_mcmc_line-{line}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'dacey_data',
                'Dacey1992_mcmc_line-{line}_step-{step_size}_prob-{accept_prob}_depth-{tree_depth}'
                '_c-{num_chains}_d-{num_draws}_w-{num_warmup}_s-{seed}_benchmark.txt'),
    run:
        import contextlib
        import pandas as pd
        import foveated_metamers as fov
        import arviz as az
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                dataset = fov.other_data.assemble_dacey_dataset(pd.read_csv(input[0]))
                mcmc = fov.other_data.run_phys_scaling_inference(dataset,
                                                                 wildcards.line == 'offset',
                                                                 float(wildcards.step_size),
                                                                 int(wildcards.num_draws),
                                                                 int(wildcards.num_chains),
                                                                 int(wildcards.num_warmup),
                                                                 int(wildcards.seed),
                                                                 float(wildcards.accept_prob),
                                                                 int(wildcards.tree_depth))
                # want to have a different seed for constructing the inference
                # data object than we did for inference itself
                inf_data = fov.other_data.assemble_inf_data(mcmc, dataset,
                                                            int(wildcards.seed)+1)
                inf_data.to_netcdf(output[0])
                fig = fov.figures.mcmc_diagnostics_plot(inf_data)
                fig.savefig(output[1], bbox_inches='tight')


rule dacey_mcmc_plot:
    input:
        op.join('data/Dacey1992_RGC.csv'),
        op.join(config['DATA_DIR'], 'dacey_data',
                'Dacey1992_mcmc_line-{line}_step-.1_prob-.8_depth-10_c-4_d-1000_w-1000_s-10.nc'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'Dacey1992_mcmc_line-{line}_{logscale}.{ext}')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'Dacey1992_mcmc_line-{line}_{logscale}_{ext}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}',
                'Dacey1992_mcmc_line-{line}_{logscale}_{ext}_benchmark.txt')
    run:
        import contextlib
        import pandas as pd
        import foveated_metamers as fov
        import arviz as az
        import seaborn as sns
        import matplotlib.pyplot as plt
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[0])
                inf_data = az.from_netcdf(input[1])
                style, fig_width = fov.style.plotting_style(wildcards.context, figsize='half')
                plt.style.use(style)
                fig = fov.figures.dacey_mcmc_plot(inf_data, df, logscale_axes='log' in wildcards.logscale,
                                                  height=fig_width)
                fig.savefig(output[0], transparent=True)


rule copy_schematic:
    input:
        op.join('reports/figures/{schematic_name}.svg'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{schematic_name}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{schematic_name}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{schematic_name}_benchmark.txt')
    shell:
        "cp {input} {output}"


rule psychophys_expt_table:
    input:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'psychophys_{expt}.svg')
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'psychophys_{expt}_with_table.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'psychophys_{expt}_with_table.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'psychophys_{expt}_with_table_benchmark.txt')
    run:
        import contextlib
        import foveated_metamers as fov
        import matplotlib.pyplot as plt
        import tempfile
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context,
                                                            figsize='full')
                plt.style.use(style)
                table_fig = fov.figures.psychophysics_schematic_table((3/4*fig_width, 3/4*fig_width/2))
                name = tempfile.NamedTemporaryFile().name + '.svg'
                table_fig.savefig(name, bbox_inches='tight')
                fig = fov.compose_figures.psychophys_schematic_with_table(input[0], name,
                                                                          wildcards.context)
                fig.save(output[0])


rule embed_bitmaps_into_figure:
    input:
        # marking this as ancient means we don't rerun this step if the
        # preferences file has changed, which is good because it changes
        # everytime we run this step
        ancient(config['INKSCAPE_PREF_FILE']),
        op.join(config['DATA_DIR'], '{folder}', '{context}', '{figure_name}.svg')
    output:
        op.join(config['DATA_DIR'], '{folder}', '{context}', '{figure_name}_dpi-{bitmap_dpi}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', '{folder}', '{context}', '{figure_name}_dpi-{bitmap_dpi}_svg.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', '{folder}', '{context}', '{figure_name}_dpi-{bitmap_dpi}_svg_benchmark.txt')
    run:
        import subprocess
        import shutil
        import contextlib
        import foveated_metamers as fov
        from glob import glob
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                orig_dpi = fov.figures.write_create_bitmap_resolution(input[0], wildcards.bitmap_dpi)
                ids, old_dd = fov.figures.get_image_ids(input[1], config['DATA_DIR'])
                shutil.copy(input[1], output[0])
                # if images were linked on a diferent machine, the data
                # directory is probably different, and this should fix it (see
                # fov.figures.grab_old_data_dir for how we figure out the old
                # data dir)
                if old_dd is not None:
                    subprocess.call(['sed', '-i', f's|{old_dd}|{config["DATA_DIR"]}|g', output[0]])
                select_ids = ''.join([f'select-by-id:{id};' for id in ids])
                action_str = select_ids + "SelectionCreateBitmap;select-clear;" + select_ids + "EditDelete;"
                action_str += "FileSave;"
                print(f"inkscape action string:\n{action_str}")
                subprocess.call(['inkscape', '--batch-process', f'--actions={action_str}', output[0]])
                # the inkscape call above embeds the bitmaps but also
                # apparently creates a separate png file containing the
                # embedded bitmaps, which we want to remove. commas get
                # replaced with underscores in the paths of those files, so
                # check for those as well
                extra_files = glob(output[0] + '-*') + glob(output[0].replace(',', '_') + '-*')
                print(f"will remove the following: {extra_files}")
                for f in extra_files:
                    try:
                        os.remove(f)
                    except FileNotFoundError:
                        # then the file was removed by something else
                        continue
                fov.figures.write_create_bitmap_resolution(input[0], orig_dpi)


def get_window_contour_figure_input(wildcards):
    if wildcards.fill.startswith('random') or wildcards.fill == 'none':
        return []
    else:
        return utils.get_ref_image_full_path(wildcards.fill)


rule window_contours_figure:
    input:
        get_window_contour_figure_input,
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'window_contours_fill-{fill}_size-{size}_scaling-{scaling}_linewidth-{lw}_background-{bg}.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'window_contours_fill-{fill}_size-{size}_scaling-{scaling}_linewidth-{lw}_background-{bg}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'window_contours_fill-{fill}_size-{size}_scaling-{scaling}_linewidth-{lw}_background-{bg}_benchmark.txt'),
    run:
        import contextlib
        import sys
        import matplotlib.pyplot as plt
        import foveated_metamers as fov
        import plenoptic as po
        sys.path.append(op.join(op.dirname(op.realpath(__file__)), 'extra_packages/pooling-windows'))
        import pooling
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                size = [int(i) for i in wildcards.size.split(',')]
                pw = pooling.PoolingWindows(float(wildcards.scaling), size, std_dev=1,
                                            window_type='gaussian')
                # we ignore figure size, because we are going to rescale this
                # when we move it around
                style, _ = fov.style.plotting_style(wildcards.context)
                if wildcards.bg == 'none':
                    # set both axes and figure facecolor to transparent
                    style['axes.facecolor'] = (0, 0, 0, 0)
                    style['figure.facecolor'] = (0, 0, 0, 0)
                elif wildcards.bg == 'white':
                    # want to see the border of the axis
                    style['axes.edgecolor'] = (0, 0, 0, 1)
                    style['axes.linewidth'] = .5*float(wildcards.lw)*style['lines.linewidth']
                else:
                    raise Exception("Can only handle background none or white!")
                plt.style.use(style)
                ax = None
                if 'random' in wildcards.fill:
                    seed = int(wildcards.fill.split('-')[-1])
                    np.random.seed(seed)
                    ax = pw.plot_window_values(subset=False)
                elif wildcards.fill != 'none':
                    im = po.load_images(input[0])
                    ax = pw.plot_window_values(im, subset=False)
                # since this is being shrunk, we need to make the lines thicker
                ax = pw.plot_windows(ax=ax, subset=False,
                                     linewidths=float(wildcards.lw)*style['lines.linewidth'])
                if wildcards.bg == 'none':
                    # this is the background image underneath the contour lines
                    # -- we want it to be invisible so we can overlay these
                    # contours on another image.
                    ax.images[0].set_visible(False)
                else:
                    # want to see the border of the axis
                    ax.set_frame_on(True)
                    ax.spines['top'].set_visible(True)
                    ax.spines['right'].set_visible(True)
                ax.figure.savefig(output[0], bbox_inches='tight')


rule model_schematic_figure:
    input:
        op.join('reports', 'figures', 'model_schematic_{width}.svg'),
        op.join(config['DATA_DIR'], 'ref_images_preproc', '{image_name}_gamma-corrected_range-.05,.95_size-2048,2600.png'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'model_schematic_{width}_{image_name}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'model_schematic_{width}_{image_name}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'model_schematic_{width}_{image_name}_benchmark.txt')
    run:
        import subprocess
        import shutil
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                shutil.copy(input[0], output[0])
                # we add the trailing " to make sure we only replace IMAGE1, not IMAGE10
                subprocess.call(['sed', '-i', f's|IMAGE1"|{input[1]}"|', output[0]])


def get_compose_figures_input(wildcards):
    path_template = os.path.join(config['DATA_DIR'], "figures", wildcards.context, "{}.svg")
    if 'model_schematic' in wildcards.fig_name:
        paths = [path_template.format(wildcards.fig_name),
                 path_template.format('window_contours_fill-random-1_size-2048,2600_scaling-1_linewidth-15_background-white'),
                 path_template.format('window_contours_fill-random-2_size-2048,2600_scaling-2_linewidth-36_background-white'),
                 path_template.format('window_contours_fill-random-3_size-2048,2600_scaling-2_linewidth-36_background-white'),
                 path_template.format('window_contours_fill-random-4_size-2048,2600_scaling-2_linewidth-36_background-white'),
                 path_template.format('window_contours_fill-random-5_size-2048,2600_scaling-2_linewidth-36_background-white')]
    if 'metamer_comparison' in wildcards.fig_name:
        paths = [path_template.format(wildcards.fig_name.replace('performance_', '').replace('scaling-extended_', ''))]
        if 'performance' in wildcards.fig_name:
            if 'scaling-extended' in wildcards.fig_name:
                paths.append(path_template.format('V1_norm_s6_gaussian/task-split_comp-ref_mcmc_scaling-extended_partially-pooled_performance_focus-outlier'))
            else:
                paths.append(path_template.format('V1_norm_s6_gaussian/task-split_comp-ref_mcmc_partially-pooled_performance_focus-outlier'))
    if 'all_comps_summary' in wildcards.fig_name:
        mcmc_model, focus = re.findall('all_comps_summary_([a-z-_]+)_focus-([a-z-_]+)', wildcards.fig_name)[0]
        paths = [
            path_template.format('{model}/task-split_comp-{comp}_mcmc_{mcmc_model}_performance_focus-{focus}').format(
                mcmc_model=mcmc_model, focus=focus, model=model, comp=comp)
            for model in ['RGC_norm_gaussian', 'V1_norm_s6_gaussian'] for comp in ['ref', 'met']
        ]
    if 'performance_comparison' in wildcards.fig_name:
        mcmc_model, details, comp, extra = re.findall('performance_comparison_([a-z-_]+)_([a-z-]+)_((?:sub-[0-9]+_)?comp-[a-z-]+)([_a-z0-9.-]+)?', wildcards.fig_name)[0]
        paths = [path_template.format(f'mcmc_{mcmc_model}_performance_{comp}{extra}'),
                 path_template.format(f'mcmc_{mcmc_model}_params-{details}_{comp}')]
    if 'radial_se' in wildcards.fig_name:
        comp, ecc = re.findall('radial_se_comp-([a-z-]+)_ecc-([A-Za-z0-9,]+)', wildcards.fig_name)[0]
        paths = [path_template.format(f'RGC_norm_gaussian/radial_se_comp-{comp}_ecc-{ecc}'),
                 path_template.format(f'V1_norm_s6_gaussian/radial_se_comp-{comp}_ecc-{ecc}')]
    if 'performance-all' in wildcards.fig_name:
        model_name, comp = re.findall('([A-Za-z0-9_]+)_comp-([a-z-]+)_performance-all', wildcards.fig_name)[0]
        paths = [path_template.format(f'{model_name}/task-split_comp-{comp}_mcmc_scaling-extended_partially-pooled_performance-all'),
                 path_template.format(f'{model_name}/task-split_comp-{comp}_mcmc_scaling-extended_unpooled_performance-all')]
    return paths


rule compose_figures:
    input:
        get_compose_figures_input,
    output:
        op.join(config['DATA_DIR'], 'compose_figures', '{context}', '{fig_name}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'compose_figures', '{context}', '{fig_name}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'compose_figures', '{context}', '{fig_name}_benchmark.txt')
    run:
        import subprocess
        import contextlib
        import foveated_metamers as fov
        import re
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if 'model_schematic' in wildcards.fig_name:
                    width = 'full' if 'full' in wildcards.fig_name else 'half'
                    fig = fov.compose_figures.model_schematic(input[0],
                                                              input[1],
                                                              input[2:], width,
                                                              wildcards.context)
                elif 'metamer_comparison' in wildcards.fig_name:
                    scaling = re.findall('scaling-([0-9,.]+)', wildcards.fig_name)[0]
                    scaling = [float(sc) for sc in scaling.split(',')]
                    if 'performance' in wildcards.fig_name:
                        # doesn't matter what we put in here, we're only using
                        # the outlier colors
                        pal = fov.plotting.get_palette('image_name_focus-outlier', ['V1'])
                        img_names = re.findall("metamer_comparison_([a-z,]+)_scaling", wildcards.fig_name)[0]
                        fig = fov.compose_figures.performance_metamer_comparison_small(input[1], input[0], scaling,
                                                                                       [pal[name] for name in img_names.split(',')])
                    elif 'natural-seed' in wildcards.fig_name:
                        if 'V1' in wildcards.fig_name:
                            labels = ['Target image',
                                      'Energy model metamer init with natural image 1',
                                      'Energy model metamer init with natural image 2',
                                      'Energy model metamer init with natural image 3',
                                      'Energy model metamer init with white noise 1',
                                      'Energy model metamer init with white noise 2']
                        elif 'RGC' in wildcards.fig_name:
                            labels = ['Target image',
                                      'Luminance metamer init with natural image 1',
                                      'Luminance metamer init with natural image 2',
                                      'Luminance metamer init with natural image 3',
                                      'Luminance metamer init with white noise 1',
                                      'Luminance metamer init with white noise 2']
                        fig = fov.compose_figures.metamer_comparison(*input, labels,
                                                                     'nocutout' not in wildcards.fig_name,
                                                                     True, wildcards.context)
                    else:
                        fig = fov.compose_figures.metamer_comparison(*input, scaling,
                                                                     'nocutout' not in wildcards.fig_name,
                                                                     False, wildcards.context)
                elif "all_comps_summary" in wildcards.fig_name:
                    fig = fov.compose_figures.combine_one_ax_figs(input, wildcards.context)
                elif "performance_comparison" in wildcards.fig_name:
                    fig = fov.compose_figures.performance_comparison(*input, wildcards.context)
                elif "radial_se" in wildcards.fig_name:
                    fig = fov.compose_figures.radial_squared_error(*input, wildcards.context)
                elif "performance-all" in wildcards.fig_name:
                    fig = fov.compose_figures.performance_all(*input, context=wildcards.context)
                fig.save(output[0])


def get_metamer_comparison_figure_inputs(wildcards):
    image_name = wildcards.image_name.split(',')
    if len(image_name) > 1 and 'small' not in wildcards.cutout:
        raise Exception("Only 'small' metamer comparison figure can have two images!")
    scaling = wildcards.scaling.split(',') * len(image_name)
    seeds = [0] * len(scaling)
    # if we're showing two of the same scaling values, for either model, want to
    # make sure the seeds are different
    models = ['RGC_norm_gaussian', 'RGC_norm_gaussian', 'V1_norm_s6_gaussian', 'V1_norm_s6_gaussian']
    if scaling[0] == scaling[1]:
        seeds[1] = 1
    if len(scaling) > 2 and scaling[2] == scaling[3]:
        seeds[3] = 1
    if len(scaling) > 5 and scaling[4] == scaling[5]:
        seeds[5] = 1
    uniq_imgs = image_name
    if 'natural-seed' in wildcards.cutout:
        if len(scaling) != 5:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 5 scaling values!")
        if 'V1' in wildcards.cutout:
            models = ['V1_norm_s6_gaussian'] * len(scaling)
        elif 'RGC' in wildcards.cutout:
            models = ['RGC_norm_gaussian'] * len(scaling)
        seeds = [0, 1, 2, 0, 1]
        image_name = image_name * len(scaling)
        comps = ['ref-natural', 'ref-natural', 'ref-natural', 'ref', 'ref']
    elif 'small' in wildcards.cutout:
        # we check against 4 but say 2 are required here, because we multiplied
        # scaling by len(image_name) above. since len(image_name) must be 2
        # (which we check next), this is equivalent to checking that the input
        # scaling was 2.
        if len(scaling) != 4:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 2 scaling values!")
        if len(image_name) != 2:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 2 image_name values!")
        models = ['V1_norm_s6_gaussian'] * len(scaling)
        image_name = image_name * 2
        # when we increased the length of scaling above, it interleaved the
        # values. this makes sure they're in the proper order
        scaling = sorted(scaling)
        comps = ['ref'] * len(scaling)
    elif 'downsample' in wildcards.cutout:
        if len(scaling) != 4:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 4 scaling values!")
        if len(image_name) != 1:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 1 image_name values!")
        models = ['V1_norm_s6_gaussian'] * len(scaling)
        image_name = image_name * len(scaling)
        # when we increased the length of scaling above, it interleaved the
        # values. this makes sure they're in the proper order
        scaling = sorted(scaling)
        comps = ['met'] * 2 +['met-downsample-2'] * 2
    else:
        if len(scaling) != 4:
            raise Exception(f"When generating {wildcards.cutout} metamer_comparison figure, need 4 scaling values!")
        image_name = image_name * len(scaling)
        comps = ['ref'] * len(scaling)
    paths = [
        op.join('reports', 'figures', 'metamer_comparison_{cutout}.svg'),
        *[op.join(config['DATA_DIR'], 'ref_images_preproc', '{image_name}_gamma-corrected_range-.05,.95_size-2048,2600.png').format(image_name=im)
          for im in uniq_imgs],
        *[op.join(config['DATA_DIR'], 'figures', '{{context}}', '{model_name}',
                  '{image_name}_range-.05,.95_size-2048,2600_scaling-{scaling}_seed-{seed}_comp-{comp}_gpu-{gpu}_linewidth-15_window.png').format(
                      model_name=m, scaling=sc, gpu=0 if float(sc) < config['GPU_SPLIT'] else 1, seed=s, image_name=im, comp=comp)
          for m, im, sc, s, comp in zip(models, image_name, scaling, seeds, comps)]
    ]
    if 'nocutout' not in wildcards.cutout:
        cuts = ['with_cutout_cross', 'foveal_cutout_cross', 'peripheral_cutout_cross']
        paths[len(uniq_imgs):] = [p.replace('.png', f'_{c}.png').replace('ref_images_preproc', f'figures{os.sep}{{context}}')
                                  for p in paths[len(uniq_imgs):] for c in cuts]
    return paths


rule metamer_comparison_figure:
    input:
        get_metamer_comparison_figure_inputs,
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'metamer_comparison_{image_name}_scaling-{scaling}_{cutout}.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'metamer_comparison_{image_name}_scaling-{scaling}_{cutout}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'metamer_comparison_{image_name}_scaling-{scaling}_{cutout}_benchmark.txt')
    run:
        import subprocess
        import shutil
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                shutil.copy(input[0], output[0])
                for i, im in enumerate(input[1:]):
                    print(f"Copying {im} into IMAGE{i+1}")
                    # we add the trailing " to make sure we only replace IMAGE1, not IMAGE10
                    subprocess.call(['sed', '-i', f's|IMAGE{i+1}"|{im}"|', output[0]])


rule mse_comparison_figure:
    input:
        op.join('reports', 'figures', 'mse_comparison_nocutout_with_target.svg'),
        lambda wildcards: utils.generate_metamer_paths(image_name=f'{wildcards.image_name}_range-.05,.95_size-2048,2600',
                                                       seed=0, scaling=float(wildcards.scaling), model_name='RGC',
                                                       gamma_corrected=True)[0],
        lambda wildcards: [op.join(config['DATA_DIR'], 'mad_images', 'ssim_{tgt}',
                                  f'RGC_norm_gaussian_comp-ref_scaling-{wildcards.scaling}_ref-{wildcards.image_name}_range-.05,.95_size-2048,2600_synth-{synth}',
                                  'opt-Adam_tradeoff-None_penalty-1e0_stop-iters-50', 'seed-0_lr-0.1_iter-10000_stop-crit-1e-9_gpu-1_mad_gamma-corrected.png').format(
                                      synth=synth, tgt=tgt) for synth, tgt in zip(['white', 'white', 'ivy_range-.05,.95_size-2048,2600'],
                                                                                  ['min', 'max', 'max'])],
        lambda wildcards: [op.join(config['DATA_DIR'], 'synth_match_mse', 'RGC_norm_gaussian_comp-ref',
                                  f'{wildcards.image_name}_range-.05,.95_size-2048,2600_init-{synth}_scaling-{wildcards.scaling}_dir-None_lr-1e-9_max-iter-200_seed-0_gamma-corrected.png').format(
                                      synth=synth) for synth in ['white', 'ivy_range-.05,.95_size-2048,2600']],
        lambda wildcards: op.join(config['DATA_DIR'], 'ref_images_preproc', '{image_name}_gamma-corrected_range-.05,.95_size-2048,2600.png'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'mse_comparison_{image_name}_scaling-{scaling}_nocutout_with_target.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mse_comparison_{image_name}_scaling-{scaling}_nocutout_with_target.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'mse_comparison_{image_name}_scaling-{scaling}_nocutout_with_target_benchmark.txt')
    run:
        import subprocess
        import shutil
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                shutil.copy(input[0], output[0])
                for i, im in enumerate(input[1:]):
                    print(f"Copying {im} into IMAGE{i+1}")
                    # we add the trailing " to make sure we only replace IMAGE1, not IMAGE10
                    subprocess.call(['sed', '-i', f's|IMAGE{i+1}"|{im}"|', output[0]])


def get_cutout_figures_input(wildcards):
    if '_window' in wildcards.image_name:
        model_name = wildcards.image_name.split(os.sep)[0]
        image_name = wildcards.image_name.split(os.sep)[1]
        return op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}', '{image_name}.png').format(
            model_name=model_name, image_name=image_name, context=wildcards.context)
    else:
        return op.join(config['DATA_DIR'], 'ref_images_preproc', '{image_name}.png').format(image_name=wildcards.image_name)


rule cutout_figures:
    input:
        get_cutout_figures_input,
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{image_name}_with_cutout_{fixation_cross}.png'),
        op.join(config['DATA_DIR'], 'figures', '{context}', '{image_name}_foveal_cutout_{fixation_cross}.png'),
        op.join(config['DATA_DIR'], 'figures', '{context}', '{image_name}_peripheral_cutout_{fixation_cross}.png'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{image_name}_with_cutout_{fixation_cross}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{image_name}_with_cutout_{fixation_cross}_benchmark.txt'),
    run:
        import subprocess
        import contextlib
        import foveated_metamers as fov
        import plenoptic as po
        import matplotlib.pyplot as plt
        import re
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, _ = fov.style.plotting_style(wildcards.context)
                # needs to be an int so we can properly use it to slice into
                # the iamge in the cutout_figure calls below
                style['lines.linewidth'] = int(16*style['lines.linewidth'])
                window_size = 400
                offset = [-800, -1000]
                cross_size = 150
                # change sizes of things so they look comparable to the regular
                # version when put into the comparison plot
                if 'downsample' in wildcards.image_name:
                    downsample_n = float(re.findall('downsample-([0-9]+)_', wildcards.image_name)[0])
                    # these need to be ints
                    window_size = int(window_size // downsample_n)
                    offset = [int(o // downsample_n) for o in offset]
                    style['lines.linewidth'] = int(style['lines.linewidth'] // downsample_n)
                    cross_size = cross_size / downsample_n
                plt.style.use(style)
                # if we're loading in the metamer with window, it will have a
                # red oval on it, which we want to preserve
                im = po.load_images(input, as_gray=False)
                # if we're loading in an image that is truly grayscale (i.e.,
                # the ref_images_preproc ones), then it will only have one
                # channel, even with as_gray=False, so we need to set as_rgb
                # correctly.
                fig = po.imshow(im, title=None, as_rgb=True if im.shape[1] > 1 else False,
                                # need to make sure vrange is set, so the
                                # dynamic range is the same as everywhere else
                                vrange=(0, 1))
                # we do the periphery and fovea separately, so we can plot them
                # in separate colors
                fov.figures.add_cutout_box(fig.axes[0], plot_periphery=False,
                                           window_size=window_size,
                                           periphery_offset=offset)
                fov.figures.add_cutout_box(fig.axes[0], plot_fovea=False, colors='b',
                                           window_size=window_size,
                                           periphery_offset=offset)
                if wildcards.fixation_cross == 'cross':
                    fov.figures.add_fixation_cross(fig.axes[0], cross_size=cross_size)
                # we add an extra bit to the window size here so that the
                # addition of the cutout box doesn't cause the axes to resize
                # (and the full width of the lines are visible)
                fovea_fig = fov.figures.cutout_figure(im[0, 0], plot_periphery=False, label=False,
                                                      window_size=window_size+style['lines.linewidth'],
                                                      periphery_offset=offset)
                periphery_fig = fov.figures.cutout_figure(im[0, 0], plot_fovea=False, label=False,
                                                          window_size=window_size+style['lines.linewidth'],
                                                          periphery_offset=offset)
                fov.figures.add_cutout_box(fovea_fig.axes[0], plot_periphery=False, periphery_offset=offset,
                                           window_size=window_size)
                # note that plot_periphery=False here because the peripheral
                # cutout is centered
                fov.figures.add_cutout_box(periphery_fig.axes[0], plot_periphery=False, colors='b',
                                           periphery_offset=offset, window_size=window_size)
                if wildcards.fixation_cross == 'cross':
                    fov.figures.add_fixation_cross(fovea_fig.axes[0], cross_size=cross_size)
                fig.savefig(output[0])
                fovea_fig.savefig(output[1])
                periphery_fig.savefig(output[2])


def get_all_windows(wildcards):
    wildcards.size = '2048,2600'
    windows = []
    for model in ['RGC_norm_gaussian', 'V1_norm_s6_gaussian']:
        wildcards.model_name = model
        for sc in config[model.split('_')[0]]['scaling']:
            wildcards.scaling = sc
            wdw = get_windows(wildcards)
            if isinstance(wdw, list):
                windows.extend(wdw)
            else:
                windows.append(wdw)
    return windows


rule number_of_stats:
    input:
        get_all_windows,
    output:
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.csv'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'number_of_stats.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'number_of_stats_benchmark.txt'),
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    run:
        import foveated_metamers as fov
        import contextlib
        import seaborn as sns
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = []
                min_ecc = config['DEFAULT_METAMERS']['min_ecc']
                max_ecc = config['DEFAULT_METAMERS']['max_ecc']
                img = torch.rand(1, 1, 2048, 2600)
                n_pix = img.nelement()
                # don't include "norm" in name, because we don't set the normalization dict
                for model_name in ['RGC_gaussian', 'V1_s6_gaussian']:
                    scaling = config[model_name.split('_')[0]]['scaling'] + config[model_name.split('_')[0]]['met_v_met_scaling']
                    for sc in scaling:
                        model = fov.create_metamers.setup_model(model_name, sc, img, min_ecc, max_ecc, params.cache_dir)[0]
                        rep = model(img)
                        tmp = pd.DataFrame({'model': model_name, 'scaling': sc, 'num_stats': rep.shape[-1],
                                            'num_pixels': n_pix}, [0])
                        df.append(tmp)
                df = pd.concat(df)
                df.model = df.model.map({k.replace('_norm', ''): v for k, v in
                                         fov.plotting.MODEL_PLOT.items()})
                df.to_csv(output[0], index=False)


rule number_of_stats_txt:
    input:
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.csv'),
    output:
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.svg'),
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.txt'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'number_of_stats_txt.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'number_of_stats_txt_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import torch
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                img = torch.rand(1, 1, 2048, 2600)
                n_pix = img.nelement()
                df = pd.read_csv(input[0])
                g, popts = fov.figures.number_of_stats(df)
                g.savefig(output[0], bbox_inches='tight')
                rgc_df = df.query('model=="Luminance model"')
                v1_df = df.query('model=="Energy model"')
                result = (
                    f'Luminance model number of statistics go from {rgc_df.num_stats.max()} for scaling {rgc_df.scaling.min()} '
                    f'({100 *rgc_df.num_stats.max() / n_pix:.03f}%)\n'
                    f'   to {rgc_df.num_stats.min()} for scaling {rgc_df.scaling.max()} ({100 * rgc_df.num_stats.min() / n_pix:.03f}%)\n'
                    f'   this is fit with the equation {popts[0][0]:.02f} * scaling ^ -2 + {popts[0][1]:.02f} * scaling ^ -1\n'
                    f'Energy model number of statistics go from {v1_df.num_stats.max()} for scaling {v1_df.scaling.min()} '
                    f'({100 * v1_df.num_stats.max() / n_pix:.03f}%)\n'
                    f'   to {v1_df.num_stats.min()} for scaling {v1_df.scaling.max()} ({100 * v1_df.num_stats.min() / n_pix:.03f}%)\n'
                    f'   this is fit with the equation {popts[1][0]:.02f} * scaling ^ -2 + {popts[1][1]:.02f} * scaling ^ -1\n'
                    f'For {n_pix} pixels\n\n'
                    'The relationship between scaling and number of windows, and thus number of stats, should be exactly an '
                    'inverse quadratic, but at larger scaling we get a breakdown of that, requiring more windows than expected,'
                    ' which is due to the fact that those windows are so large and so we need large windows that are barely on '
                    'the image in order for them to still uniformly tile.'
                )
                with open(output[-1], 'w') as f:
                    f.writelines(result)


rule critical_scaling_txt:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                                   'task-split_comp-{comp}_mcmc_partially-pooled_{hyper}_scaling-extended.nc').format(comp=c, model_name=m,
                                                                                                                      hyper=utils.get_mcmc_hyperparams(wildcards, comp=c, model_name=m,
                                                                                                                                                       mcmc_model='partially-pooled'))
                           for m in MODELS
                           for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'ref-natural'], 'RGC_norm_gaussian': ['ref', 'met']}[m]],
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.csv'),
    output:
        op.join(config['DATA_DIR'], 'statistics', 'critical_scaling.txt')
    log:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'critical_scaling.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'statistics', 'critical_scaling_benchmark.txt')
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import arviz as az
        from scipy import optimize
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                df = pd.read_csv(input[-1])
                crit_scaling = []
                for f in input[:-1]:
                    tmp = az.from_netcdf(f)
                    tmp = fov.mcmc.inf_data_to_df(tmp, 'parameter grouplevel means',
                                                  "distribution == 'posterior' & level == 'all' & hdi == 50",
                                                  hdi=.95)
                    crit_scaling.append(tmp.query('parameter=="s0"'))
                crit_scaling = pd.concat(crit_scaling)
                crit_scaling['model'] = crit_scaling['model'].map(fov.plotting.MODEL_PLOT)
                def quadratic_dec(scaling, a2, a1):
                    return a2*scaling**-2 + a1*scaling**-1
                out_str = ''
                for n, gb in df.groupby('model'):
                    popt, _ = optimize.curve_fit(quadratic_dec, gb.scaling.values,
                                                 gb.num_stats.values)
                    for m, gb2 in crit_scaling.query(f"model=='{n}'").groupby('trial_type'):
                        assert(len(gb2)==1)
                        crit = gb2.value.iloc[0]
                        n_stat = quadratic_dec(crit, *popt)
                        pct = 100 * n_stat / df.num_pixels.mean()
                        m = fov.plotting.TRIAL_TYPE_PLOT[m].replace('\n', ' ')
                        out_str += (f'For model {n} and comparison {m}, we have:\n   critical scaling: {crit:.04f}\n'
                                    f'   n stats: {n_stat:.02f}\n   percentage of pixels: {pct:.02f}%\n')
                with open(output[0], 'w') as f:
                    f.writelines(out_str)


rule critical_scaling_pointplot:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                                   'task-split_comp-{comp}_mcmc_partially-pooled_{hyper}_scaling-extended.nc').format(comp=c, model_name=m,
                                                                                                     hyper=utils.get_mcmc_hyperparams(wildcards, comp=c, model_name=m,
                                                                                                                                      mcmc_model='partially-pooled'))
                           for m in MODELS
                           for c in {'V1_norm_s6_gaussian': ['met', 'ref'], 'RGC_norm_gaussian': ['ref']}[m]],
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'critical_scaling_norm-{norm}.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'critical_scaling_norm-{norm}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'critical_scaling_norm-{norm}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import contextlib
        import pandas as pd
        import arviz as az
        import seaborn as sns
        import matplotlib as mpl
        import warnings
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, fig_width = fov.style.plotting_style(wildcards.context, figsize='half')
                crit_scaling = []
                for f in input:
                    tmp = az.from_netcdf(f)
                    tmp = fov.mcmc.inf_data_to_df(tmp, 'parameter grouplevel means',
                                                  "distribution == 'posterior' & level == 'all' & hdi == 50",
                                                  hdi=.95).query("parameter=='s0'")
                    tmp = tmp.drop(columns=['distribution', 'hdi', 'parameter', 'level', 'dependent_var',
                                            'mcmc_model_type'])
                    tmp = tmp.rename(columns={'value': 'critical_scaling'}).reset_index(drop=True)
                    crit_scaling.append(tmp)
                crit_scaling = pd.concat(crit_scaling)
                crit_scaling['model'] = crit_scaling['model'].map(fov.plotting.MODEL_PLOT)
                crit_scaling = pd.concat([crit_scaling, fov.figures.wallis_critical_scaling()])
                pal = fov.plotting.get_palette('model', fov.plotting.MODEL_PLOT.values())
                crit_scaling.model = crit_scaling.model.apply(lambda x: x.replace(' model', ''))
                pal = {k.replace(' model', ''): v for k, v in pal.items()}
                # color copied from Freeman and Simoncelli, 2011's V2 purple
                pal['Texture'] = (165/255, 109/255, 189/255)
                # put dummy data in for this Luminance met vs met, since we
                # can't actually fit
                tmp = crit_scaling.query("model=='Luminance' & trial_type == 'metamer_vs_reference'")
                tmp['trial_type']  = 'metamer_vs_metamer'
                if wildcards.norm == 'True':
                    tmp['critical_scaling'] = 8*tmp.critical_scaling
                else:
                    tmp['critical_scaling'] = 24*tmp.critical_scaling
                crit_scaling = pd.concat([crit_scaling, tmp])
                g = sns.FacetGrid(crit_scaling, hue='model', palette=pal, height=fig_width)
                g.map_dataframe(fov.plotting.vertical_pointplot, x='model', y='critical_scaling',
                                norm_y=wildcards.norm=='True')
                ylabel = 'Critical scaling'
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.norm == 'True':
                    ylabel += '\n(proportion of Original vs. Synth)'
                    g.ax.set_yscale('log', base=2)
                    g.ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(2, subs=(.5, )))
                    g.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
                g.set(xlabel='Pooling model', ylabel=ylabel, xlim=[-.5, 2.5])
                g.savefig(output[0])


def get_all_metamers(wildcards, comp=['energy_ref', 'energy_met', 'energy_ref-nat', 'energy_met-nat',
                                      'luminance_ref', 'luminance_met', 'energy_ref_downsample'],
                     include_gamma=True):
    from foveated_metamers import stimuli
    mets = {}
    if 'energy_ref' in comp:
        mets['energy_ref'] = utils.generate_metamer_paths('V1_norm_s6_gaussian')
    if 'energy_met' in comp:
        mets['energy_met'] = utils.generate_metamer_paths('V1_norm_s6_gaussian', comp='met')
    if 'energy_ref-nat' in comp:
        mets['energy_ref-nat'] = utils.generate_metamer_paths('V1_norm_s6_gaussian', comp='ref-natural')
    if 'energy_met-nat' in comp:
        mets['energy_met-nat'] = utils.generate_metamer_paths('V1_norm_s6_gaussian', comp='met-natural')
    if 'luminance_ref' in comp:
        mets['luminance_ref'] = utils.generate_metamer_paths('RGC_norm_gaussian')
    if 'luminance_met' in comp:
        mets['luminance_met'] = utils.generate_metamer_paths('RGC_norm_gaussian', comp='met')

    if 'energy_ref_downsample' in comp:
        imgs = []
        for i in range(3):
            imgs.extend(stimuli.get_images_for_session('sub-00', i, True))
        mets['energy_ref_downsample'] = utils.generate_metamer_paths('V1_norm_s6_gaussian',
                                                                     comp='met-downsample-2',
                                                                     image_name=imgs)
    if include_gamma:
        gamma_mets = {}
        for k, v in mets.items():
            gamma_mets[k+'_gamma'] = [img.replace('metamer.png', 'metamer_gamma-corrected.png')
                                      for img in v]
        mets.update(gamma_mets)
    return mets


rule rearrange_metamers_for_osf:
    input:
        # the first wildcards is just ignored, needed so we can also call
        # get_all_metamers the way rearrange_metamers_for_browser does it
        unpack(lambda wildcards: get_all_metamers(wildcards, [wildcards.share_str], False))
    output:
        op.join(config['DATA_DIR'], 'to_share', 'metamers_{share_str}.tar.gz')
    run:
        import json
        import shutil
        import tarfile
        import foveated_metamers as fov
        ln_path_template = ('{model_path_name}/{target_image}/scaling-{scaling}/'
                            'seed-{random_seed}_init-{initialization_type}.png')
        # this is just a temporary directory that we'll delete once we've
        # created the .tar.gz file
        output_dir = output[0].replace('.tar.gz', '')
        metadata = fov.utils.rearrange_metamers_for_sharing(input, output_dir, ln_path_template)
        with open(op.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        with tarfile.open(output[0], 'w:gz') as tar:
            # we use the arcname argument to get just the relative directory
            # structure (e.g., 'metamers_energy_ref/' instead of
            # 'mnt/ceph/users/wbroderick/metamers/to_share/metamers_energy_ref/')
            tar.add(output_dir, arcname=op.split(output_dir)[-1])
        shutil.rmtree(output_dir)


rule rearrange_synthesis_inputs_for_osf:
    input:
        ref_images = lambda wildcards: [utils.get_ref_image_full_path(img) for img in IMAGES],
        stats = op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture_degamma_norm_stats.pt'),
        # the full tiffs
        tiffs = lambda wildcards: [utils.get_ref_image_full_path(img).replace('_symmetric', '').replace('.png', '.tiff')
                                   for img in LINEAR_IMAGES],
    output:
        op.join(config['DATA_DIR'], 'to_share', 'synthesis_input.tar.gz'),
        op.join(config['DATA_DIR'], 'to_share', 'tiff_images.tar.gz'),
    run:
        import tarfile
        import json
        import os
        import shutil
        tmp_dirs = [op.join(op.dirname(output[0]), *d) for d in [('synthesis_input', 'ref_images_preproc'),
                                                                 ('synthesis_input', 'norm_stats'),
                                                                 ('tiff_images', 'ref_images')]]
        for d in tmp_dirs:
            os.makedirs(d)
        output_dir = op.join(op.dirname(output[0]), 'synthesis_input')
        metadata = []
        for inp in input.ref_images:
            outp = op.join(tmp_dirs[0], op.split(inp)[-1])
            metadata.append({'file': op.sep.join(outp.split(op.sep)[-2:]),
                             'target_image': op.split(inp)[-1].split('_')[0],
                             'gamma': False})
            os.link(inp, outp)
        with open(op.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        os.link(input.stats, op.join(tmp_dirs[1], op.split(input.stats)[-1]))
        with tarfile.open(output[0], 'w:gz') as tar:
            # we use the arcname argument to get just the relative directory
            # structure (e.g., 'synthesis_input/' instead of
            # 'mnt/ceph/users/wbroderick/metamers/to_share/synthesis_input/')
            tar.add(output_dir, arcname='synthesis_input')

        shutil.rmtree(output_dir)
        output_dir = op.join(op.dirname(output[0]), 'tiff_images')
        metadata = []
        for inp in input.tiffs:
            outp = op.join(tmp_dirs[-1], op.split(inp)[-1])
            metadata.append({'file': op.sep.join(outp.split(op.sep)[-2:]),
                             'target_image': op.split(inp)[-1].split('.')[0],
                             'gamma': False})
            os.link(inp, outp)
        with open(op.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        with tarfile.open(output[-1], 'w:gz') as tar:
            tar.add(output_dir, arcname='tiff_images')
        shutil.rmtree(output_dir)


def get_osf_names(wildcards, to_upload):
    model_name = {'energy': 'V1_norm_s6_gaussian', 'luminance': 'RGC_norm_gaussian'}[wildcards.osf_model_name]
    comp = wildcards.osf_comp.replace('-nat', '-natural').replace('_downsample', '-downsample-2')
    if to_upload == 'stimuli':
        return [op.join(config['DATA_DIR'], 'stimuli', model_name, fn) for fn in [f'stimuli_comp-{comp}.npy', f'stimuli_description_comp-{comp}.csv']]
    elif to_upload == 'mcmc':
        return op.join(config["DATA_DIR"], 'mcmc', model_name, 'task-split_comp-{comp}',
                       'task-split_comp-{comp}_mcmc_{mcmc_model}_{hyper}_scaling-extended.nc').format(mcmc_model=wildcards.mcmc_model,
                                                                                                      comp=comp,
                                                                                                      hyper=utils.get_mcmc_hyperparams({'mcmc_model': wildcards.mcmc_model,
                                                                                                                                        'model_name': model_name,
                                                                                                                                  'comp': comp}))
    elif to_upload == 'mcmc_compare':
        return op.join(config["DATA_DIR"], 'mcmc', model_name, 'task-split_comp-{comp}', 'task-split_comp-{comp}_mcmc_compare_ic-loo.csv').format(comp=comp)


rule rearrange_stimuli_for_osf:
    input:
        lambda wildcards: get_osf_names(wildcards, 'stimuli')
    output:
        # use a different name for this wildcards, because comp has a limited set
        # of values, that are different than the ones we use for upload
        op.join(config['DATA_DIR'], 'to_share', 'stimuli_{osf_model_name}_{osf_comp}.tar.gz')
    run:
        import shutil
        import tarfile
        import os
        # this is just a temporary directory that we'll delete once we've
        # created the .tar.gz file
        output_dir = output[0].replace('.tar.gz', '')
        os.makedirs(output_dir)
        for inp in input:
            outp = op.join(output_dir, op.split(inp)[-1])
            os.link(inp, outp)
        with tarfile.open(output[0], 'w:gz') as tar:
            tar.add(output_dir, arcname=op.split(output_dir)[-1])
        shutil.rmtree(output_dir)


rule rearrange_expt_training_for_osf:
    input:
        [[op.join(config['DATA_DIR'], 'stimuli', 'training_{model}', 'stimuli_comp-{comp}.npy').format(model=m, comp=c),
          op.join(config['DATA_DIR'], 'stimuli', 'training_{model}', 'stimuli_description_comp-{comp}.csv').format(model=m, comp=c),
          op.join(config['DATA_DIR'], 'stimuli', 'training_{model}', 'task-split_comp-{comp}', 'sub-training', 'sub-training_task-split_comp-{comp}_idx_sess-00_run-00.npy').format(model=m, comp=c),
          op.join(config['DATA_DIR'], 'stimuli', 'training_{model}', 'task-split_comp-{comp}', 'sub-training', 'sub-training_task-split_comp-{comp}_sess-00_run-00_correct-responses.npy').format(model=m, comp=c)]
         for m in ['noise'] + MODELS for c in {'noise': ['ref']}.get(m, ['ref', 'met'])]
    output:
        op.join(config['DATA_DIR'], 'to_share', 'experiment_training.tar.gz')
    run:
        import shutil
        import tarfile
        import os
        # this is just a temporary directory that we'll delete once we've
        # created the .tar.gz file
        output_dir = output[0].replace('.tar.gz', '') + os.sep
        os.makedirs(output_dir)
        for inp in input:
            outp = inp.replace(config["DATA_DIR"], output_dir)
            outp_dir = op.split(outp)[0]
            os.makedirs(outp_dir, exist_ok=True)
            os.link(inp, outp)
        with tarfile.open(output[0], 'w:gz') as tar:
            tar.add(output_dir, arcname=op.split(output_dir)[-1])
        shutil.rmtree(output_dir)


rule rearrange_behavioral_data_for_osf:
    input:
        [op.join(config['DATA_DIR'], 'behavioral', m, 'task-split_comp-{comp}', 'task-split_comp-{comp}_data.csv').format(comp=c)
         for m in MODELS
         for c in {'V1_norm_s6_gaussian': ['met', 'ref', 'met-natural', 'met-downsample-2', 'ref-natural']}.get(m, ['met', 'ref'])]
    output:
        op.join(config['DATA_DIR'], 'to_share', 'behavioral_data.tar.gz')
    run:
        import shutil
        import tarfile
        import os
        # this is just a temporary directory that we'll delete once we've
        # created the .tar.gz file
        output_dir = output[0].replace('.tar.gz', '')
        os.makedirs(output_dir)
        for inp in input:
            model, comp = re.findall('behavioral/([A-Z0-9]+)_norm.*_comp-([a-z-0-9]+)/task', inp)[0]
            model = {'V1': 'energy', 'RGC': 'luminance'}[model]
            comp = comp.replace('-natural', '-nat').replace('-downsample-2', '_downsample')
            outp = op.join(output_dir, f'{model}_comp-{comp}_data.csv')
            os.link(inp, outp)
        with tarfile.open(output[0], 'w:gz') as tar:
            tar.add(output_dir, arcname=op.split(output_dir)[-1])
        shutil.rmtree(output_dir)


rule rearrange_mcmc_model_fits_for_osf:
    input:
        lambda wildcards: get_osf_names(wildcards, 'mcmc')
    output:
        op.join(config['DATA_DIR'], 'to_share', 'mcmc_{osf_model_name}_{osf_comp}_{mcmc_model}.nc')
    run:
        import os
        os.link(input[0], output[0])


rule rearrange_figure_input_for_osf:
    input:
        [op.join(config['DATA_DIR'], 'mad_images', 'ssim_max', 'RGC_norm_gaussian_comp-ref_scaling-0.01_ref-bike_range-.05,.95_size-2048,2600_synth-ivy_range-.05,.95_size-2048,2600', 'opt-Adam_tradeoff-None_penalty-1e0_stop-iters-50', 'seed-0_lr-0.1_iter-10000_stop-crit-1e-9_gpu-1_mad.npy'),
         op.join(config['DATA_DIR'], 'mad_images', 'ssim_max', 'RGC_norm_gaussian_comp-ref_scaling-0.01_ref-bike_range-.05,.95_size-2048,2600_synth-white', 'opt-Adam_tradeoff-None_penalty-1e0_stop-iters-50', 'seed-0_lr-0.1_iter-10000_stop-crit-1e-9_gpu-1_mad.npy'),
         op.join(config['DATA_DIR'], 'mad_images', 'ssim_min', 'RGC_norm_gaussian_comp-ref_scaling-0.01_ref-bike_range-.05,.95_size-2048,2600_synth-white', 'opt-Adam_tradeoff-None_penalty-1e0_stop-iters-50', 'seed-0_lr-0.1_iter-10000_stop-crit-1e-9_gpu-1_mad.npy'),
         op.join(config['DATA_DIR'], 'synth_match_mse', 'RGC_norm_gaussian_comp-ref', 'bike_range-.05,.95_size-2048,2600_init-ivy_range-.05,.95_size-2048,2600_scaling-0.01_dir-None_lr-1e-9_max-iter-200_seed-0.npy'),
         op.join(config['DATA_DIR'], 'synth_match_mse', 'RGC_norm_gaussian_comp-ref', 'bike_range-.05,.95_size-2048,2600_init-white_scaling-0.01_dir-None_lr-1e-9_max-iter-200_seed-0.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'bike_range-.05,.95_size-2048,2600', 'scaling-0.01', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-0_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'gnarled_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-511500_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'gnarled_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-511501_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'ivy_range-.05,.95_size-2048,2600', 'scaling-0.01', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-420000_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'ivy_range-.05,.95_size-2048,2600', 'scaling-0.058', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-420700_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.23', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-41100_init-lettuce_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.23', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-41100_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.23', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-41101_init-nyc_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.23', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-41101_init-white_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'RGC_norm_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.23', 'opt-Adam_loss-mse', 'fr-0_lc-1_lt-1e-04_li-50_cf-False_clamp-True', 'seed-41102_init-terraces_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-5000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'gnarled_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1511400_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'gnarled_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1511401_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'ivy_range-.05,.95_size-2048,2600', 'scaling-0.063', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1420900_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'ivy_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1420500_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.063', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1040900_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'llama_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-0_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'nyc_range-.05,.95_size-2048,2600', 'scaling-0.063', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1430900_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-0_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'nyc_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1430500_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'portrait_symmetric_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1240500_init-bike_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'portrait_symmetric_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1240500_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'portrait_symmetric_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1240501_init-tiles_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'portrait_symmetric_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1240501_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'portrait_symmetric_range-.05,.95_size-2048,2600', 'scaling-0.27', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1240502_init-highway_symmetric_range-.05,.95_size-2048,2600_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'tiles_downsample-2_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1011400_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'tiles_downsample-2_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1011401_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'tiles_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1011400_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'metamers', 'V1_norm_s6_gaussian', 'tiles_range-.05,.95_size-2048,2600', 'scaling-1.5', 'opt-Adam-SWA_s-50_f-5_loss-mse_range-0,1_beta-0.5', 'fr-0_lc-1_lt-1e-04_li-50_cf-together_clamp-False', 'seed-1011401_init-white_lr-0.01_e0-0.500_em-26.800_iter-15000_thresh-1e-09_gpu-1_metamer.npy'),
         op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.csv'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-0.01_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-0.058_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-0.063_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-0.23_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-0.27_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),
         op.join(config['DATA_DIR'], 'windows_cache', 'scaling-1.5_size-2048,2600_e0-0.500_em-26.8_w-1.0_gaussian_single.pt'),]

    output:
        op.join(config['DATA_DIR'], 'to_share', 'figure_input.tar.gz')
    run:
        import shutil
        import tarfile
        import os
        # this is just a temporary directory that we'll delete once we've
        # created the .tar.gz file
        output_dir = output[0].replace('.tar.gz', '') + os.sep
        os.makedirs(output_dir)
        for inp in input:
            outp = inp.replace(config["DATA_DIR"], output_dir)
            outp_dir = op.split(outp)[0]
            os.makedirs(outp_dir, exist_ok=True)
            os.link(inp, outp)
        with tarfile.open(output[0], 'w:gz') as tar:
            tar.add(output_dir, arcname=op.split(output_dir)[-1])
        shutil.rmtree(output_dir)


rule upload_to_osf:
    input:
        '.osfcli.config',
        op.join(config['DATA_DIR'], 'to_share', '{to_share}'),
    output:
        op.join(config['DATA_DIR'], 'to_share', 'osf_upload', '{to_share}.txt'),
    run:
        import subprocess
        from datetime import datetime
        if wildcards.to_share.startswith('metamers_'):
            upload_path = f'metamers/{wildcards.to_share}'
        elif wildcards.to_share.startswith('stimuli_'):
            upload_path = f'stimuli/{wildcards.to_share}'
        elif wildcards.to_share.startswith('mcmc_'):
            upload_path = f'mcmc/{wildcards.to_share}'
        else:
            upload_path = wildcards.to_share
        subprocess.call(['osf', 'upload', '-Uf', input[1], f'osfstorage/{upload_path}'])
        with open(output[0], 'w') as f:
            f.write(f'Uploaded to {upload_path} at {datetime.now()}')


# these files are only required for the MCMC comparison, in the appendix, and
# make the overall project too large for the OSF. Instead, we use NYU's RCS,
# which we'll then upload the faculty data archive
rule copy_mcmc_compare_to_rcs:
    input:
        lambda wildcards: get_osf_names(wildcards, 'mcmc')
    output:
        op.join(config['RCS_DIR'], 'mcmc_{osf_model_name}_{osf_comp}_{mcmc_model}.nc')
    run:
        import shutil
        # because these are two different filesystems, we copy rather than link
        shutil.copy(input[0], output[0])


rule copy_mcmc_compare_csv_to_rcs:
    input:
        lambda wildcards: get_osf_names(wildcards, 'mcmc_compare')
    output:
        op.join(config['RCS_DIR'], 'mcmc_{osf_model_name}_{osf_comp}_compare_ic-loo.csv')
    run:
        import shutil
        # because these are two different filesystems, we copy rather than link
        shutil.copy(input[0], output[0])


rule copy_all_mcmc_compare_to_rcs:
    input:
        [op.join(config['RCS_DIR'], 'mcmc_{osf_model_name}_{osf_comp}_{mcmc_model}.nc').format(osf_model_name=m,
                                                                                               osf_comp=c,
                                                                                               mcmc_model=mc)
         for m in ['energy', 'luminance'] for mc in ['unpooled', 'partially-pooled-interactions']
         for c in {'energy': ['met', 'ref', 'met-nat', 'ref-nat', 'met_downsample'], 'luminance': ['ref', 'met']}[m]],
        [op.join(config['RCS_DIR'], 'mcmc_{osf_model_name}_{osf_comp}_compare_ic-loo.csv').format(osf_model_name=m,
                                                                                                  osf_comp=c)
         for m in ['energy', 'luminance']
         for c in {'energy': ['met', 'ref', 'met-nat', 'ref-nat', 'met_downsample'], 'luminance': ['ref', 'met']}[m]]


def create_checksum_input(wildcards):
    # then these are shared on the RCS, not the OSF, and so the input should be
    # the file found in the mcmc directory
    if 'unpooled' in wildcards.filename or 'interactions' in wildcards.filename:
        m, c, mc = re.findall('mcmc_([a-z]+)_([a-z-_]+)_([a-z-]+).nc', wildcards.filename)[0]
        wildcards.osf_model_name = m
        wildcards.osf_comp = c
        wildcards.mcmc_model = mc
        return get_osf_names(wildcards, 'mcmc')
    elif 'compare_ic' in wildcards.filename:
        m, c = re.findall('mcmc_([a-z]+)_([a-z-_]+)_compare_ic-loo.csv', wildcards.filename)[0]
        wildcards.osf_model_name = m
        wildcards.osf_comp = c
        return get_osf_names(wildcards, 'mcmc_compare')
    else:
        return op.join(config["DATA_DIR"], 'to_share', '{filename}')


rule create_checksum:
    input:
        create_checksum_input
    output:
        op.join(config["DATA_DIR"], 'to_share', 'checksums', '{filename}_checksum.json')
    run:
        import hashlib
        import json
        with open(input[0], 'rb') as f:
            checksum = hashlib.blake2b(f.read())
        with open(output[0], 'w') as f:
            json.dump({wildcards.filename: checksum.hexdigest()}, f)


rule create_checksum_dict:
    input:
        [op.join(config['DATA_DIR'], 'to_share', 'checksums', f"{f}_checksum.json") for f in
         ['behavioral_data.tar.gz', 'figure_input.tar.gz', 'synthesis_input.tar.gz', 'freeman_check.tar.gz', 'freeman_check_inputs.tar.gz',
          'mcmc_luminance_ref_partially-pooled.nc', 'mcmc_luminance_met_partially-pooled.nc', 'mcmc_energy_ref_partially-pooled.nc', 'mcmc_energy_ref-nat_partially-pooled.nc',
          'mcmc_energy_met_partially-pooled.nc', 'mcmc_energy_met-nat_partially-pooled.nc', 'mcmc_energy_met_downsample_partially-pooled.nc',
          'stimuli_luminance_ref.tar.gz', 'stimuli_luminance_met.tar.gz', 'stimuli_energy_ref.tar.gz', 'stimuli_energy_ref-nat.tar.gz',
          'stimuli_energy_met.tar.gz', 'stimuli_energy_met-nat.tar.gz', 'stimuli_energy_met_downsample.tar.gz', 'experiment_training.tar.gz',
          "mcmc_energy_met_compare_ic-loo.csv", "mcmc_energy_met_downsample_compare_ic-loo.csv", "mcmc_energy_met_downsample_partially-pooled-interactions.nc",
          "mcmc_energy_met_downsample_unpooled.nc", "mcmc_energy_met-nat_compare_ic-loo.csv", "mcmc_energy_met-nat_partially-pooled-interactions.nc",
          "mcmc_energy_met-nat_unpooled.nc", "mcmc_energy_met_partially-pooled-interactions.nc", "mcmc_energy_met_unpooled.nc",
          "mcmc_energy_ref_compare_ic-loo.csv", "mcmc_energy_ref-nat_compare_ic-loo.csv", "mcmc_energy_ref-nat_partially-pooled-interactions.nc",
          "mcmc_energy_ref-nat_unpooled.nc", "mcmc_energy_ref_partially-pooled-interactions.nc", "mcmc_energy_ref_unpooled.nc",
          "mcmc_luminance_met_compare_ic-loo.csv", "mcmc_luminance_met_partially-pooled-interactions.nc", "mcmc_luminance_met_unpooled.nc",
          "mcmc_luminance_ref_compare_ic-loo.csv", "mcmc_luminance_ref_partially-pooled-interactions.nc"]]
    output:
        op.join('data', 'checksums.json')
    run:
        import json
        checksums = {}
        for inp in input:
            with open(inp) as f:
                tmp = json.load(f)
            if len(tmp) > 1:
                raise Exception(f"Expected one file but found {len(tmp)}!")
            # there's only one key, so this grabs it
            if list(tmp.keys())[0] in checksums.keys():
                raise Exception(f"{tmp.keys()[0]} already found in checksums dict!")
            checksums.update(tmp)
        with open(output[0], 'w') as f:
            json.dump(checksums, f)


rule rearrange_metamers_for_browser:
    input:
        unpack(get_all_metamers),
    output:
        # this is hard-coded, because we're only doing it on simons cluster
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/metamer_metadata.json'
    run:
        import json
        import foveated_metamers as fov
        ln_path_template = ('{model_path_name}/{target_image}/downsample-{downsampled}/scaling-{scaling}/'
                            'seed-{random_seed}_init-{initialization_type}_gamma-{gamma_corrected}.png')
        output_dir = op.dirname(output[0])
        metadata = fov.utils.rearrange_metamers_for_sharing(input, output_dir, ln_path_template)
        with open(output[0], 'w') as f:
            json.dump(metadata, f)


rule rearrange_natural_imgs_for_browser:
    input:
        lambda wildcards: [utils.get_ref_image_full_path(img) for img in IMAGES],
        lambda wildcards: [utils.get_ref_image_full_path(utils.get_gamma_corrected_ref_image(img))
                           for img in IMAGES],
    output:
        # this is hard-coded, because we're only doing it on simons cluster
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/azulejos_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/tiles_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/bike_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/graffiti_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/llama_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/terraces_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/treetop_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/grooming_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/palm_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/leaves_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/portrait_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/troop_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/quad_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/highway_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/ivy_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/nyc_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/rocks_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/boats_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/gnarled_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/lettuce_gamma-False.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/azulejos_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/tiles_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/bike_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/graffiti_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/llama_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/terraces_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/treetop_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/grooming_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/palm_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/leaves_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/portrait_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/troop_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/quad_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/highway_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/ivy_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/nyc_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/rocks_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/boats_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/gnarled_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_images/lettuce_gamma-True.png',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_metadata.json',
    run:
        import os
        import json
        metadata = []
        for inp, outp in zip(input, output):
            os.link(inp, outp)
            tgt_img, gamma = re.findall('([a-z]+)_gamma-([A-Za-z]+).png', outp)[0]
            ln_path = outp.replace('/mnt/ceph/users/wbroderick/foveated_metamers_to_share/', '')
            metadata.append({'file': ln_path, 'target_image': tgt_img, 'gamma': bool(gamma)})
        with open(output[-1], 'w') as f:
            json.dump(metadata, f)


rule combine_browser_metadata:
    input:
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/metamer_metadata.json',
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/natural_metadata.json',
    output:
        '/mnt/ceph/users/wbroderick/foveated_metamers_to_share/metadata.json',
    run:
        import json
        metadata = {}
        with open(input[0]) as f:
            metadata['metamers'] = json.load(f)
        with open(input[1]) as f:
            metadata['natural_images'] = json.load(f)
        with open(output[0], 'w') as f:
            json.dump(metadata, f)


def figure_paper_input(wildcards):
    inputs = [
        # main figures
        op.join(config['DATA_DIR'], 'figures', 'paper', "image_space_human.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', "image_space_models.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'model_schematic_halfwidth_ivy_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'metamer_comparison_ivy_scaling-.01,.058,.063,.27_cutout_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', "ref_images_dpi-300.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', "performance_comparison_scaling-extended_partially-pooled_log-ci_comp-base.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'metamer_comparison_gnarled_scaling-1.5,1.5,1.5,1.5_cutout_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'performance_scaling-extended_metamer_comparison_nyc,llama_scaling-.063,.27_nocutout_small_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', "performance_comparison_scaling-extended_partially-pooled_log-ci_sub-00_comp-natural_line-scaling-0.27.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'metamer_comparison_portrait_symmetric_scaling-.27,.27,.27,.27,.27_cutout_V1_natural-seed_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', "image_space_ideal.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', "metamer_asymmetry.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', "critical_scaling_norm-False.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', "image_space_results.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', "image_space_lum.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'psychophys_expt2_with_table.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', "max_dprime_asymp_perf.svg"),
        # appendix figures
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'metamer_comparison_llama_scaling-.23,.23,.23,.23,.23_cutout_RGC_natural-seed_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'Dacey1992_mcmc_line-nooffset_linear.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', "freeman_windows_comparison.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', "performance_comparison_scaling-extended_partially-pooled_log-ci_sub-00_comp-downsample.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'metamer_comparison_tiles_scaling-1.5,1.5,1.5,1.5_cutout_downsample_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'mse_comparison_bike_scaling-0.01_nocutout_with_target_dpi-300.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', "radial_se_comp-ref_ecc-None.svg"),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'all_comps_summary_scaling-extended_partially-pooled_focus-outlier.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'all_comps_summary_scaling-extended_partially-pooled_focus-subject_one-ax.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', "mcmc_compare_ic-loo.svg"),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'V1_norm_s6_gaussian', 'task-split_comp-ref_mcmc_compare_psychophysical-grouplevel_log_yax-double.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'V1_norm_s6_gaussian_comp-ref_performance-all.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'RGC_norm_gaussian', 'task-split_comp-ref_mcmc_compare_psychophysical-grouplevel_log_yax-double.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'RGC_norm_gaussian_comp-ref_performance-all.svg'),
        op.join(config['DATA_DIR'], 'figures', 'paper', 'V1_norm_s6_gaussian', 'task-split_comp-met_mcmc_compare_psychophysical-grouplevel_log_yax-double.svg'),
        op.join(config['DATA_DIR'], 'compose_figures', 'paper', 'V1_norm_s6_gaussian_comp-met_performance-all.svg'),
        # txt files used to get numbers in the paper
        op.join(config['DATA_DIR'], 'statistics', 'critical_scaling.txt'),
        op.join(config['DATA_DIR'], 'statistics', 'number_of_stats.txt'),
    ]
    appendix_n_figs = {3: 3, 4: 2, 5: 2, 6: 7}
    outputs = (['fig-{:02d}.svg'.format(i) for i in range(1, 18)] +
               ['fig-A{:01d}-{:02d}.svg'.format(i, j) for i in range(1, 7)
                for j in range(1, appendix_n_figs.get(i, 1)+1)] +
               ['critical_scaling.txt', 'number_of_stats.txt'])
    mapping = dict(zip(outputs, inputs))
    mapping.update({k.replace('.svg', '_no_embed.svg'): v.replace('_dpi-300', '') for k, v in mapping.items()
                    if 'dpi-300' in v})
    return mapping[wildcards.fig_name]


rule figure_paper:
    input:
        figure_paper_input,
    output:
        os.path.join('reports', 'paper_figures', '{fig_name}'),
    run:
        import shutil
        shutil.copy(input[0], output[0])


rule main_paper_figures_no_embed:
    input:
        [op.join('reports', 'paper_figures', 'fig-{:02d}{}.svg').format(i, '_no_embed' if i in [3, 4, 5, 7, 8, 10] else '')
         for i in range(1, 18)],
        op.join('reports', 'paper_figures', 'critical_scaling.txt'),
        op.join('reports', 'paper_figures', 'number_of_stats.txt'),


rule appendix_figures_no_embed:
    input:
        [op.join('reports', 'paper_figures', 'fig-A{:01d}-{:02d}{}.svg').format(i, j, '_no_embed' if (i, j) in [(1, 1), (3, 3), (4, 1)] else '')
         for i in range(1, 6)
         for j in range(1, {3: 3, 4: 2, 5: 2}.get(i, 1)+1)]


rule main_paper_figures:
    input:
        [op.join('reports', 'paper_figures', 'fig-{:02d}.svg').format(i)
         for i in range(1, 18)],
        op.join('reports', 'paper_figures', 'critical_scaling.txt'),
        op.join('reports', 'paper_figures', 'number_of_stats.txt'),


rule appendix_figures:
    input:
        [op.join('reports', 'paper_figures', f'fig-A{i:01d}-{j:02d}.svg') for i in range(1, 6)
         for j in range(1, {3: 3, 4: 2, 5: 2}.get(i, 1)+1)]


rule appendix_figures_mcmc_compare:
    input:
        [op.join('reports', 'paper_figures', f'fig-A6-{j:02d}.svg') for j in range(1, 8)]
