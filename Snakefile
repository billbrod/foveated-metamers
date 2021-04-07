import os
import math
import re
import imageio
import time
import os.path as op
import numpy as np
from foveated_metamers import utils
import numpyro
import multiprocessing


configfile:
    "config.yml"
if not op.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
if os.system("module list") == 0:
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
    img_preproc="full|degamma|gamma-corrected|gamma-corrected_full|range-[,.0-9]+|gamma-corrected_range-[,.0-9]+",
    # einstein for testing setup, fountain for comparing with Freeman and Simoncelli, 2011 metamers
    preproc_image_name="|".join([im+'_?[a-z]*' for im in config['IMAGE_NAME']['ref_image']])+"|einstein|fountain",
    preproc="|_degamma|degamma",
    gpu="0|1",
    sess_num="|".join([f'{i:02d}' for i in config['PSYCHOPHYSICS']['SESSIONS']]),
    run_num="|".join([f'{i:02d}' for i in config['PSYCHOPHYSICS']['RUNS']]),
    comp='met|ref',
    save_all='|_saveall',
    gammacorrected='|_gamma-corrected'
ruleorder:
    collect_training_metamers > collect_training_noise > collect_metamers > demosaic_image > preproc_image > crop_image > generate_image > degamma_image > create_metamers > download_freeman_check

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
METAMER_LOG_PATH = METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'logs/metamers/{model_name}').replace('_metamer.png', '.log')
CONTINUE_TEMPLATE_PATH = (METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'metamers_continue/{model_name}')
                          .replace("{clamp_each_iter}/", "{clamp_each_iter}/attempt-{num}_iter-{extra_iter}"))
CONTINUE_LOG_PATH = CONTINUE_TEMPLATE_PATH.replace('metamers_continue/{model_name}', 'logs/metamers_continue/{model_name}').replace('_metamer.png', '.log')
TEXTURE_DIR = config['TEXTURE_DIR']
if TEXTURE_DIR.endswith(os.sep) or TEXTURE_DIR.endswith('/'):
    TEXTURE_DIR = TEXTURE_DIR[:-1]
if len(os.listdir(TEXTURE_DIR)) <= 800 and 'textures-subset-for-testing' not in TEXTURE_DIR:
    raise Exception(f"TEXTURE_DIR {TEXTURE_DIR} is incomplete!")

BEHAVIORAL_DATA_DATES = {
    'V1_norm_s6_gaussian': {
        'ref': {
            'sub-00': {'sess-00': '2021-Mar-23', 'sess-01': '2021-Mar-24', 'sess-02': '2021-Mar-24'},
            'sub-01': {'sess-00': '2021-Mar-30', 'sess-01': '2021-Mar-30', 'sess-02': '2021-Apr-01'},
            'sub-03': {'sess-00': '2021-Apr-02', 'sess-01': '2021-Apr-07', },
            'sub-04': {'sess-00': '2021-Apr-05', 'sess-01': '2021-Apr-06', },
        },
        'met': {
            'sub-00': {'sess-00': '2021-Apr-05', 'sess-01': '2021-Apr-07',},
        },
    },
    'RGC_norm_gaussian': {
        'ref': {
            'sub-00': {'sess-00': '2021-Apr-02', 'sess-01': '2021-Apr-06', 'sess-02': '2021-Apr-06'},
            'sub-02': {'sess-00': '2021-Apr-07',},
        }
    }
}


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
                shutil.copytree(op.dirname(input[0]), output[0])


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
                if 'gamma-corrected' in wildcards.img_preproc:
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
            if 'gaussian' in wildcards.model_name:
                if 'V1' in wildcards.model_name:
                    if float(wildcards.scaling) < .1:
                        mem = 128
                    else:
                        mem = 64
                if 'RGC' in wildcards.model_name:
                    # this is an approximation of the size of their windows,
                    # and if you have at least 3 times this memory, you're
                    # good. double-check this value -- the 1.36 is for
                    # converting form 2048,3528 (which the numbers came
                    # from) to 2048,2600 (which has 1.36x fewer pixels)
                    window_size = 1.17430726 / (1.36*float(wildcards.scaling))
                    mem = int(5 * window_size)
            if 'cosine' in wildcards.model_name:
                if 'V1' in wildcards.model_name:
                    # most it will need is 32 GB
                    mem = 32
                if 'RGC' in wildcards.model_name:
                    # this is an approximation of the size of their windows,
                    # and if you have at least 3 times this memory, you're
                    # good
                    window_size = 0.49238059 / float(wildcards.scaling)
                    mem = int(5 * window_size)
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
            # over requested memory)
            mem = ''
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
    if 'norm' in wildcards.model_name:
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
    window_template = op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}'
                              '_e0-{min_ecc:.03f}_em-{max_ecc:.01f}_w-{t_width}_{window_type}.pt')
    try:
        if 'size-' in wildcards.image_name:
            im_shape = wildcards.image_name[wildcards.image_name.index('size-') + len('size-'):]
            im_shape = im_shape.replace('.png', '')
            im_shape = [int(i) for i in im_shape.split(',')]
        else:
            try:
                im = imageio.imread(REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.image_name))
                im_shape = im.shape
            except FileNotFoundError:
                raise Exception("Can't find input image %s or infer its shape, so don't know what "
                                "windows to cache!" %
                                REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.image_name))
    except AttributeError:
        # then there was no wildcards.image_name, so grab the first one from
        # the DEFAULT_METAMERS list
        default_im = IMAGES[0]
        im_shape = default_im[default_im.index('size-') + len('size-'):]
        im_shape = im_shape.replace('.png', '')
        im_shape = [int(i) for i in im_shape.split(',')]
    try:
        max_ecc=float(wildcards.max_ecc)
        min_ecc=float(wildcards.min_ecc)
    except AttributeError:
        # then there was no wildcards.max/min_ecc, so grab the default values
        min_ecc = config['DEFAULT_METAMERS']['min_ecc']
        max_ecc = config['DEFAULT_METAMERS']['max_ecc']
    if 'cosine' in wildcards.model_name:
        window_type = 'cosine'
        t_width = 1.0
    elif 'gaussian' in wildcards.model_name:
        window_type = 'gaussian'
        t_width = 1.0
    if wildcards.model_name.startswith("RGC"):
        size = ','.join([str(i) for i in im_shape])
        return window_template.format(scaling=wildcards.scaling, size=size,
                                      max_ecc=max_ecc, t_width=t_width,
                                      min_ecc=min_ecc, window_type=window_type,)
    elif wildcards.model_name.startswith('V1'):
        windows = []
        # need them for every scale
        try:
            num_scales = int(re.findall('s([0-9]+)', wildcards.model_name)[0])
        except (IndexError, ValueError):
            num_scales = 4
        for i in range(num_scales):
            output_size = ','.join([str(int(np.ceil(j / 2**i))) for j in im_shape])
            windows.append(window_template.format(scaling=wildcards.scaling, size=output_size,
                                                  max_ecc=max_ecc,
                                                  min_ecc=min_ecc,
                                                  t_width=t_width, window_type=window_type))
        return windows

def get_partition(wildcards, cluster):
    # if our V1 scaling value is small enough, we need a V100 and must specify
    # it. otherwise, we can use any GPU, because they'll all have enough
    # memory. The partition name depends on the cluster (prince or rusty), so
    # we have two different params, one for each, and the cluster config grabs
    # the right one
    if cluster not in ['prince', 'rusty']:
        raise Exception(f"Don't know how to handle cluster {cluster}")
    if int(wildcards.gpu) == 0:
        if cluster == 'rusty':
            return 'ccn'
        elif cluster == 'prince':
            return None
    else:
        scaling = float(wildcards.scaling)
        if cluster == 'rusty':
            return 'gpu'
        elif cluster == 'prince':
            part = 'simoncelli_gpu,v100_sxm2_4,v100_pci_2'
            if scaling >= .18:
                part += ',p40_4'
            if scaling >= .4:
                part += ',p100_4'
            return part

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
        if float(wildcards.scaling) > .06:
            cpus = 21
        elif float(wildcards.scaling) > .03:
            cpus = 26
        else:
            cpus = 28
    return cpus


def get_init_image(wildcards):
    if wildcards.init_type in ['white', 'gray', 'pink', 'blue']:
        return []
    else:
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
        time = lambda wildcards: {'V1': '12:00:00', 'RGC': '7-00:00:00'}[wildcards.model_name.split('_')[0]],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        prince_partition = lambda wildcards: get_partition(wildcards, 'prince'),
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
        prince_partition = lambda wildcards: get_partition(wildcards, 'prince'),
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
                with fov.utils.get_gpu_id(get_gid) as gpu_id:
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
    scaling = [config[wildcards.model_name.split('_')[0]]['scaling'][0],
               config[wildcards.model_name.split('_')[0]]['scaling'][-1]]
    mets = utils.generate_metamer_paths(scaling=scaling, image_name=IMAGES[:2],
                                        seed_n=[0], **wildcards)
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
        lambda wildcards: [utils.get_ref_image_full_path(i) for i in IMAGES]
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
                                                                              int(wildcards.sess_num))
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
                    # then this is the test subject
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
                comp = 'met_v_' + wildcards.comp
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
    if wildcards.model_name == 'RGC_norm_gaussian' and wildcards.comp == 'met':
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
                '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}.hdf5'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}_expt.csv'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}', '{subject}',
                '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}_trials.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                '{subject}', '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}_expt.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                '{subject}', '{date}_{subject}_task-split_comp-{comp}_sess-{sess_num}_run-{run_num}_expt_benchmark.txt'),
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
                df = fov.analysis.add_response_info(df, trials, wildcards.subject,
                                                    wildcards.sess_num, wildcards.run_num)
                df.to_csv(output[0], index=False)


rule combine_all_behavior:
    input:
        lambda wildcards: [op.join(config["DATA_DIR"], 'behavioral', '{{model_name}}', 'task-split_comp-{{comp}}', '{subject}',
                                   '{date}_{subject}_task-split_comp-{{comp}}_{sess}_run-{i:02d}_expt.csv').format(
                                       i=i, sess=ses, date=date, subject=subj)
                           for i in range(5) for subj in BEHAVIORAL_DATA_DATES[wildcards.model_name][wildcards.comp]
                           for ses, date in BEHAVIORAL_DATA_DATES[wildcards.model_name][wildcards.comp][subj].items()],
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-{comp}.csv'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_data.csv'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_performance.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_run_lengths.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_plots.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_plots_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim_df = pd.read_csv(input[-1])
                expt_df = pd.concat([pd.read_csv(i) for i in input[:-1]])
                expt_df.to_csv(output[0], index=False)
                g = fov.figures.performance_plot(expt_df, hue='subject_name', comparison=wildcards.comp,
                                                 height=2.5)
                g.fig.savefig(output[1], bbox_inches='tight')
                g = fov.figures.run_length_plot(expt_df, hue='subject_name', comparison=wildcards.comp)
                g.fig.savefig(output[2], bbox_inches='tight')


# only make this plot for the ref comparison, see the comments of the function for why
rule plot_loss_performance_comparison:
    input:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_data.csv'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description_comp-ref.csv'),
    output:
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_loss_comparison.svg'),
        op.join(config["DATA_DIR"], 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_loss_comparison_subjects.svg'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_loss_comparison.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'behavioral', '{model_name}', 'task-split_comp-ref',
                'task-split_comp-ref_loss_comparison_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                stim_df = pd.read_csv(input[-1])
                expt_df = pd.read_csv(input[0])
                g = fov.figures.compare_loss_and_performance_plot(expt_df, stim_df)
                g.fig.savefig(output[0], bbox_inches='tight')
                g = fov.figures.compare_loss_and_performance_plot(expt_df, stim_df, col_wrap=None, row='subject_name')
                g.fig.savefig(output[1], bbox_inches='tight')


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
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}.nc'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}_benchmark.txt'),
    run:
        import contextlib
        import foveated_metamers as fov
        import pandas as pd
        import jax
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print(f"Running on {jax.lib.xla_bridge.device_count()} cpus!")
                if 'V1' in wildcards.model_name:
                    model = 'V1'
                elif 'RGC' in wildcards.model_name:
                    model = 'RGC'
                dataset = fov.mcmc.assemble_dataset_from_expt_df(pd.read_csv(input[0]))
                mcmc = fov.mcmc.run_inference(dataset, model,
                                              float(wildcards.step_size),
                                              int(wildcards.num_draws),
                                              int(wildcards.num_chains),
                                              int(wildcards.num_warmup),
                                              int(wildcards.seed))
                # want to have a different seed for constructing the inference
                # data object than we did for inference itself
                inf_data = fov.mcmc.assemble_inf_data(mcmc, dataset, int(wildcards.seed)+1)
                inf_data.to_netcdf(output[0])
                

rule mcmc_plots:
    input:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}.nc'),
    output:
        op.join(config["DATA_DIR"], 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}.png'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'mcmc', '{model_name}', 'task-split_comp-{comp}',
                'task-split_comp-{comp}_mcmc_step-{step_size}_c-{num_chains}_'
                'd-{num_draws}_w-{num_warmup}_s-{seed}_{plot_type}_benchmark.txt'),
    run:
        import foveated_metamers as fov
        import arviz as az
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                inf_data = az.from_netcdf(input[0])
                if wildcards.plot_type == 'post-pred-check':
                    print("Creating posterior predictive check.")
                    g = fov.figures.posterior_predictive_check(inf_data,
                                                               facetgrid_kwargs={'col': 'subject_name',
                                                                                 'row': 'image_name'})
                    fig = g.fig
                elif wildcards.plot_type == 'diagnostics':
                    print("Creating MCMC diagnostics plot.")
                    fig = fov.figures.mcmc_diagnostics_plot(inf_data)
                elif wildcards.plot_type == 'psychophysical-params':
                    print("Creating psychophysical parameters plot.")
                    g = fov.figures.psychophysical_parameters(inf_data, rotate_xticklabels=True, aspect=1.2)
                    fig = g.fig
                elif wildcards.plot_type == 'pairplot':
                    print("Creating parameter pairplot.")
                    g = fov.figures.parameter_pairplot(inf_data, hue='subject_name')
                    fig = g.fig
                elif wildcards.plot_type == 'distribs':
                    print("Creating parameter distribution plot.")
                    g = fov.figures.parameter_distributions(inf_data, row='subject_name')
                    fig = g.fig
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


rule window_example_figure:
    input:
        image = lambda wildcards: [m.replace('metamer.png', 'metamer_gamma-corrected.png') for m in
                                   utils.generate_metamer_paths(**wildcards)],
    output:
        report(op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                       '{image_name}_scaling-{scaling}_seed-{seed}_gpu-{gpu}_window.png'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_scaling-{scaling}_seed-{seed}_gpu-{gpu}_window.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_scaling-{scaling}_seed-{seed}_gpu-{gpu}_window_benchmark.txt')
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
    resources:
        mem = get_mem_estimate,
    run:
        import foveated_metamers as fov
        import seaborn as sns
        import contextlib
        import imageio
        import plenoptic as po
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = {'poster': 1.7}.get(wildcards.context, 1)
                min_ecc = config['DEFAULT_METAMERS']['min_ecc']
                max_ecc = config['DEFAULT_METAMERS']['max_ecc']
                with sns.plotting_context(wildcards.context, font_scale=font_scale):
                    image = fov.utils.convert_im_to_float(imageio.imread(input.image[0]))
                    # remove the normalizing aspect, since we don't need it here
                    model, _, _, _ = fov.create_metamers.setup_model(wildcards.model_name.replace('_norm', ''),
                                                                     float(wildcards.scaling),
                                                                     image, min_ecc, max_ecc, params.cache_dir)
                    fig = fov.figures.pooling_window_example(model.PoolingWindows, image)
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
                'task-split_comp-ref_data.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                'task-split_comp-{comp}_performance_benchmark.txt')
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
                g = fov.figures.performance_plot(expt_df, hue='subject_name', comparison=wildcards.comp,
                                                 height=fig_width/6)
                g.fig.savefig(output[0], bbox_inches='tight')


rule ref_image_figure:
    input:
        op.join(config["DATA_DIR"], 'stimuli', MODELS[1], 'stimuli_comp-ref.npy'),
        op.join(config["DATA_DIR"], 'stimuli', MODELS[1], 'stimuli_description_comp-ref.csv'),
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'ref_images.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'ref_images.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'ref_images_benchmark.txt')
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
                style, fig_width = fov.style.plotting_style(wildcards.context)
                plt.style.use(style)
                fig = fov.figures.ref_image_summary(stim, stim_df)
                fig.savefig(output[0], bbox_inches='tight')


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


rule compute_distances:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
        synth_images = lambda wildcards: [m.replace('.png', '-16.png') for m in
                                          utils.generate_metamer_paths(wildcards.synth_model_name,
                                                                       image_name=wildcards.image_name)],
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
        ref_image = po.load_images(input.ref_image)
        if input.norm_dict:
            norm_dict = torch.load(input.norm_dict)
        else:
            norm_dict = None
        model = fov.create_metamers.setup_model(wildcards.model_name, float(wildcards.scaling),
                                                ref_image, float(wildcards.min_ecc),
                                                float(wildcards.max_ecc), params.cache_dir,
                                                norm_dict)[0]
        synth_scaling = config[wildcards.synth_model_name.split('_')[0]]['scaling']
        df = []
        for sc in synth_scaling:
            df.append(fov.distances.model_distance(model, wildcards.synth_model_name,
                                                   wildcards.image_name, sc))
        df = pd.concat(df).reset_index(drop=True)
        df['distance_model'] = wildcards.model_name
        df['distance_scaling'] = float(wildcards.scaling)
        df.to_csv(output[0], index=False)


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


rule download_freeman_check_input:
    output:
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer1.png'),
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer2.png'),
        os.path.join(DATA_DIR, 'ref_images', 'fountain_size-512,512.png'),
    log:
        os.path.join(DATA_DIR, 'logs', 'freeman_check', 'download_input.log')
    benchmark:
        os.path.join(DATA_DIR, 'logs', 'freeman_check', 'download_input_benchmark.txt')
    params:
        met_dir_name = lambda wildcards, output: op.dirname(output[0]),
        ref_dir_name = lambda wildcards, output: op.dirname(output[-1]),
    shell:
        "curl -O -J -L https://osf.io/e2zn8/download; "
        "tar xf freeman_check_inputs.tar.gz; "
        "mv freeman_check_inputs/metamer1.png {params.met_dir_name}/; "
        "mv freeman_check_inputs/metamer2.png {params.met_dir_name}/; "
        "mv freeman_check_inputs/fountain_size-512,512.png {params.ref_dir_name}/; "
        "rm freeman_check_inputs.tar.gz; rmdir freeman_check_inputs;"


rule download_freeman_check:
    input:
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer1.png'),
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer2.png'),
        os.path.join(DATA_DIR, 'ref_images', 'fountain_size-512,512.png'),
    output:
        # unclear from paper what exact scaling was used.
        utils.generate_metamer_paths(model_name='V1_norm_s4_gaussian',
                                     image_name='fountain_size-512,512',
                                     scaling=[.3, .4, .5],
                                     max_ecc=13,
                                     gpu=1,
                                     seed=0),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.5', 'plotwindows.mat'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.25', 'plotwindows.mat'),
    log:
        os.path.join(DATA_DIR, 'logs', 'freeman_check', 'download.log')
    benchmark:
        os.path.join(DATA_DIR, 'logs', 'freeman_check', 'download_benchmark.txt')
    params:
        met_dir_name = op.join(DATA_DIR, 'metamers'),
        windows_dir_name = op.join(DATA_DIR, 'freeman_check', 'windows'),
    shell:
        "curl -O -J -L https://osf.io/wa2zu/download; "
        "tar xf freeman_check.tar.gz; "
        "rm freeman_check.tar.gz; "
        "cp -R V1_norm_s4_gaussian {params.met_dir_name}/; "
        "cp -R windows/* {params.windows_dir_name}/; "
        "rm -r V1_norm_s4_gaussian; "
        "rm -r windows; "


rule freeman_check:
    input:
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer1.png'),
        os.path.join(DATA_DIR, 'freeman_check', 'Freeman2011_metamers', 'metamer2.png'),
        os.path.join(DATA_DIR, 'ref_images', 'fountain_size-512,512.png'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.5', 'plotwindows.mat'),
        op.join(config['DATA_DIR'], 'freeman_check', 'windows', 'scaling-0.25', 'plotwindows.mat'),
        # unclear from paper what exact scaling was used.
        utils.generate_metamer_paths(model_name='V1_norm_s4_gaussian',
                                     image_name='fountain_size-512,512',
                                     scaling=[.3, .4, .5],
                                     max_ecc=13,
                                     gpu=1,
                                     seed=0),
