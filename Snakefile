import os
import re
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
    num="[0-9]+",
    pad_mode="constant|symmetric",
    period="[0-9]+",
    size="[0-9,]+",
    bits="[0-9]+",
    img_preproc="full|cone|cone_full|degamma_cone|gamma-corrected|gamma-corrected_full",
    preproc_image_name="azulejos|tiles|market|flower|einstein",
    preproc="|_degamma|_degamma_cone|_cone|degamma|degamma_cone|cone"
ruleorder:
    collect_metamers_example > collect_metamers > demosaic_image > preproc_image > crop_image > generate_image > degamma_image


LINEAR_IMAGES = ['azulejos', 'tiles', 'market', 'flower']
MODELS = ['RGC_cone-1.0_gaussian', 'V1_cone-1.0_norm_s6_gaussian']
IMAGES = ['azulejos_cone_full_size-2048,3528', 'tiles_cone_full_size-2048,3528',
          'market_cone_full_size-2048,3528', 'flower_cone_full_size-2048,3528']
METAMER_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'metamers', '{model_name}', '{image_name}',
                                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-'
                                '{loss_fract}_cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}',
                                'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-'
                                '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_'
                                'metamer.png')
OUTPUT_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'metamers_display', '{model_name}', '{image_name}',
                               'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-'
                               '{loss_fract}_cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}',
                               'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-'
                               '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_'
                               'metamer.png')
CONTINUE_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'metamers_continue', '{model_name}',
                                 '{image_name}', 'scaling-{scaling}', 'opt-{optimizer}',
                                 'fr-{fract_removed}_lc-{loss_fract}_cf-{coarse_to_fine}_{clamp}-'
                                 '{clamp_each_iter}', 'attempt-{num}_iter-{extra_iter}',
                                 'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-'
                                 '{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_'
                                 'metamer.png')
REF_IMAGE_TEMPLATE_PATH = op.join(config['DATA_DIR'], 'ref_images', '{image_name}.png')
SUBJECTS = ['sub-%02d' % i for i in range(1, 31)]
SESSIONS = [0, 1, 2]
RGC_SCALING = [.01, .013, .017, .021, .027, .035, .045, .058, .075]
V1_SCALING = [.075, .095, .12, .15, .19, .25, .31, .39, .5]
V1_LR_DICT = {.075: 1, .095: 1, .12: 1}
V1_ITER_DICT = {.075: 8500, .095: 7500, .12: 7500}
V1_GPU_DICT = {.075: 3, .095: 2, .12: 2}


def get_all_metamers(min_idx=0, max_idx=-1, model_name=None):
    rgc_metamers = [OUTPUT_TEMPLATE_PATH.format(model_name=MODELS[0], image_name=i, scaling=sc,
                                                optimizer='Adam', fract_removed=0, loss_fract=1,
                                                coarse_to_fine=0, seed=s, init_type='white',
                                                learning_rate=.1, min_ecc=3.71, max_ecc=41,
                                                max_iter=750, loss_thresh=1e-8, gpu=0,
                                                clamp='clamp', clamp_each_iter=True)
                    for sc in RGC_SCALING for i in IMAGES for s in range(3)]
    v1_metamers = [OUTPUT_TEMPLATE_PATH.format(model_name=MODELS[1], image_name=i, scaling=sc,
                                               optimizer='Adam', fract_removed=0, loss_fract=1,
                                               coarse_to_fine=1e-2, seed=s, init_type='white',
                                               learning_rate=V1_LR_DICT.get(sc, .1), min_ecc=.5,
                                               max_ecc=41, max_iter=V1_ITER_DICT.get(sc, 5000),
                                               loss_thresh=1e-8, gpu=V1_GPU_DICT.get(sc, 1),
                                               clamp='clamp', clamp_each_iter=True)
                    for sc in V1_SCALING for i in IMAGES for s in range(3)]
    if model_name is None:
        all_metamers = rgc_metamers + v1_metamers
    elif model_name == MODELS[0]:
        all_metamers = rgc_metamers
    elif model_name == MODELS[1]:
        all_metamers = v1_metamers
    else:
        raise Exception("model_name must be one of %s" % MODELS)
    # we use -1 as a dummy value, ignoring it
    if max_idx != -1:
        all_metamers = all_metamers[:max_idx]
    return all_metamers[min_idx:]


# quick rule to check that there are GPUs available and the environment
# has been set up correctly.
rule test_setup:
    input:
        METAMER_TEMPLATE_PATH.format(model_name=MODELS[0],
                                     image_name='einstein_degamma_cone_size-256,256',
                                     scaling=.1, optimizer='Adam', fract_removed=0, loss_fract=1,
                                     coarse_to_fine=0, seed=0, init_type='white',
                                     learning_rate=1, min_ecc=2, max_ecc=15,
                                     max_iter=100, loss_thresh=1e-8, gpu=0,
                                     clamp='clamp', clamp_each_iter=True),
        METAMER_TEMPLATE_PATH.format(model_name=MODELS[1],
                                     image_name='einstein_degamma_cone_size-256,256',
                                     scaling=.5, optimizer='Adam', fract_removed=0, loss_fract=1,
                                     coarse_to_fine=0.01, seed=0, init_type='white',
                                     learning_rate=.1, min_ecc=.5, max_ecc=15,
                                     max_iter=100, loss_thresh=1e-8, gpu=0,
                                     clamp='clamp', clamp_each_iter=True),
        METAMER_TEMPLATE_PATH.format(model_name=MODELS[0],
                                     image_name='einstein_degamma_cone_size-256,256',
                                     scaling=.1, optimizer='Adam', fract_removed=0, loss_fract=1,
                                     coarse_to_fine=0, seed=0, init_type='white',
                                     learning_rate=1, min_ecc=2, max_ecc=15,
                                     max_iter=100, loss_thresh=1e-8, gpu=1,
                                     clamp='clamp', clamp_each_iter=True),
        METAMER_TEMPLATE_PATH.format(model_name=MODELS[1],
                                     image_name='einstein_degamma_cone_size-256,256',
                                     scaling=.5, optimizer='Adam', fract_removed=0, loss_fract=1,
                                     coarse_to_fine=0.01, seed=0, init_type='white',
                                     learning_rate=.1, min_ecc=.5, max_ecc=15,
                                     max_iter=100, loss_thresh=1e-8, gpu=1,
                                     clamp='clamp', clamp_each_iter=True),
    output:
        directory(op.join(config['DATA_DIR'], 'test_setup', MODELS[0], 'einstein')),
        directory(op.join(config['DATA_DIR'], 'test_setup', MODELS[1], 'einstein'))
    log:
        op.join(config['DATA_DIR'], 'logs', 'test_setup.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'test_setup_benchmark.txt')
    run:
        import contextlib
        import shutil
        import os.path as op
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print("Moving outputs from %s to %s" % (op.dirname(input[0]), output[0]))
                shutil.move(op.dirname(input[0]), output[0])
                print("Moving outputs from %s to %s" % (op.dirname(input[1]), output[1]))
                shutil.move(op.dirname(input[1]), output[1])


rule all_refs:
    input:
        [op.join(config['DATA_DIR'], 'ref_images_preproc', i + '.png') for i in IMAGES],
        [op.join(config['DATA_DIR'], 'ref_images_preproc', i.replace('cone_', '') + '.png')
         for i in IMAGES],


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
        import foveated_metamers as met
        from skimage import color
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                # when loaded in, the range of this will be 0 to 255, we
                # want to convert it to 0 to 1
                im = met.utils.convert_im_to_float(im)
                # convert to grayscale
                im = color.rgb2gray(im)
                # 1/2.2 is the standard encoding gamma for jpegs, so we
                # raise this to its reciprocal, 2.2, in order to reverse
                # it
                im = im**2.2
                dtype = eval('np.uint%s' % wildcards.bits)
                imageio.imwrite(output[0], met.utils.convert_im_to_int(im, dtype))


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
        import foveated_metamers as met
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                curr_shape = np.array(im.shape)[:2]
                target_shape = [int(i) for i in wildcards.size.split(',')]
                if len(target_shape) == 1:
                    target_shape = 2* target_shape
                target_shape = np.array(target_shape)
                crop_amt = curr_shape - target_shape
                cropped_im = im[crop_amt[0]//2:-crop_amt[0]//2, crop_amt[1]//2:-crop_amt[1]//2]
                cropped_im = color.rgb2gray(cropped_im)
                imageio.imwrite(output[0], met.utils.convert_im_to_int(cropped_im, np.uint16))
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
        import foveated_metamers as met
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
                else:
                    print("Image will *not* use full dynamic range")
                    im = im / np.iinfo(dtype).max
                if 'cone' in wildcards.img_preproc:
                    print("Raising image to the 1/3, to approximate cone response")
                    im = im ** (1/3)
                if 'gamma-corrected' in wildcards.img_preproc:
                    print("Raising image to 1/2.2, to gamma correct it")
                    im = im ** (1/2.2)
                # always save it as 16 bit
                print("Saving as 16 bit")
                im = met.utils.convert_im_to_int(im, np.uint16)
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
        op.join(config['DATA_DIR'], 'ref_images', '{image_type}_period-{period}_size-{size}.png')
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

rule preproc_textures:
    input:
        config['TEXTURE_DIR']
    output:
        directory(config['TEXTURE_DIR'] + "_{preproc}")
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
        import foveated_metamers as met
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                os.makedirs(output[0])
                for i in glob(op.join(input[0], '*.jpg')):
                    im = imageio.imread(i)
                    im = met.utils.convert_im_to_float(im)
                    if im.ndim == 3:
                        # then it's a color image, and we need to make it grayscale
                        im = color.rgb2gray(im)
                    if 'degamma' in wildcards.preproc:
                        # 1/2.2 is the standard encoding gamma for jpegs, so we
                        # raise this to its reciprocal, 2.2, in order to reverse
                        # it
                        im = im ** 2.2
                    if 'cone' in wildcards.preproc:
                        im = im ** (1/3)
                    # save as a 16 bit png
                    im = met.utils.convert_im_to_int(im, np.uint16)
                    imageio.imwrite(op.join(output[0], op.split(i)[-1].replace('jpg', 'png')), im)


rule gen_norm_stats:
    input:
        config['TEXTURE_DIR'] + "{preproc}"
    output:
        # here V1 and texture could be considered wildcards, but they're
        # the only we're doing this for now
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm_stats-'
                '{num}.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm_'
                'stats-{num}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm_'
                'stats-{num}_benchmark.txt')
    params:
        index = lambda wildcards: (int(wildcards.num) * 100, (int(wildcards.num)+1) * 100)
    run:
        import plenoptic as po
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # scaling doesn't matter here
                if 'gamma' == wildcards.cone:
                    cone_power = 1/2.2
                elif 'phys' == wildcards.cone:
                    cone_power = 1/3
                else:
                    cone_power = float(wildcards.cone)
                v1 = po.simul.PrimaryVisualCortex(1, (512, 512), half_octave_pyramid=True,
                                                  num_scales=6, cone_power=cone_power,
                                                  include_highpass=True)
                po.simul.non_linearities.generate_norm_stats(v1, input[0], output[0], (512, 512),
                                                             index=params.index)


# we need to generate the stats in blocks, and then want to re-combine them
rule combine_norm_stats:
    input:
        lambda wildcards: [op.join(config['DATA_DIR'], 'norm_stats', 'V1_cone-{cone}_texture'
                                   '{preproc}_norm_stats-{num}.pt').format(num=i, **wildcards)
                           for i in range(9)]
    output:
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm_stats.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm_'
                'stats.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_cone-{cone}_texture{preproc}_norm'
                '_stats_benchmark.txt')
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
                'em-{max_ecc}_w-{t_width}_{window_type}.pt')
    log:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.benchmark.txt')
    run:
        import contextlib
        import plenoptic as po
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                img_size = [int(i) for i in wildcards.size.split(',')]
                if wildcards.window_type == 'cosine':
                    t_width = float(wildcards.t_width)
                    std_dev = None
                elif wildcards.window_type == 'gaussian':
                    std_dev = float(wildcards.t_width)
                    t_width = None
                po.simul.PoolingWindows(float(wildcards.scaling), img_size, float(wildcards.min_ecc),
                                        float(wildcards.max_ecc), cache_dir=op.dirname(output[0]),
                                        transition_region_width=t_width, std_dev=std_dev,
                                        window_type=wildcards.window_type)


def get_norm_dict(wildcards):
    if 'norm' in wildcards.model_name and 'V1' in wildcards.model_name:
        preproc = ''
        # lienar images should also use the degamma'd textures
        if 'degamma' in wildcards.image_name or any([i in wildcards.image_name for i in LINEAR_IMAGES]):
            preproc += '_degamma'
        if 'cone' in wildcards.image_name:
            preproc += '_cone'
        try:
            if 'cone-gamma' in wildcards.model_name:
                cone_power = 'gamma'
            elif 'cone-phys' in wildcards.model_name:
                cone_power = 'phys'
            else:
                cone_power = float(re.findall('cone-([.0-9]+)', wildcards.model_name)[0])
        except IndexError:
            # default is 1, linear response
            cone_power = 1
        return op.join(config['DATA_DIR'], 'norm_stats', 'V1_cone-%s_texture%s_norm_stats.pt'
                       % (cone_power, preproc))
    else:
        return []


def get_windows(wildcards):
    r"""determine the cached window path for the specified model
    """
    window_template = op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}'
                              '_e0-{min_ecc:.03f}_em-{max_ecc:.01f}_w-{t_width}_{window_type}.pt')
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
    if 'cosine' in wildcards.model_name:
        window_type = 'cosine'
        t_width = 1.0
    elif 'gaussian' in wildcards.model_name:
        window_type = 'gaussian'
        t_width = 1.0
    if wildcards.model_name.startswith("RGC"):
        size = ','.join([str(i) for i in im_shape])
        return window_template.format(scaling=wildcards.scaling, size=size,
                                      max_ecc=float(wildcards.max_ecc), t_width=t_width,
                                      min_ecc=float(wildcards.min_ecc), window_type=window_type,)
    elif wildcards.model_name.startswith('V1'):
        windows = []
        # need them for every scale
        try:
            num_scales = int(re.findall('s([0-9]+)', wildcards.model_name)[0])
        except (IndexError, ValueError):
            num_scales = 4
        for i in range(num_scales):
            output_size = ','.join([str(int(np.ceil(j / 2**i))) for j in im_shape])
            min_ecc, _ = pooling.calc_min_eccentricity(float(wildcards.scaling),
                                                       [np.ceil(j / 2**i) for j in im_shape],
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
                                                  min_ecc=min_ecc, t_width=t_width,
                                                  window_type=window_type))
        return windows


def get_batches(wildcards):
    if len(wildcards.gpu.split(':')) > 1:
        return int(wildcards.gpu.split(':')[1])
    else:
        return 1


def get_ref_image(image_name):
    r"""get ref image
    """
    if 'full' in image_name or 'cone' in image_name or 'gamma-corrected' in image_name:
        template = REF_IMAGE_TEMPLATE_PATH.replace('ref_images', 'ref_images_preproc')
    else:
        template = REF_IMAGE_TEMPLATE_PATH
    return template.format(image_name=image_name)


def get_mem_estimate(wildcards):
    r"""estimate the amount of memory that this will need, in GB
    """
    if 'size-2048,3528' in wildcards.image_name:
        if 'gaussian' in wildcards.model_name:
            if 'V1' in wildcards.model_name:
                if float(wildcards.scaling) >= .31:
                    return 16
                elif float(wildcards.scaling) >= .15:
                    return 32
                elif float(wildcards.scaling) >= .095:
                    return 64
                else:
                    return 96
            if 'RGC' in wildcards.model_name:
                # this is an approximation of the size of their windows,
                # and if you have at least 3 times this memory, you're
                # good
                window_size = 1.17430726 / float(wildcards.scaling)
                return int(3 * window_size)
        if 'cosine' in wildcards.model_name:
            if 'V1' in wildcards.model_name:
                # most it will need is 32 GB
                return 32
            if 'RGC' in wildcards.model_name:
                # this is an approximation of the size of their windows,
                # and if you have at least 3 times this memory, you're
                # good
                window_size = 0.49238059 / float(wildcards.scaling)
                return int(3 * window_size)
    else:
        # don't have a good estimate for these
        return 16


rule create_metamers:
    input:
        ref_image = lambda wildcards: get_ref_image(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
    output:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'metamer-16.png'),
        METAMER_TEMPLATE_PATH,
    log:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}', 'seed-{seed}_init-{init_type}_'
                'lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}'
                '_gpu-{gpu}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'metamers', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}', 'seed-{seed}_init-{init_type}_'
                'lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-{max_iter}_thresh-{loss_thresh}'
                '_gpu-{gpu}_benchmark.txt')
    resources:
        gpu = lambda wildcards: int(wildcards.gpu.split(':')[0]),
        mem = get_mem_estimate,
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        num_batches = get_batches,
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.clamp_each_iter == 'True':
                    clamp_each_iter = True
                elif wildcards.clamp_each_iter == 'False':
                    clamp_each_iter = False
                if wildcards.init_type not in ['white', 'blue', 'pink', 'gray']:
                    init_type = REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.init_type)
                else:
                    init_type = wildcards.init_type
                met.create_metamers.main(wildcards.model_name, float(wildcards.scaling),
                                         input.ref_image, int(wildcards.seed), float(wildcards.min_ecc),
                                         float(wildcards.max_ecc), float(wildcards.learning_rate),
                                         int(wildcards.max_iter), float(wildcards.loss_thresh),
                                         output[0], init_type, resources.gpu>0,
                                         params.cache_dir, input.norm_dict, resources.gpu,
                                         wildcards.optimizer, float(wildcards.fract_removed),
                                         float(wildcards.loss_fract),
                                         float(wildcards.coarse_to_fine), int(params.num_batches),
                                         wildcards.clamp, clamp_each_iter)


def find_attempts(wildcards):
    wildcards = dict(wildcards)
    num = wildcards.pop('num', None)
    wildcards.pop('extra_iter', None)
    i = 0
    while len(glob(CONTINUE_TEMPLATE_PATH.format(num=i, extra_iter='*', **wildcards))) > 0:
        i += 1
    # I would like to ensure that num is i, but to make the DAG we have
    # to go backwards and check each attempt, so this function does not
    # only get called for the rule the user calls
    if num is not None and int(num) > i:
        raise Exception("attempts at continuing metamers need to use strictly increasing num")
    if i > 0:
        return glob(CONTINUE_TEMPLATE_PATH.format(num=i-1, extra_iter='*', **wildcards))[0]
    else:
        return METAMER_TEMPLATE_PATH.format(**wildcards)


rule continue_metamers:
    input:
        ref_image = lambda wildcards: get_ref_image(wildcards.image_name),
        norm_dict = get_norm_dict,
        continue_path = lambda wildcards: find_attempts(wildcards).replace('_metamer.png', '.pt'),
    output:
        CONTINUE_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        CONTINUE_TEMPLATE_PATH.replace('metamer.png', 'metamer-16.png'),
        CONTINUE_TEMPLATE_PATH,
    log:
        op.join(config["DATA_DIR"], 'logs', 'metamers_continue', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}', 'attempt-{num}_iter-{extra_iter}',
                'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-'
                '{max_iter}_thresh-{loss_thresh}_gpu-{gpu}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'metamers_continue', '{model_name}', '{image_name}',
                'scaling-{scaling}', 'opt-{optimizer}', 'fr-{fract_removed}_lc-{loss_fract}_'
                'cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}', 'attempt-{num}_iter-{extra_iter}',
                'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-'
                '{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_benchmark.txt')
    resources:
        gpu = lambda wildcards: int(wildcards.gpu.split(':')[0]),
        mem = get_mem_estimate,
    params:
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        num_batches = get_batches,
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.clamp_each_iter == 'True':
                    clamp_each_iter = True
                elif wildcards.clamp_each_iter == 'False':
                    clamp_each_iter = False
                if wildcards.init_type not in ['white', 'blue', 'pink', 'gray']:
                    init_type = REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.init_type)
                else:
                    init_type = wildcards.init_type
                # this is the same as the original call in the
                # create_metamers rule, except we replace max_iter with
                # extra_iter, set learning_rate to None, and add the
                # input continue_path at the end
                met.create_metamers.main(wildcards.model_name, float(wildcards.scaling),
                                         input.ref_image, int(wildcards.seed), float(wildcards.min_ecc),
                                         float(wildcards.max_ecc), None,
                                         int(wildcards.extra_iter), float(wildcards.loss_thresh),
                                         output[0], init_type, resources.gpu>0,
                                         params.cache_dir, input.norm_dict, resources.gpu,
                                         wildcards.optimizer, float(wildcards.fract_removed),
                                         float(wildcards.loss_fract),
                                         float(wildcards.coarse_to_fine), int(params.num_batches),
                                         wildcards.clamp, clamp_each_iter, input.continue_path)


rule postproc_metamers:
    input:
        lambda wildcards: find_attempts(wildcards).replace('metamer.png', 'summary.csv'),
        lambda wildcards: find_attempts(wildcards).replace('metamer.png', 'synthesis.mp4'),
        lambda wildcards: find_attempts(wildcards).replace('metamer.png', 'rep.png'),
        lambda wildcards: find_attempts(wildcards).replace('metamer.png', 'windowed.png'),
        lambda wildcards: find_attempts(wildcards).replace('metamer.png', 'metamer-16.png'),
        lambda wildcards: find_attempts(wildcards),
    output:
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'summary.csv'),
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'synthesis.mp4'),
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'rep.png'),
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'windowed.png'),
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'metamer-16.png'),
        OUTPUT_TEMPLATE_PATH,
        OUTPUT_TEMPLATE_PATH.replace('metamer.png', 'metamer_gamma-corrected.png'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'postproc_metamers', '{model_name}',
                '{image_name}', 'scaling-{scaling}', 'opt-{optimizer}',
                'fr-{fract_removed}_lc-{loss_fract}_cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}',
                'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-'
                '{max_iter}_thresh-{loss_thresh}_gpu-{gpu}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'postproc_metamers', '{model_name}',
                '{image_name}', 'scaling-{scaling}', 'opt-{optimizer}',
                'fr-{fract_removed}_lc-{loss_fract}_cf-{coarse_to_fine}_{clamp}-{clamp_each_iter}',
                'seed-{seed}_init-{init_type}_lr-{learning_rate}_e0-{min_ecc}_em-{max_ecc}_iter-'
                '{max_iter}_thresh-{loss_thresh}_gpu-{gpu}_benchmark.txt')
    run:
        import foveated_metamers as met
        import contextlib
        import numpy as np
        import shutil
        import foveated_metamers as met
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                for i, f in enumerate(input):
                    if ('cone' in wildcards.image_name and
                        (f.endswith('metamer.png') or f.endswith('metamer-16.png'))):
                        print("De-conifying image %s, saving at %s" % (f, output[i]))
                        im = imageio.imread(f)
                        dtype = im.dtype
                        print("Retaining image dtype %s" % dtype)
                        im = met.utils.convert_im_to_float(im)
                        im = im ** 3
                        im = met.utils.convert_im_to_int(im, dtype)
                        imageio.imwrite(output[i], im)
                    else:
                        print("Copy file %s to %s" % (f, output[i]))
                        shutil.copy(f, output[i])
                    if f.endswith('metamer.png'):
                        if ('degamma' in wildcards.image_name or
                            any([i in wildcards.image_name for i in LINEAR_IMAGES])):
                            print("Saving gamma-corrected image %s" % output[-1])
                            im = imageio.imread(f)
                            dtype = im.dtype
                            print("Retaining image dtype %s" % dtype)
                            im = met.utils.convert_im_to_float(im)
                            # need to first de-cone the image, then
                            # gamma-correct it
                            if 'cone' in wildcards.image_name:
                                im = im ** 3
                            im = im ** (1/2.2)
                            im = met.utils.convert_im_to_int(im, dtype)
                            imageio.imwrite(output[-1], im)
                        else:
                            print("Image already gamma-corrected, copying to %s" % output[-1])
                            shutil.copy(f, output[-1])


rule dummy_metamer_gen:
    input:
        lambda wildcards: get_all_metamers(int(wildcards.min_idx), int(wildcards.max_idx),
                                           wildcards.model_name),
    output:
        op.join(config['DATA_DIR'], 'metamers_display', 'dummy_{model_name}_{min_idx}_{max_idx}.txt')
    shell:
        "touch {output}"


def get_metamers_for_example(wildcards):
    metamers = get_all_metamers(model_name=MODELS[0])
    return [m.replace('metamer.png', 'metamer-16.png') for m in metamers if 'market' in m
            if 'seed-2' not in m]


rule collect_metamers_example:
    # this is for a shorter version of the experiment, the goal is to
    # create a test version for teaching someone how to run the
    # experiment or for demos
    input:
        get_metamers_for_example,
        [get_ref_image(IMAGES[2].replace('cone_', ''))],
    output:
        op.join(config["DATA_DIR"], 'stimuli', 'RGC_demo', 'stimuli.npy'),
        op.join(config["DATA_DIR"], 'stimuli', 'RGC_demo', 'stimuli_description.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'RGC_demo', 'stimuli.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', 'RGC_demo', 'stimuli_benchmark.txt'),
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.collect_images(input, output[0])
                met.stimuli.create_metamer_df(input, output[1])


rule collect_metamers:
    input:
        lambda wildcards: [m.replace('metamer.png', 'metamer-16.png') for m in
                           get_all_metamers(**wildcards)],
        # we don't want the "cone_full" images, we want the "full"
        # images.
        lambda wildcards: [get_ref_image(i.replace('cone_', '')) for i in IMAGES]
    output:
        # we collect across image_name and scaling, and don't care about
        # learning_rate, max_iter, loss_thresh
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli.npy'),
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description.csv'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'stimuli.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', 'stimuli_benchmark.txt'),
    run:
        import foveated_metamers as met
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.collect_images(input, output[0])
                met.stimuli.create_metamer_df(input, output[1])


rule generate_experiment_idx:
    input:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', 'stimuli_description.csv'),
    output:
        op.join(config["DATA_DIR"], 'stimuli', '{model_name}', '{subject}_idx_sess-{num}.npy'),
    log:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', '{subject}_idx_sess-{num}'
                '.log'),
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'stimuli', '{model_name}', '{subject}_idx_sess-{num}'
                '_benchmark.txt'),
    params:
        # the number from subject will be a number from 1 to 30, which
        # we multiply by 10 in order to get the tens/hundreds place, and
        # the session number will be between 0 and 2, which we use for
        # the ones place. we use the same seed for different model
        # stimuli, since those will be completely different sets of
        # images.
        seed = lambda wildcards: 10*int(wildcards.subject.replace('sub-', '')) + int(wildcards.num)
    run:
        import foveated_metamers as met
        import pandas as pd
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                met.stimuli.generate_indices(pd.read_csv(input[0]), params.seed, output[0])


rule gen_all_idx:
    input:
        [op.join(config["DATA_DIR"], 'stimuli', '{model_name}', '{subject}_idx_sess-'
                 '{num}.npy').format(model_name=m, subject=s, num=n)
         for s in SUBJECTS for n in SESSIONS for m in MODELS],


rule scaling_comparison_figure:
    input:
        lambda wildcards: [m.replace('metamer.png', 'metamer_gamma-corrected.png') for m in
                           get_all_metamers(model_name=wildcards.model_name)
                           if wildcards.image_name in m if 'seed-%s' % wildcards.seed in m],
        lambda wildcards: get_ref_image(wildcards.image_name.replace('cone_', 'gamma-corrected_'))
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', '{model_name}',
                '{image_name}_seed-{seed}_scaling.svg')
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_seed-{seed}_scaling.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', '{model_name}',
                '{image_name}_seed-{seed}_scaling_benchmark.txt')
    run:
        import foveated_metamers as met
        import seaborn as sns
        import contextlib
        import re
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                font_scale = {'poster': 1.7}.get(wildcards.context, 1)
                template_path = input[0].replace(wildcards.image_name, '{image_name}')
                for key in ['seed', 'scaling']:
                    template_path = re.sub(f'{key}-[0-9.]+', f'{key}-{{{key}}}', template_path)
                max_ecc = int(re.findall('em-([0-9]+)', template_path)[0])
                ref_path = input[-1].replace(wildcards.image_name.replace('cone_', 'gamma-corrected_'),
                                             '{image_name}')
                with sns.plotting_context(wildcards.context, font_scale=font_scale):
                    if wildcards.model_name == MODELS[0]:
                        fig = met.figures.scaling_comparison_figure(
                            wildcards.image_name, RGC_SCALING, wildcards.seed, max_ecc=max_ecc,
                            ref_template_path=ref_path, metamer_template_path=template_path)
                    elif wildcards.model_name == MODELS[1]:
                        for key in ['lr', 'iter', 'gpu']:
                            template_path = re.sub(f'{key}-[0-9.]+', f'{key}-{{{key}}}',
                                                   template_path)
                        gpu_dict = V1_GPU_DICT.copy()
                        lr_dict = V1_LR_DICT.copy()
                        iter_dict = V1_ITER_DICT.copy()
                        for sc in V1_SCALING:
                            gpu_dict.setdefault(sc, 1)
                            lr_dict.setdefault(sc, .1)
                            iter_dict.setdefault(sc, 5000)
                        fig = met.figures.scaling_comparison_figure(
                            wildcards.image_name, V1_SCALING, wildcards.seed, gpu=gpu_dict,
                            lr=lr_dict, iter=iter_dict, max_ecc=max_ecc,
                            ref_template_path=ref_path, metamer_template_path=template_path)
                    fig.savefig(output[0], bbox_inches='tight')
