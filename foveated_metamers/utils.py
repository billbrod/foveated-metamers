"""various utilities
"""
import os
import torch
import re
import copy
import os.path as op
from glob import glob
import yaml
import argparse
import warnings
from contextlib import contextmanager
from itertools import cycle, product
import GPUtil
import numpy as np
from collections import OrderedDict
from . import plotting


def convert_im_to_float(im):
    r"""Convert image from saved data type to float

    Images are saved as either 8 or 16 bit integers, and for our
    purposes we generally want them to be floats that lie between 0 and
    1. In order to properly convert them, we divide the image by the
    maximum value its dtype can take (255 for 8 bit, 65535 for 16 bit).

    Note that for this to work, it should be called right after the
    image was loaded in; most manipulations will implicitly convert the
    image to a float, and then we cannot determine what to divide it by.

    Parameters
    ----------
    im : numpy array or imageio Array
        The image to convert

    Returns
    -------
    im : numpy array
        The converted image, now with dtype=np.float32 and all values
        between 0 and 1

    """
    return im / np.iinfo(im.dtype).max


def convert_im_to_int(im, dtype=np.uint8):
    r"""Convert image from float to 8 or 16 bit image

    We work with float images that lie between 0 and 1, but for saving
    them (either as png or in a numpy array), we want to convert them to
    8 or 16 bit integers. This function does that by multiplying it by
    the max value for the target dtype (255 for 8 bit 65535 for 16 bit)
    and then converting it to the proper type.

    We'll raise an exception if the max is higher than 1, in which case
    we have no idea what to do.

    Parameters
    ----------
    im : numpy array
        The image to convert
    dtype : {np.uint8, np.uint16}
        The target data type

    Returns
    -------
    im : numpy array
        The converted image, now with dtype=dtype

    """
    if im.max() > 1:
        if im.max() - 1 < 1e-4:
            warnings.warn("There was a precision/rounding error somewhere and im.max is "
                          f"{im.max()}. Setting that to 1 and converting anyway")
            im = np.clip(im, 0, 1)
        else:
            raise Exception("all values of im must lie between 0 and 1, but max is %s" % im.max())
    if im.min() < 0:
        if abs(im.min()) < 1e-4:
            warnings.warn("There was a precision/rounding error somewhere and im.min is "
                          f"{im.min()}. Setting that to 0 and converting anyway")
            im = np.clip(im, 0, 1)
        else:
            raise Exception("all values of im must lie between 0 and 1, but min is %s" % im.min())
    return (im * np.iinfo(dtype).max).astype(dtype)


@contextmanager
def get_gpu_id(get_gid=True, n_gpus=4, on_cluster=False):
    """get next available GPU and lock it

    Note that the lock file created will be at
    /tmp/LCK_gpu_{allocated_gid}.lock

    This is based on the solution proposed at
    https://github.com/snakemake/snakemake/issues/281#issuecomment-610796104
    and then modified slightly

    Parameters
    ----------
    get_gid : bool, optional
        if True, return the ID of the first available GPU. If False,
        return None. This weirdness is to allow us to still use this
        contextmanager when we don't actually want to create a lockfile
    n_gpus : int, optional
        number of GPUs on this device
    on_cluster : bool, optional
        whether we're on a cluster or not. if so, then we just return the gid
        for the first available GPU, since the job scheduler has taken care of
        this for us. We don't use dotlockfile in this case

    Returns
    -------
    allocated_gid : int
        the ID of the GPU to use

    """
    allocated_gid = None
    if not get_gid:
        avail_gpus = []
    else:
        avail_gpus = GPUtil.getAvailable(order='memory', maxLoad=.1, maxMemory=.1,
                                         includeNan=False, limit=n_gpus)
    for gid in cycle(avail_gpus):
        # just grab first gpu in this case
        if on_cluster:
            allocated_gid = gid
            break
        # then we've successfully created the lockfile
        if os.system(f"dotlockfile -r 1 /tmp/LCK_gpu_{gid}.lock") == 0:
            allocated_gid = gid
            break
    try:
        yield allocated_gid
    finally:
        os.system(f"dotlockfile -u /tmp/LCK_gpu_{allocated_gid}.lock")


def _find_img_size(image_name):
    """use regex to grab size from image_name

    We look for the pattern "_size-([0-9]+,[0-9]+)", grab the first one,
    and will raise an exception if no matches are found

    Parameters
    ----------
    image_name : str
        name of the reference image, as used in the Snakefile. will contain the
        base image, any preprocessing (degamma, changing the range), and size

    Returns
    -------
    image_size : np.array
        array with 2 ints, giving the size of the image

    """
    image_size = re.findall("_size-([0-9]+,[0-9]+)", image_name)[0]
    return np.array(image_size.split(',')).astype(int)


def find_attempts(wildcards, increment=False, extra_iter=None, gpu_split=.09):
    """Find most recently-generated metamer with specified wilcards.

    We allow for the possibility of continuing metamer synthesis, and so need a
    way to find the last synthesis attempt. This uses the wildcards dictionary
    (which specifies the desired metamer) and searches for the existing metamer
    with the highest attempt. Then, if `increment is False`, we return the path
    to that metamer (returning the original metamer path, as specified by
    METAMER_TEMPLATE_PATH if none can be found) or, if `increment is True`, we
    increment the attempt number by one (in this case, `extra_iter` must be an
    int, so we know how many extra iterations to add; if the original metamer
    path is not found, we'll return that).

    Parameters
    ----------
    wildcards : dict
        wildcards dictionary (as created by Snake make) whose keys are the
        format keys from the METAMER_TEMPLATE_PATH found in config.yml plus,
        potentially, `num` and `extra_iter`, which specify which continue
        attempt this is and how many iterations to add, respectively.
    increment : bool, optional
        Whether to return the most recently found metamer or increment attempt
        by one.
    extra_iter : int or None, optional
        If increment is True, this must be an int specifying how many extra
        iterations to add. If increment is False, this is ignored.

    Returns
    -------
    path : str
        path to the metamer.png file. see above for description.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    METAMER_TEMPLATE_PATH = defaults['METAMER_TEMPLATE_PATH']
    CONTINUE_TEMPLATE_PATH = (METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'metamers_continue/{model_name}')
                              .replace("{clamp_each_iter}/", "{clamp_each_iter}/attempt-{num}_iter-{extra_iter}/"))
    num = wildcards.pop('num', None)
    gpu = wildcards.pop('gpu', None)
    if gpu is None:
        if wildcards['scaling'] < gpu_split:
            gpu = 0
        else:
            gpu = 1
    wildcards['gpu'] = gpu
    wildcards.pop('extra_iter', None)
    wildcards['max_ecc'] = float(wildcards['max_ecc'])
    wildcards['min_ecc'] = float(wildcards['min_ecc'])
    i = 0
    while len(glob(CONTINUE_TEMPLATE_PATH.format(num=i, extra_iter='*', **wildcards))) > 0:
        i += 1
    # I would like to ensure that num is i, but to make the DAG we have
    # to go backwards and check each attempt, so this function does not
    # only get called for the rule the user calls
    if num is not None and int(num) > i:
        raise Exception("attempts at continuing metamers need to use strictly increasing num")
    if increment:
        if extra_iter is None:
            raise Exception("If increment is True, extra_iter must be an int, not None!")
        if i > 0:
            p = CONTINUE_TEMPLATE_PATH.format(num=i, extra_iter=extra_iter, **wildcards)
        else:
            if op.exists(METAMER_TEMPLATE_PATH.format(**wildcards)):
                p = CONTINUE_TEMPLATE_PATH.format(num=0, extra_iter=extra_iter, **wildcards)
            else:
                p = METAMER_TEMPLATE_PATH.format(**wildcards)
    else:
        if i > 0:
            p = glob(CONTINUE_TEMPLATE_PATH.format(num=i-1, extra_iter='*', **wildcards))[0]
        else:
            p = METAMER_TEMPLATE_PATH.format(**wildcards)
    # the next bit will remove all slashes from the string, so we need to
    # figure out whether we want to start with os.sep or not
    if p.startswith('/'):
        start = os.sep
    else:
        start = ''
    # this makes sure we're using the right os.sep and also removes any double
    # slashes we might have accidentally introduced
    return start + op.join(*p.split('/'))


def get_ref_image_full_path(image_name,
                            preproc_methods=['full', 'gamma-corrected',
                                             'range', 'degamma',
                                             'downsample'],
                            downsample=False):
    """check whether image is in ref_image or ref_image_preproc dir

    Parameters
    ----------
    image_name : str
        name of the (e.g., like those seen in `config.yml:
        DEFAULT_METAMERS: image_name`)
    preproc_methods : list, optional
        list of preproc methods we may have applied. probably shouldn't
        change this
    downsample : bool or int, optional
        whether we want the downsampled version of the ref images or not. If
        True, we downsample by 2. If an int, we downsample by that amount.

    Returns
    -------
    path : str
        full path to the reference image

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
        template = defaults['REF_IMAGE_TEMPLATE_PATH']
        DATA_DIR = defaults['DATA_DIR']
    if any([i in image_name for i in preproc_methods]):
        template = template.replace('ref_images', 'ref_images_preproc')
    if downsample:
        if downsample is True:
            downsample = 2
        if 'range' in image_name:
            image_name = image_name.replace('_ran', f'_downsample-{downsample}_ran')
        else:
            image_name += f'_downsample-{downsample}'
    template = template.format(image_name=image_name, DATA_DIR=DATA_DIR)
    # the next bit will remove all slashes from the string, so we need to
    # figure out whether we want to start with os.sep or not
    if template.startswith('/'):
        start = os.sep
    else:
        start = ''
    # this makes sure we're using the right os.sep and also removes any double
    # slashes we might have accidentally introduced
    return start + op.join(*template.split('/'))


def get_gamma_corrected_ref_image(image_name):
    """get name of gamma-corrected reference image

    Parameters
    ----------
    image_name : str
        name of the (e.g., like those seen in `config.yml:
        DEFAULT_METAMERS: image_name`)

    Returns
    -------
    path : str
        full path to the gamma-corrected reference image

    """
    image_name = image_name.split('_')
    target_i = 0
    # these two are special: if either are present, we want gamma-corrected to
    # be inserted right before it
    if any(['range' in i for i in image_name]):
        target_i = image_name.index(np.array(image_name)[['range' in i for i in image_name]])
        target_i -= 1
    elif any(['full' in i for i in image_name]):
        target_i = image_name.index(np.array(image_name)[['full' in i for i in image_name]])
        target_i -= 1
    image_name_target = []
    for part in [image_name[:(target_i+1)], ['gamma-corrected'], image_name[(target_i+1):]]:
        if not isinstance(part, list):
            part = [part]
        image_name_target += part
    return '_'.join(image_name_target)


def generate_image_names(ref_image=None, preproc=None, size=None):
    """Generate image names in a programmatic way

    This generates image names, to be passed to
    generate_metamer_paths. This is much more constrained than that
    template-following, as we only allow three fields: ref_image,
    preproc, and size. The template path is `IMAGE_NAME: template`, as
    found in config.yml

    Parameters
    ----------
    ref_image, preproc, size : str, list, or None, optional
        values to fill in for the three allowed fields. If None, we use
        the default, as found in `confing.yml: IMAGE_NAME`. If multiple
        values, we'll do a product, getting all possible combinations of
        the fields

    Returns
    -------
    image_names : list
        list of generated image names

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)['IMAGE_NAME']
    kwargs = OrderedDict({'ref_image': ref_image, 'preproc': preproc, 'size': size})
    template = defaults['template']
    for k, v in kwargs.items():
        if v is None:
            v = defaults[k]
        if not isinstance(v, list):
            v = [v]
        kwargs[k] = v
    image_names = []
    for vals in product(*kwargs.values()):
        tmp = dict(zip(kwargs.keys(), vals))
        im = template.format(**tmp)
        im = op.join(*im.split('/'))
        image_names.append(im)
    return image_names


def generate_metamer_seeds_dict(model_name):
    """Generate dictionary giving mapping to random seed.

    For each (model_name, image, scaling), we want unique seeds. This generates a
    dictionary that does that in a reasonable manner: the model must be
    specified, and it has (image, scaling) as keys (based on the config.yml
    file), with lists of n_seeds possible seeds as values.

    This should work well with additional models, images, scaling values, or
    seeds (as long as the ordering isn't changed) up to 100 of each.

    Parameters
    ----------
    model_name : {'RGC', 'V1'}
        Name(s) of the model to run.

    Returns
    -------
    seeds : dict
        Dict of seeds, see above for structure.

    """
    # separate each model_name by 1 million, image_name by 10k, each scaling value
    # by 100, which allows us to have up to 100 images, 100 scaling vaules, 100
    # seeds.
    model_name_sep = 1000000
    image_name_sep = 10000
    scaling_sep = 100
    n_seeds = 100
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    image_names = defaults['DEFAULT_METAMERS']['image_name']
    met_v_met = defaults[model_name].get('met_v_met_scaling', [])
    scaling = defaults[model_name]['scaling'] + met_v_met
    seeds = {}
    model_name_base = {'RGC': 0, 'V1': 1}[model_name] * model_name_sep

    def _get_fixed_idx_dict(vals, kind='image'):
        if kind == 'image':
            fixed_idx = defaults['FIXED_IMAGE_IDX'].copy()
        elif kind == 'scaling':
            fixed_idx = defaults['FIXED_SCALING_IDX'][model_name].copy()
        # we now loop through the values we want to show and, if they're not
        # already in fixed_idx, give them the lowest index that is not already
        # used.
        i = 0
        for v in vals:
            if v not in fixed_idx.keys():
                while i in fixed_idx.values():
                    i += 1
                fixed_idx[v] = i
                i += 1
        return fixed_idx

    # the fixed_idx dict gives us a specific image index for a set of images or
    # scaling values (whose metamers were already generated using that value,
    # back when I was looking at a broader set of images / different set of
    # scaling values).
    fixed_image_idx = _get_fixed_idx_dict(image_names, 'image')
    fixed_scaling_idx = _get_fixed_idx_dict(scaling, 'scaling')
    for im in image_names:
        image_base = fixed_image_idx[im] * image_name_sep
        for sc in scaling:
            scaling_base = fixed_scaling_idx[sc] * scaling_sep
            if ((im in defaults['OLD_SEEDS']['image_names']
                 and sc in defaults['OLD_SEEDS']['scaling'][model_name])):
                seed = [k for k in defaults['OLD_SEEDS']['seeds']]
                seed += [model_name_base + image_base + scaling_base + k for k
                         in range(len(seed), n_seeds)]
            else:
                seed = [model_name_base + image_base + scaling_base + k for k
                        in range(n_seeds)]
            seeds[(im, sc)] = seed
    return seeds


def generate_natural_init(image_name, scaling):
    """Generate the natural images to use for initialization in comp=ref-natural and met-natural.

    Pick 3 other natural images to use for initializing this metamer. Seed is
    based on image_name and scaling.

    Parameters
    ----------
    image_name : str
        name of the (e.g., like those seen in `config.yml:
        DEFAULT_METAMERS: image_name`).
    scaling : float
        Scaling value for this metamer.

    Returns
    -------
    natural_init : list
        Three natural images to use to initialize this metamer.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        config = yaml.safe_load(f)
    all_imgs = config['DEFAULT_METAMERS']['image_name']
    # img_seed will be between 0 and 19
    img_seed = all_imgs.index(image_name)
    # scaling has at most 3 places after the decimal, so scaling_seed will be
    # an integer between 100 (since smallest scaling right now .01) and 15000
    # (since highest is 1.5).
    scaling_seed = int(scaling*10000)
    # therefore every combination of img_seed and scaling_seed will be unique
    np.random.seed(scaling_seed+img_seed)
    all_imgs.pop(img_seed)
    return np.random.choice(all_imgs, 3, replace=False).tolist()


def generate_metamer_paths(model_name, increment=False, extra_iter=None,
                           gamma_corrected=False, comp='ref',
                           seed_n=None, **kwargs):
    """Generate metamer paths in a programmatic way

    This generates paths to the metamer.png files found in the
    metamer_display folder in a programmatic way. The intended use case
    is to call this script from the command-line and pipe the output to
    snakemake.

    We use the values found in config.yml, which give the metamers used
    in the experiment. To over-write them, and therefore generate a
    different set, you should pass extra `key=value` pairs, where `key`
    are the format strings from METAMER_TEMPLATE_PATH (found in
    config.yml) and `value` can be either a single value or a list. For
    any `key` not passed, we'll use the model-specific defaults from
    config.yml

    We will return the most recent attempt found `increment is False` and we
    will increment the attempt if it's True (in this case, `extra_iter` must be
    an int so we know what to put there).

    Parameters
    ----------
    model_name : str
        Name(s) of the model(s) to run. Must begin with either V1 or
        RGC. If model name is just 'RGC' or just 'V1', we will use the
        default model name for that brain area from config.yml
    increment : bool, optional
        Whether to return the most recently found metamer or increment attempt
        by one.
    extra_iter : int or None, optional
        If increment is True, this must be an int specifying how many extra
        iterations to add. If increment is False, this is ignored.
    gamma_corrected : bool, optional
        If True, return the path to the gamma-corrected version. If False, the
        non-gamma-corrected
    comp : {'ref', 'met', 'met-downsample-2', 'met-natural', 'ref-natural'}, optional
        If 'scaling' is not included in kwargs, this determines which range of
        default scaling values we use. If 'ref' (the defualt), we use those
        under the model:scaling key in the config file. If 'met' or
        'met-downsample-2', we look for model:met_v_met_scaling key; we use
        these plus the highest ones from model:scaling so that we end up with
        the same number of values. If there is no model:met_v_met_scaling key,
        we return the same values as before. If 'met-downsample-2', we
        downsample the reference images by a factor of 2 while leaving
        everything else the same (so physical pixel pitch is increased). If
        ref-natural, we use the same scaling range as ref, but initialize
        metamers with the ivy, grooming, and tiles reference images (and,
        unless specified, only do one seed per each) instead of 3 white noise
        seeds. If met-natural, we initialize with those reference images, but
        drop the smallest two scaling from the ref comparison and use the
        smallest two from met_v_met_scaling
    seed_n : list or None, optional
        List specifying which seeds to grab for each (model, image, scaling).
        If seed is in kwargs, this is ignored. If None (default), we use [0, 1,
        2]
    kwargs :
        keys must be configurable options for METAMER_TEMPLATE_PATH, as
        found in config.yml. If an option is *not* set, we'll use the
        model-specific default value from that yml file. values can be
        either a single value or a list

    Returns
    ----------
    paths : list
        list of strs, containing the absolute paths to the metamer.png
        files found in the metamer_display folder

    """
    if not isinstance(model_name, list):
        model_name = [model_name]
    if comp not in ['ref', 'met', 'met-downsample-2', 'met-natural', 'ref-natural', 'met-pink']:
        raise Exception("comp must be one of {'ref', 'met', 'met-downsample-2',"
                        " 'met-natural', 'ref-natural', 'met-pink'}!")
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    default_img_size = _find_img_size(defaults['DEFAULT_METAMERS']['image_name'][0])
    pix_to_deg = float(defaults['DEFAULT_METAMERS']['max_ecc']) / default_img_size.max()
    if comp.startswith('met') and any([m.startswith('RGC') for m in model_name]):
        imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
        warnings.warn("With RGC model and if comp starts with met, we use a reduced set of 5 default images!")
        default_ims = generate_image_names(imgs)
    elif comp.startswith('met-downsample'):
        default_ims = defaults['DEFAULT_METAMERS'].pop('image_name')
        default_ims = [im.replace('_range', '_' + comp.replace('met-', '') + '_range')
                       for im in default_ims]
        warnings.warn(f"With comp={comp}, we downsample the default images!")
    else:
        default_ims = defaults['DEFAULT_METAMERS'].pop('image_name')
    images = kwargs.pop('image_name', default_ims)
    if not isinstance(images, list):
        images = [images]
    args = {}
    paths = []
    for im in images:
        for model in model_name:
            args.update(copy.deepcopy(defaults['DEFAULT_METAMERS']))
            if 'max_ecc' not in kwargs.keys():
                img_size = _find_img_size(im)
                args['max_ecc'] = img_size.max() * pix_to_deg
            if model.startswith('RGC'):
                args.update(defaults['RGC'])
                if model == 'RGC':
                    model = defaults['RGC']['model_name']
                seeds_dict = generate_metamer_seeds_dict('RGC')
            elif model.startswith('V1'):
                args.update(defaults['V1'])
                if model == 'V1':
                    model = defaults['V1']['model_name']
                seeds_dict = generate_metamer_seeds_dict('V1')
            args['DATA_DIR'] = defaults['DATA_DIR']
            if 'scaling' not in kwargs.keys():
                scaling = defaults[model.split('_')[0]]['scaling']
                if comp.startswith('met') and 'natural' not in comp:
                    try:
                        more_scaling = defaults[model.split('_')[0]]['met_v_met_scaling']
                        scaling = scaling[-(len(scaling)-len(more_scaling)):] + more_scaling
                    except KeyError:
                        pass
                elif comp.startswith('met') and 'natural' in comp:
                    try:
                        more_scaling = defaults[model.split('_')[0]]['met_v_met_scaling'][:2]
                        scaling = scaling[-(len(scaling)-len(more_scaling)):] + more_scaling
                    except KeyError:
                        pass
            else:
                try:
                    scaling = float(kwargs['scaling'])
                except TypeError:
                    # then this is a list
                    scaling = [float(sc) for sc in kwargs['scaling']]
            if 'init' not in kwargs.keys():
                if 'natural' in comp:
                    args['init_type'] = 'natural'                    
                elif 'pink' in comp:
                    args['init_type'] = 'pink'
                else:
                    args['init_type'] = 'white'
            # by putting this last, we'll over-write the defaults
            args.update(kwargs)
            args.update({'model_name': model, 'image_name': im,
                         'scaling': scaling})
            if 'seed' not in args.keys():
                if seed_n is None:
                    if 'natural' in comp:
                        seed_n = ['index_0', 'index_1', 'index_2']
                    else:
                        seed_n = [0, 1, 2]
                args['seed_n'] = seed_n
            # remove this key if it's here. if it were included, it would
            # create duplicates of the paths
            args.pop('met_v_met_scaling', None)
            # we want to handle lists differently, iterating through them. we
            # use an ordered dict because we want the keys and values to have
            # the same ordering when we iterate through them separately
            list_args = OrderedDict({k: v for k, v in args.items() if isinstance(v, list)})
            for k in list_args.keys():
                args.pop(k)
            for vals in product(*list_args.values()):
                tmp = dict(zip(list_args.keys(), vals))
                tmp.update(args)
                if tmp.get('init_type', '') == 'natural':
                    init_type = generate_natural_init(tmp['image_name'], tmp['scaling'])
                    if 'seed_n' not in tmp.keys():
                        raise Exception("Cannot use comp=met-natural or ref-natural with "
                                        "seed set! (use seed_n instead)")
                    try:
                        init_idx_ = int(tmp['seed_n'].replace('index_', ''))
                    except AttributeError:
                        # then it's an int
                        init_idx_ = tmp['seed_n']
                    tmp['init_type'] = init_type[init_idx_]
                if 'seed_n' in tmp.keys():
                    try:
                        tmp_seed_n = tmp.pop('seed_n')
                        if isinstance(tmp_seed_n, str) and tmp_seed_n.startswith('index'):
                            tmp_seed_n = init_type.index(tmp['init_type'])
                        tmp['seed'] = seeds_dict[(tmp['image_name'].replace('_downsample-2', ''),
                                                  tmp['scaling'])][int(tmp_seed_n)]
                    except KeyError:
                        raise Exception(f"{tmp['image_name']} and {tmp['scaling']} (for model {model}) "
                                        "not found in the default set of metamers with pre-generated seeds"
                                        " -- please specify the seed argument")
                p = find_attempts(tmp, increment=increment, extra_iter=extra_iter,
                                  gpu_split=defaults['GPU_SPLIT'])
                if gamma_corrected:
                    p = p.replace('metamer.png', 'metamer_gamma-corrected.png')
                paths.append(p)
    return paths


def rearrange_metamers_for_sharing(file_dict, output_dir, ln_path_template):
    """Rearrange metamer .png files into convenient layout for sharing

    Currently, the metamer image files live within giant directories full of
    other files. To share (both for web browser and for tarfiles shared on
    OSF), we want just them, with some metadata. This file uses hardlinks
    (rather than copying) to rearrange into a new user-specified structure

    Parameters
    ----------
    file_dict : dict
        dictionary whose keys are strings of the form
        '{model}_{comparison}_{extra}', where model is one of {'energy',
        'luminance'}, comparison is in {'ref', 'met', 'ref-nat', 'met-nat'},
        and extra is in {'', '_downsample', '_gamma', '_downsample_gamma'}. the
        values are lists of files to include.
    output_dir : str
        string giving the path to the output directory, under which everything
        will be nested.
    ln_path_template : str
        python format str specifying the path for the new hardlinks. E.g.,
        '{model_path_name}/{target_image}/downsample-{downsampled}/scaling-{scaling}/
        seed-{random_seed}_init-{initialization_type}_gamma-{gamma_corrected}.png'.
        Can only contain the format keys given above (model_path_name is e.g.,
        "energy_model", rather than "Energy model").

    Returns
    -------
    metadata : list
        List of dictionaries giving metadata for each of these files (parsed
        from their original path; one dict per file). Intended to be saved as a
        metadata.json file.

    """
    comp_map = {'met': 'Synth vs. Synth: white noise', 'ref': 'Original vs. Synth: white noise',
                'met-nat': 'Synth vs. Synth: natural image', 'ref-nat': 'Original vs. Synth: natural image'}
    regex_str = ('metamers/metamers/([a-zA-z0-9_]+)/([a-z.0-9,-_]+?)/scaling-([0-9.]+)/.*?/.*?/'
                 'seed-([0-9]+)_init-([a-z.0-9,-_]+?)_lr')
    metadata = []
    for comp, data in file_dict.items():
        if 'downsample' in comp:
            downsampled = True
        else:
            downsampled = False
        if 'gamma' in comp:
            gamma_corrected = True
        else:
            gamma_corrected = False
        comp = comp_map[comp.split('_')[1]]
        for f in data:
            model, img, scaling, seed, init = re.findall(regex_str, f)[0]
            md = {'model_name': plotting.MODEL_PLOT[model], 'downsampled': downsampled,
                  'gamma_corrected': gamma_corrected, 'scaling': float(scaling),
                  'target_image': img.split('_')[0], 'random_seed': int(seed),
                  'initialization_type': init.split('_')[0], 'psychophysics_comparison': comp}
            model_path_name = md['model_name'].lower().replace(' ', '_')
            ln_path = ln_path_template.format(model_path_name=model_path_name, **md)
            md['file'] = ln_path
            # don't create hardlink if it's already there
            if not op.exists(op.join(output_dir, ln_path)):
                os.makedirs(op.join(output_dir, op.dirname(ln_path)), exist_ok=True)
                os.link(f, op.join(output_dir, ln_path))
            metadata.append(md)
    return metadata


def get_mcmc_hyperparams(wildcards, **kwargs):
    """Get hyperparameters for mcmc path string.

    Parameters
    ----------
    wildcards : dict
        Dictionary of snakemake wildcards.
    kwargs :
        Additional keyword=value pairs to specify comparison. keys can be one
        of: 'mcmc_model', 'model_name', 'comp'.

    Return
    ------
    hyper_str : str
        String of form 'step-{}_prob-{}_depth-{}_c-{}_d-{}_w-{}_s-{}'
        specifying hyperparameters for this specific comparison.

    """
    hyper_str = 'step-{}_prob-{}_depth-{}_c-{}_d-{}_w-{}_s-{}'
    kwargs.update(wildcards)
    if kwargs['mcmc_model'] == 'partially-pooled':
        if kwargs['model_name'] == 'V1_norm_s6_gaussian':
            if kwargs['comp'] == 'met':
                return hyper_str.format(1, '.8', 15, 4, 10000, 10000, 0)
        elif kwargs['model_name'] == 'RGC_norm_gaussian':
            if kwargs['comp'] == 'met':
                return hyper_str.format('.5', '.9', 20, 4, 15000, 15000, 0)
    elif kwargs['mcmc_model'] == 'partially-pooled-interactions':
        if kwargs['model_name'] == 'V1_norm_s6_gaussian':
            if kwargs['comp'] == 'met-natural':
                return hyper_str.format(1, '.9', 10, 4, 10000, 10000, 0)
            elif kwargs['comp'] == 'met':
                return hyper_str.format('.5', '.9', 10, 4, 15000, 15000, 1)
            elif kwargs['comp'] == 'met-downsample-2':
                return hyper_str.format('.5', '.8', 10, 4, 10000, 10000, 0)
            elif kwargs['comp'] == 'ref-natural':
                return hyper_str.format(1, '.8', 20, 4, 10000, 10000, 0)
        elif kwargs['model_name'] == 'RGC_norm_gaussian':
            if kwargs['comp'] == 'met':
                return hyper_str.format(1, '.95', 15, 4, 15000, 15000, 1)
    return hyper_str.format(1, '.8', 10, 4, 10000, 10000, 0)


def grab_single_window(windows, target_eccentricity=None, windows_scale=0):
    """Return single specified window.

    Parameters
    ----------
    windows : pooling.PoolingWindows
        The PoolingWindows object to grab the window from.
    target_eccentricity : float or None, optional
        The approximate central eccentricity of the window to grab. If None, we
        aim for 89.6% of the way out.
    windows_scale : int, optional
        The scale of the windows to grab.

    Returns
    -------
    window : torch.Tensor
        2d tensor containing the single window

    """
    if target_eccentricity is None:
        target_eccentricity = .896 * windows.max_eccentricity
    target_ecc_idx = abs(windows.central_eccentricity_degrees -
                         target_eccentricity).argmin()
    ecc_windows = (windows.ecc_windows[windows_scale] /
                   windows.norm_factor[windows_scale])
    return torch.einsum('hw,hw->hw',
                        windows.angle_windows[windows_scale][0],
                        ecc_windows[target_ecc_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Generate metamer paths in a programmatic way, for passing to snakemake. "
                     "With the exception of model_name, --increment and "
                     "--extra_iter all other arguments are the various configurable options from the "
                     "metamer template path, which control synthesis behavior. All arguments "
                     "can take multiple values, in which case we'll generate all possible "
                     "combinations. If a value is unset, we'll use the model-specific "
                     "defaults from config.yml."))

    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    template_path = defaults['METAMER_TEMPLATE_PATH']
    # this grabs the keys from the template path, including the optional format
    # strings (e.g., ':.03f', ':s'), but dropping those format strings to just
    # grab the possible args
    possible_args = [i[0] for i in re.findall('{([A-Za-z_]+?)(:[A-Za-z\d\.]+)?}', template_path)]
    parser.add_argument('--increment', '-i', action='store_true',
                        help=("Whether we should return the last found attempt or increment it "
                              "by one. If passed, --extra_iter must also be set"))
    parser.add_argument('--gamma_corrected', '-g', action='store_true',
                        help=("Whether we should return the gamma-corrected path or not."))
    parser.add_argument('--comp', '-c', default='ref',
                        help=("{ref, met, met-downsample-2, met-natural, ref-natural}, Whether to generate the scaling values for comparing "
                              "metamers to reference images, to other metamers, or to other metamers using "
                              "a downsampled ref image"))
    parser.add_argument('--seed_n', '-n', nargs='+', type=int, default=None,
                        help=(" List specifying which seeds to grab for each (model, image, "
                              "scaling). If seed is also passed, this is ignored."))
    parser.add_argument('--extra_iter', type=int,
                        help=("If --increment is passed, this specifies how many extra "
                              "iterations to run synthesis for"))
    for k in possible_args:
        nargs = {'DATA_DIR': 1}.get(k, '+')
        if k == 'model_name':
            parser.add_argument("model_name", nargs=nargs,
                                help=("Name(s) of the model(s) to run. Must begin with either "
                                      "V1 or RGC. If model name is just 'RGC' or just 'V1', we "
                                      "will use the default model name for that brain area from"
                                      f" config.yml ({defaults['RGC']['model_name']} or "
                                      f"{defaults['V1']['model_name']}, respectively)"))
        else:
            parser.add_argument(f"--{k}", default=None, nargs=nargs)
    for k in ['ref_image', 'size', 'preproc']:
        parser.add_argument(f'--{k}', default=None, nargs='+',
                            help=("These arguments are used to construct "
                                  "additional image_names using the template "
                                  f"{defaults['IMAGE_NAME']['template']}. Their defaults are in "
                                  "the IMAGE_NAME section of config.yml, and any image names "
                                  "constructed from them will be appended to those passed. If "
                                  "these are set and image_name is unset, will just use these"))
    args = vars(parser.parse_args())
    increment = args.pop('increment')
    extra_iter = args.pop('extra_iter')
    gamma_corrected = args.pop('gamma_corrected')
    image_kwargs = {k: args.pop(k) for k in ['ref_image', 'size', 'preproc']}
    if args['comp'].startswith('met') and any([m.startswith('RGC') for m in args['model_name']]):
        if image_kwargs['ref_image'] is None and args['image_name'] is None:
            imgs = ['llama', 'highway_symmetric', 'rocks', 'boats', 'gnarled']
            warnings.warn("With RGC model and if comp starts with met, we use a reduced set of 5 default images!")
            image_kwargs['ref_image'] = imgs
    elif args['comp'].startswith('met-downsample'):
        if image_kwargs['ref_image'] is None and args['image_name'] is None:
            with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
                defaults = yaml.safe_load(f)
            imgs = defaults['IMAGE_NAME'].pop('ref_image')
            imgs = [im + '_downsample-2' for im in imgs]
            warnings.warn(f"With comp={args['comp']}, we downsample the default images!")
            image_kwargs['ref_image'] = imgs
    images = generate_image_names(**image_kwargs)
    new_args = {}
    for k, v in args.items():
        if v is None:
            continue
        if k == 'DATA_DIR':
            new_args[k] = v
        elif k in ['seed', 'max_iter', 'gpu', 'loss_change_iter', 'seed_n']:
            # then it's an int
            new_args[k] = [int(vi) for vi in v]
        else:
            try:
                # everything else is a list. here we try and convert
                # each item to a float
                new_args[k] = [float(vi) for vi in v]
            except ValueError:
                # then it's a list of strings, and we keep it as is
                new_args[k] = v
    if 'image_name' not in new_args.keys():
        new_args['image_name'] = images
    elif 'image_name' in new_args.keys() and images and not all([v is None for v in image_kwargs.values()]):
        raise Exception("Must set either image_name or its components (ref_image, size, preproc)!")
    paths = generate_metamer_paths(increment=increment, extra_iter=extra_iter,
                                   gamma_corrected=gamma_corrected,
                                   **new_args)
    # need to do a bit of string manipulation to get this in the right
    # format
    paths = ' '.join(paths)
    print(paths)
