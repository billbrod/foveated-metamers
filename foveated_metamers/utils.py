"""various utilities
"""
import os
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


def find_attempts(wildcards, increment=False, extra_iter=None):
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
    # this makes sure we're using the right os.sep and also removes any double
    # slashes we might have accidentally introduced
    return os.sep + op.join(*p.split('/'))


def get_ref_image_full_path(image_name,
                            preproc_methods=['full', 'gamma-corrected',
                                             'range', 'degamma']):
    """check whether image is in ref_image or ref_image_preproc dir

    Parameters
    ----------
    image_name : str
        name of the (e.g., like those seen in `config.yml:
        DEFAULT_METAMERS: image_name`)
    preproc_methods : list, optional
        list of preproc methods we may have applied. probably shouldn't
        change this

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
    template = template.format(image_name=image_name, DATA_DIR=DATA_DIR)
    return os.sep + op.join(*template.split('/'))


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
    for i, im in enumerate(image_names):
        image_base = i * image_name_sep
        for j, sc in enumerate(scaling):
            scaling_base = j * scaling_sep
            if im in defaults['OLD_SEEDS']['image_names']:
                seed = [k for k in defaults['OLD_SEEDS']['seeds']]
                seed += [model_name_base + image_base + scaling_base + k for k
                         in range(len(seed), n_seeds)]
            else:
                seed = [model_name_base + image_base + scaling_base + k for k
                        in range(n_seeds)]
            seeds[(im, sc)] = seed
    return seeds


def generate_metamer_paths(model_name, increment=False, extra_iter=None,
                           gamma_corrected=False, comp='ref',
                           seed_n=[0, 1, 2], **kwargs):
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
    comp : {'ref', 'met'}, optional
        If 'scaling' is not included in kwargs, this determines which range of
        default scaling values we use. If 'ref' (the defualt), we use those
        under the model:scaling key in the config file. If 'met', we look for
        model:met_v_met_scaling key; we use these plus the highest ones from
        model:scaling so that we end up with 9 total values. If there is no
        model:met_v_met_scaling key, we return the same values as before.
    seed_n : list, optional
        List specifying which seeds to grab for each (model, image, scaling).
        If seed is in kwargs, this is ignored.
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
    if comp not in ['ref', 'met']:
        raise Exception("comp must be one of {'ref', 'met'}!")
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    default_img_size = _find_img_size(defaults['DEFAULT_METAMERS']['image_name'][0])
    pix_to_deg = float(defaults['DEFAULT_METAMERS']['max_ecc']) / default_img_size.max()
    images = kwargs.pop('image_name', defaults['DEFAULT_METAMERS'].pop('image_name'))
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
                if comp == 'met':
                    try:
                        more_scaling = defaults[model.split('_')[0]]['met_v_met_scaling']
                        scaling = scaling[-(9-len(more_scaling)):] + more_scaling
                    except KeyError:
                        pass
            else:
                scaling = kwargs['scaling']
            # by putting this last, we'll over-write the defaults
            args.update(kwargs)
            args.update({'model_name': model, 'image_name': im,
                         'scaling': scaling})
            if 'seed' not in args.keys():
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
                if 'seed_n' in tmp.keys():
                    try:
                        tmp['seed'] = seeds_dict[(tmp['image_name'], tmp['scaling'])][tmp.pop('seed_n')]
                    except KeyError:
                        raise Exception(f"{tmp['image_name']} and {tmp['scaling']} (for model {model}) "
                                        "not found in the default set of metamers with pre-generated seeds"
                                        " -- please specify the seed argument")
                p = find_attempts(tmp, increment=increment, extra_iter=extra_iter)
                if gamma_corrected:
                    p = p.replace('metamer.png', 'metamer_gamma-corrected.png')
                paths.append(p)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Generate metamer paths in a programmatic way, for passing to snakemake. "
                     "With the exception of model_name, --print, --save_path, --increment and "
                     "--extra_iter all other arguments are the various configurable options from the "
                     "metamer template path, which control synthesis behavior. All arguments "
                     "can take multiple values, in which case we'll generate all possible "
                     "combinations. If a value is unset, we'll use the model-specific "
                     "defaults from config.yml."))

    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    template_path = defaults['METAMER_TEMPLATE_PATH']
    possible_args = re.findall('{([A-Za-z_]+?)}', template_path)
    parser.add_argument('--print', '-p', action='store_true',
                        help="Print out the paths. Note either this or --save_path must be set")
    parser.add_argument('--save_path', '-s', default='',
                        help=("Path to a .txt file to save the paths at. If not set, will not "
                              "save. Note either this or --print must be set"))
    parser.add_argument('--increment', '-i', action='store_true',
                        help=("Whether we should return the last found attempt or increment it "
                              "by one. If passed, --extra_iter must also be set"))
    parser.add_argument('--gamma_corrected', '-g', action='store_true',
                        help=("Whether we should return the gamma-corrected path or not."))
    parser.add_argument('--comp', '-c', default='ref',
                        help=("{ref, met}, Whether to generate the scaling values for comparing "
                              "metamers to reference images or to other metamers"))
    parser.add_argument('--seed_n', '-n', nargs='+', type=int, default=[0, 1, 2],
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
    print_output = args.pop('print')
    save_path = args.pop('save_path')
    increment = args.pop('increment')
    extra_iter = args.pop('extra_iter')
    gamma_corrected = args.pop('gamma_corrected')
    image_kwargs = {k: args.pop(k) for k in ['ref_image', 'size', 'preproc']}
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
    if not save_path and not print_output:
        raise Exception("Either --save or --print must be true!")
    if save_path and not save_path.endswith('.txt'):
        raise Exception("--save must point towards a .txt file")
    paths = generate_metamer_paths(increment=increment, extra_iter=extra_iter,
                                   gamma_corrected=gamma_corrected,
                                   **new_args)
    # need to do a bit of string manipulation to get this in the right
    # format
    paths = ' '.join(paths)
    if print_output:
        print(paths)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(paths)
