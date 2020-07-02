"""various utilities
"""
import os
import re
import copy
import os.path as op
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
    return (im * np.iinfo(dtype).max).astype(dtype)


@contextmanager
def get_gpu_id(get_gid=True, n_gpus=4):
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
        name of the reference image, as used in the Snakefile. will
        contain the base image, any preprocessing (cone power, degamma,
        changing the range), and size

    Returns
    -------
    image_size : np.array
        array with 2 ints, giving the size of the image

    """
    image_size = re.findall("_size-([0-9]+,[0-9]+)", image_name)[0]
    return np.array(image_size.split(',')).astype(int)


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


def generate_metamer_paths(model_name, **kwargs):
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

    Parameters
    ----------
    model_name : str
        Name(s) of the model(s) to run. Must begin with either V1 or
        RGC. If model name is just 'RGC' or just 'V1', we will use the
        default model name for that brain area from config.yml
    kwargs :
        keys must be configurable options for METAMER_TEMPLATE_PATH, as
        found in config.yml. If an option is *not* set, we'll use the
        model-specific default value from that yml file. values can be
        either a single value or a list

    Parameters
    ----------
    paths : list
        list of strs, containing the absolute paths to the metamer.png
        files found in the metamer_display folder

    """
    if not isinstance(model_name, list):
        model_name = [model_name]
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    default_img_size = _find_img_size(defaults['DEFAULT_METAMERS']['image_name'][0])
    pix_to_deg = float(defaults['DEFAULT_METAMERS']['max_ecc']) / default_img_size.max()
    images = kwargs.pop('image_name', defaults['DEFAULT_METAMERS'].pop('image_name'))
    if not isinstance(images, list):
        images = [images]
    args = {}
    paths = []
    template_path = defaults['METAMER_TEMPLATE_PATH'].replace('metamers/{model_name}',
                                                              'metamers_display/{model_name}')
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
            elif model.startswith('V1'):
                args.update(defaults['V1'])
                if model == 'V1':
                    model = defaults['V1']['model_name']
            args['DATA_DIR'] = defaults['DATA_DIR']
            # by putting this last, we'll over-write the defaults
            args.update(kwargs)
            args.update({'model_name': model, 'image_name': im})
            # we want to handle lists differently, iterating through them. we
            # use an ordered dict because we want the keys and values to have
            # the same ordering when we iterate through them separately
            list_args = OrderedDict({k: v for k, v in args.items() if isinstance(v, list)})
            for k in list_args.keys():
                args.pop(k)
            for vals in product(*list_args.values()):
                tmp = dict(zip(list_args.keys(), vals))
                p = template_path.format(**tmp, **args)
                p = op.join(*p.split('/'))
                paths.append(p)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Generate metamer paths in a programmatic way, for passing to snakemake. "
                     "With the exception of model_name, --print, and --save_path, all other "
                     "arguments are the various configurable options from the metamer template "
                     "path, which control synthesis behavior. All arguments can take multiple "
                     "values, in which case we'll generate all possible combinations. If a value "
                     "is unset, we'll use the model-specific defaults from config.yml."))
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
    template_path = defaults['METAMER_TEMPLATE_PATH']
    possible_args = re.findall('{([A-Za-z_]+?)}', template_path)
    parser.add_argument('--print', '-p', action='store_true',
                        help="Print out the paths. Note either this or --save_path must be set")
    parser.add_argument('--save_path', '-s', default='',
                        help=("Path to a .txt file to save the paths at. If not set, will not "
                              "save. Note either this or --print must be set"))
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
    image_kwargs = {k: args.pop(k) for k in ['ref_image', 'size', 'preproc']}
    images = generate_image_names(**image_kwargs)
    args = {k: v for k, v in args.items() if v is not None}
    if 'image_name' in args.keys():
        args['image_name'].extend(images)
    else:
        args['image_name'] = images
    if not save_path and not print_output:
        raise Exception("Either --save or --print must be true!")
    if save_path and not save_path.endswith('.txt'):
        raise Exception("--save must point towards a .txt file")
    paths = generate_metamer_paths(**args)
    # need to do a bit of string manipulation to get this in the right
    # format
    paths = os.sep + f' {os.sep}'.join(paths)
    if print_output:
        print(paths)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(paths)
