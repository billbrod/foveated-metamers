"""various utilities
"""
import os
import warnings
from contextlib import contextmanager
from itertools import cycle
import GPUtil
import numpy as np


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
    avail_gpus = GPUtil.getAvailable(order='memory', maxLoad=.1, maxMemory=.1, includeNan=False,
                                     limit=n_gpus)
    if not get_gid:
        avail_gpus = []
    for gid in cycle(avail_gpus):
        # then we've successfully created the lockfile
        if os.system(f"dotlockfile -r 1 /tmp/LCK_gpu_{gid}.lock") == 0:
            allocated_gid = gid
            break
    yield allocated_gid
    os.system(f"dotlockfile -u /tmp/LCK_gpu_{allocated_gid}.lock")
