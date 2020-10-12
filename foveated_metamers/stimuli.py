"""code to assemble stimuli for running experiment
"""
import imageio
import itertools
import warnings
import numpy as np
import pyrtools as pt
import pandas as pd
import os.path as op
from skimage import util, color
from .utils import convert_im_to_float, convert_im_to_int


def create_image(image_type, image_size, save_path=None, period=4):
    r"""Create a simple image

    Parameters
    ----------
    image_type : {'plaid', 'checkerboard'}
        What type of image to create
    image_size : tuple
        2-tuple of ints, specifying the image size
    save_path : str or None, optional
        If a str, the path to save the padded image at. If None, we
        don't save
    period : int, optional
        If image_type is 'plaid' or 'checkerboard', what period to use
        for the square waves that we use to generate them.

    Returns
    -------
    image : np.array
        The image we created

    """
    if image_type in ['plaid', 'checkerboard']:
        image = pt.synthetic_images.square_wave(image_size, period=period)
        image += pt.synthetic_images.square_wave(image_size, period=period, direction=np.pi/2)
        image += np.abs(image.min())
        image /= image.max()
        if image_type == 'checkerboard':
            image = np.where((image < .75) & (image > .25), 1, 0)
    else:
        raise Exception("Don't know how to handle image_type %s!" % image_type)
    if save_path is not None:
        imageio.imwrite(save_path, image)
    return image


def pad_image(image, pad_mode, save_path=None, constant_values=.5, **pad_kwargs):
    r"""Pad an image using ``skimage.util.pad``

    Currently, we only support grayscale images

    All additional kwargs are passed directly to ``skimage.util.pad``

    Parameters
    ----------
    image : str or np.array
        Either an image array or a string containing the path to an
        image. If it's a string, we load it in as a grayscale image. If
        an image, we check that it only has 2 dimensions (i.e., is
        grayscale) and raise an Exception if not
    pad_mode : str
        A valid ``pad_mode`` for ``skimage.util.pad`` (see that function
        for more description). For our purposes, probably ``'constant'``
        or ``'symmetric'``
    save_path : str or None, optional
        If a str, the path to save the padded image at. If None, we
        don't save
    constant_values : sequence or int, optional
        The values to set the the padded values to for each axis. See
        ``skimage.util.pad`` for more details. If ``pad_mode`` is not
        ``'constant'``, we ignore this.

    Returns
    -------
    padded_image : np.array
        The padded image

    """
    if isinstance(image, str):
        image = imageio.imread(image)
    else:
        if image.ndim > 2:
            raise Exception("We need image to be grayscale!")
    if image.max() > 1:
        warnings.warn("Assuming image range is (0, 255)")
        image = convert_im_to_float(image)
    if image.ndim == 3:
        # then it's a color image, and we need to make it grayscale
        image = color.rgb2gray(image)
    if pad_mode == 'constant':
        pad_kwargs['constant_values'] = constant_values
    image = util.pad(image, int(image.shape[0]/2), pad_mode, **pad_kwargs)
    if save_path is not None:
        imageio.imwrite(save_path, image)
    return image


def collect_images(image_paths, save_path=None):
    r"""Collect images into a single array

    We loop through a list of paths, loading in images (as grayscale),
    stack them so that the different images are indexed along the first
    dimension, and then cast them as ``np.uint8``. We finally optionally
    save them and return.

    Parameters
    ----------
    image_paths : list
        A list of strs, each of which is the path to an image
    save_path : str or None, optional
        The path to save the resulting np.array at. If None, we don't
        save

    Returns
    -------
    images : np.array
        The stacked array of grayscale images
    """
    images = []
    for i in image_paths:
        # then this is the image file
        if i.endswith('.png'):
            im = imageio.imread(i)
            # normalize everything to lie between 0 and 1
            im = convert_im_to_float(im)
        # then it's a float32 array, with range [0, 1]
        elif i.endswith('.npy'):
            im = np.load(i)
            if im.min() < 0 or im.max() > 1:
                raise Exception(f"image should have range [0, 1], but has range {im.min(), im.max()}!")
        else:
            raise Exception(f"Don't know how to handle file extension {i.split('.')[-1]}!")
        # then properly convert everything to uint8
        im = convert_im_to_int(im, np.uint8)
        images.append(im)
    # want our images to be indexed along the first dimension
    images = np.einsum('ijk -> kij', np.dstack(images))
    if save_path is not None:
        np.save(save_path, images)
    return images


def create_metamer_df(image_paths, save_path=None):
    r"""Create dataframe summarizing metamer information

    We do this by loading in and concatenating the summary.csv files
    created as one of the outputs of metamer creation.

    Parameters
    ----------
    image_paths : list
        A list of strings to image paths
    save_path : str or None
        If a str, must end in csv, and we save the dataframe here as a
        csv. If None, we don't save the dataframe

    Returns
    -------
    df : pd.DataFrame
        The metamer information dataframe

    """
    metamer_info = []
    for p in image_paths:
        # images can end in either metamer.png (8 bit), metamer-16.png (16
        # bit), or metamer.npy (32 bit)
        csv_path = p.replace('metamer.png', 'summary.csv').replace('metamer-16.png', 'summary.csv').replace('metamer.npy', 'summary.csv')
        if csv_path.endswith('csv'):
            # then this was a metamer image and the replace above
            # completed successfully
            tmp = pd.read_csv(csv_path)
        else:
            # then this was one of our original images, and the replace
            # above failed
            image_name = op.basename(p).replace('.pgm', '').replace('.png', '')
            tmp = pd.DataFrame({'base_signal': p, 'image_name': image_name}, index=[0])
        # all base_signals are .pgm files and each tmp df will only contain value
        if len(tmp.image_name.unique()) > 1:
            raise Exception("Somehow we have more than one image_name for metamer %s" % p)
        metamer_info.append(tmp)
    df = pd.concat(metamer_info)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def generate_indices(df, seed, save_path=None):
    r"""Generate the randomized presentation indices

    We take in the dataframe describing the metamer images combined into
    our stimuli array and correctly structure the presentation indices
    (as used in our experiment.py file), randomize them using the given
    seed, and optionally save them.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the metamer information, as created by
        ``create_metamer_df``
    seed : int
        The seed passed to ``np.random.seed``.
    save_path : str or None, optional
        If a str, the path to save the indices array at (should end in
        .npy). If None, we don't save the array.

    Returns
    -------
    trials : np.array
        The n x 3 of presentation indices.
    """
    np.random.seed(seed)
    # The reference images will have a bunch of NaNs in the
    # metamer-relevant fields. In order to be able to select them, we
    # replace these with "None"
    df = df.fillna('None')
    trials_dict = {}
    # Here we find the unique (non-None) values for the three fields
    # that determine each trial
    for k in ['scaling', 'image_name', 'model']:
        v = [i for i in df[k].unique() if i != 'None']
        trials_dict[k] = v
    # Now go through and find the indices for each unique combination of
    # these three (there should be multiple for each of these because of
    # the different seeds used) and then add the reference image
    trial_types = []
    for s, i, m in itertools.product(*trials_dict.values()):
        t = df.query('scaling==@s & image_name==@i & model==@m').index
        t = t.append(df.query('scaling=="None" & image_name==@i & model=="None"').index)
        trial_types.append(t)
    trial_types = np.array(trial_types)
    # Now generate the indices for the trial. At the end of this, trials
    # is a 2d array, n by 3, where each row corresponds to a single ABX
    # trial: two images from the same row of trial_types and then a
    # repeat of one of them
    trials = np.array([list(itertools.permutations(t, 2)) for t in trial_types])
    trials = trials.reshape(-1, trials.shape[-1])
    trials = np.array([[[i, j, i], [i, j, j]] for i, j in trials])
    trials = trials.reshape(-1, trials.shape[-1])
    # Now permute and save. we set the random seed at the top of this
    # function for reproducibility
    trials = np.random.permutation(trials)
    if save_path is not None:
        np.save(save_path, trials)
    return trials
