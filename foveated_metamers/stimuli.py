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
    pad_kwargs :
        Passed to ``skimage.util.pad``

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


def _gen_trial_types(df):
    """Generate the trial types arrays

    These arrays contains the indices of images that should be compared against
    each other: the first contains all metamers synthesized from the same
    reference image, for the same model with the same scaling value, and the
    second contains reference image they're based on. To combine them into one
    array: `np.concatenate((metamers, ref_images), 1)`

    It then gets used by other functions to convert that into the format needed
    for experiments

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the metamer information, as created by
        ``create_metamer_df``

    Returns
    -------
    metamers : np.ndarray
        2d array, `n_comparisons` by `n_initializations`.
    ref_images : np.ndarray
        2d array, `n_comparisons` by 1.

    """
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
    metamers = []
    reference_images = []
    for s, i, m in itertools.product(*trials_dict.values()):
        metamers.append(df.query('scaling==@s & image_name==@i & model==@m').index)
        reference_images.append(df.query('scaling=="None" & image_name==@i & model=="None"').index)
    return np.array(metamers), np.array(reference_images)


def generate_indices_split(df, seed, comparison='met_v_ref', n_repeats=None):
    """Generate randomized presentation indices for split-screen task.

    We take in the dataframe describing the metamer images combined into our
    stimuli array and correctly structure the presentation indices for the
    split-screen task (as used in our experiment.py file), randomize them using
    the given seed, and return them.

    This task always presents the reference image first for met_v_ref
    comparison (for met_v_met, presentation order is randomized; thus there
    will be more trials).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the metamer information, as created by
        ``create_metamer_df``
    seed : int
        The seed passed to ``np.random.seed``.
    comparison : {'met_v_met', 'met_v_ref'}, optional
        Whether to create the indices for comparing metamers against each other
        or against the reference image
    n_repeats : int or None, optional
        How many repeats for each (image, scaling). Must be an even number (so
        it's on the left and right an even number of times). Default (None)
        doesn't change anything, so that each (image, scaling, seed) shows up
        on left and right each once. 6 is the value used for our experiment.
        Will balance across seeds.

    Returns
    -------
    trials : np.array
        The 2 x n x 2 of presentation indices. n is n_repeats * number of
        images * number of scaling values.

    """
    if n_repeats is not None and n_repeats % 2 != 0:
        raise Exception(f"n_repeats must be even but got {n_repeats}!")
    np.random.seed(seed)
    # get the trial types array, which gives the indices for images to compare
    # against each other.
    mets, refs = _gen_trial_types(df)
    # after this, each row of trials contains three indices, first two are the
    # left and right, respectively, of initial stimulus, and the final is the
    # image to change to.
    if comparison == 'met_v_met':
        if n_repeats is not None and n_repeats != (mets.shape[-1] * mets.shape[-1]-1):
            # there will be n*(n-1) number of repeats, where n is the number of
            # seeds (equivalently, the number of metamers with the same image,
            # scaling). need to think through how to implement any other values...
            raise NotImplementedError("Haven't thought through how to implement"
                                      " n_repeats for met_v_met yet!")
        # from each set of comparisons, grab all possible permutations of list
        # 2. that is, for each set of metamers with same scaling and reference
        # image, get every possible combination of two, and both orderings
        trials = np.array([list(itertools.permutations(t, 2)) for t in mets])
        # then duplicate the first image (so it shows up on both left and
        # right)
        trials = np.dstack([trials[:, :, 0], trials])
        # and make this 2d
        trials = trials.reshape(-1, trials.shape[-1])
    elif comparison == 'met_v_ref':
        if n_repeats is not None:
            # this is how many repeats of each (image, scaling) we would have
            # if we didn't do anything more (mets.shape[-1] is the number of
            # seeds, and we make sure each shows up once on left and once on
            # right)
            base_repeats = mets.shape[-1] * 2
            # we pad out mets so that mets.shape[-1] is n_repeats//2 and that
            # each seed shows up with the same frequency (or as close as
            # possible)
            mets = np.tile(mets, int(np.ceil(n_repeats / base_repeats)))
            mets = mets[:, :n_repeats//2]
        # grab each reference image and metamer (in that order). We're showing
        # the reference image first on each trial.
        trials = np.array([[refs[i, 0], c] for i, comp in enumerate(mets) for c
                           in comp])
        # and now duplicate the first image (so it shows up on both left and
        # right)
        trials = trials[:, [0, 0, 1]]
    # now we duplicate the side that we don't change. this will thus double the
    # number of rows, as each row gets a no-change on the left and on the right
    trials = np.array([[[i, j, k, j], [i, j, i, k]]
                       for i, j, k in trials])
    # this reshapes it so trials are indexed along the first dimension, and
    # then each trial is 2x2, representing the left and right sides for the
    # first and second stimulus.
    trials = trials.reshape(-1, 2, 2)
    # Now permute. we set the random seed at the top of this function for
    # reproducibility
    trials = np.random.permutation(trials)
    # and this rearranges it so left and right are along the first dimension,
    # then trials, then first and second stimulus. This needs to happen after
    # the permutation, because np.random.permutation only permutes along the
    # first axis
    trials = np.moveaxis(trials, 2, 0)
    return trials
