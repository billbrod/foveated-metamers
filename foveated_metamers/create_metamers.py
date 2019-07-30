#!/usr/bin/python
"""create metamers for the experiment
"""
import torch
import imageio
import logging
import numpy as np
import plenoptic as po
import os.path as op
import matplotlib
# by default matplotlib uses the TK gui toolkit which can cause problems
# when I'm trying to render an image into a file, see
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('Agg')


def setup_logger(log_file=None):
    r"""setup the logger we use

    This just sets up the logger, sets the appropriate level and adds a
    StreamHandler. If log_file is not None, we also log to that file

    The logger we set up is the 'create_metamers' logger. After this has
    been run, it can be grabbed by calling
    ``logging.getLogger('create_metamers')``

    Parameters
    ----------
    log_file : str or None, optional
        If str, the path to the file we'll log to. If None, we don't log
        to file

    Returns
    -------
    logger : logging.Logger
        The initialized create_metamers logger
    log_file : None or file
        If log_file parameter was None, so is this. If it was a str,
        this is the open file object corresponding to that str. After
        you've finished logging, should call ``log_file.close()``

    """
    logger = logging.getLogger('create_metamers')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    if log_file is not None:
        log_file = open(log_file, 'w')
        logger.addHandler(logging.StreamHandler(log_file))
    return logger, log_file


def setup_image(image, device):
    r"""setup the image

    We load in the image, if it's not already done so (converting it to
    gray-scale in the process), make sure it lies between 0 and 1, and
    make sure it's a tensor of the correct type and specified device

    Parameters
    ----------
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    device : torch.device
        The torch device to put the image on

    Returns
    -------
    image : torch.Tensor
        The image tensor, ready to go

    """
    logger = logging.getLogger('create_metamers')
    if isinstance(image, str):
        logger.info("Loading in seed image from %s" % image)
        # use imageio.imread in order to handle rgb correctly. this uses the ITU-R 601-2 luma
        # transform, same as matlab
        image = imageio.imread(image, as_gray=True)
    if image.max() > 1:
        logger.warning("Assuming image range is (0, 255)")
        image /= 255
    else:
        logger.warning("Assuming image range is (0, 1)")
    image = torch.tensor(image, dtype=torch.float32, device=device)
    while image.ndimension() < 4:
        image = image.unsqueeze(0)
    return image


def setup_model(model_name, scaling, image, min_ecc, max_ecc):
    r"""setup the model

    We initialize the model, with the specified parameters, and return
    it with the appropriate figsize.

    Parameters
    ----------
    model_name : {'RGC', 'V1'}
        Which type of model to create.
    scaling : float
        The scaling parameter for the model
    image : torch.tensor or np.array
        The image we will call the model on. This is only necessary
        because we need to know how big it is; we just use its shape
    min_ecc : float
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)

    Returns
    -------
    model : plenoptic.simul.VentralStream
        A ventral stream model, ready to use
    figsize : tuple
        The figsize tuple to use with ``metamer.animate`` or other
        plotting functions

    """
    if model_name == 'RGC':
        model = po.simul.RetinalGanglionCells(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                              max_eccentricity=max_ecc)
        figsize = (17, 5)
        # default figsize arguments work for an image that is 256x256,
        # may need to expand
        figsize = tuple([s*max(1, image.shape[i]/256) for i, s in enumerate(figsize)])
    elif model_name == 'V1':
        model = po.simul.PrimaryVisualCortex(scaling, image.shape[-2:], min_eccentricity=min_ecc,
                                             max_eccentricity=max_ecc)
        figsize = (35, 11)
        # default figsize arguments work for an image that is 512x512,
        # may need to expand
        figsize = tuple([s*max(1, image.shape[i]/512) for i, s in enumerate(figsize)])
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    return model, figsize


def finalize_metamer_image(model, metamer_image, image):
    r"""Add the center back to the metamer image

    The VentralStream class of models will do nothing to the center of
    the image (they don't see the fovea) and so we ned to add the fovea
    from the original image back in for our experiments.

    Parameters
    ----------
    model : plenoptic.simul.VentralStream
        The model used to create the metamer. Specifically, we need its
        windows attribute
    metamer_image : torch.Tensor
        The image created by metamer synthesis (its the argument
        returned by ``metamer.synthesis`` or, equivalently,
        ``metamer.matched_image``)
    image : torch.Tensor
        The original/target image for synthesis
        (``metamer.target_image``)

    Returns
    -------
    metamer_image : torch.Tensor
        The metamer image with the center added back in

    """
    metamer_image = metamer_image.squeeze()
    image = image.squeeze()
    windows = model.PoolingWindows.windows[0].flatten(0, -3)
    # for some reason ~ (invert) is not implemented for booleans in
    # pytorch yet, so we do this instead.
    return ((windows.sum(0) * metamer_image) + ((1 - windows.sum(0)) * image))


def save(save_path, metamer, figsize):
    r"""save the metamer output

    We save three things here:
    - The metamer object itself, at ``save_path``. This contains, among
      other things, the saved image and representation over the course
      of synthesis.
    - The finished metamer image, at ``os.path.splitext(save_path)[0] +
      "_metamer.png"``. This is not just ``metamer.matched_image``, but
      has had the center added back in, as done by
      ``finalize_metamer_image``
    - The video showing synthesis progress at
      ``os.path.splitext(save_path)[0] + "_synthesis.mp4"``. We use this
      to visualize the optimization progress.

    Parameters
    ----------
    save_path : str
        The path to save the metamer object at, which we use as a
        starting-point for the other save paths
    metamer : plenoptic.synth.Metamer
        The metamer object after synthesis
    figsize : tuple
        The tuple describing the size of the figure for the synthesis
        video, as returned by ``setup_model``.

    """
    logger = logging.getLogger('create_metamers')
    logger.info("Saving at %s" % save_path)
    metamer.save(save_path, save_model_reduced=True)
    # save png of metamer
    metamer_path = op.splitext(save_path)[0] + "_metamer.png"
    logger.info("Saving metamer image at %s" % metamer_path)
    metamer_image = finalize_metamer_image(metamer.model, metamer.matched_image,
                                           metamer.target_image)
    imageio.imwrite(metamer_path, metamer_image.squeeze().detach())
    video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
    logger.info("Saving synthesis video at %s" % video_path)
    anim = metamer.animate(figsize=figsize)
    anim.save(video_path)


def main(model_name, scaling, image, seed=0, min_ecc=.5, max_ecc=15, learning_rate=1, max_iter=100,
         loss_thresh=1e-4, log_file=None, save_path=None):
    r"""create metamers!

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    Parameters
    ----------
    model_name : {'RGC', 'V1'}
        Which type of model to create.
    scaling : float
        The scaling parameter for the model
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    seed : int, optional
        The number to use for initializing numpy and torch's random
        number generators
    min_ecc : float, optional
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float, optional
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    learning_rate : float, optional
        The learning rate to pass to metamer.synthesize's optimizer
    max_iter : int, optional
        The maximum number of iterations we allow the synthesis
        optimization to run for
    loss_thresh : float, optional
        The loss threshold. If our loss is every below this, we stop
        synthesis and consider ourselves done.
    log_file : str or None, optional
        If a str, the path to the file to log to. If None, we don't log
        to a file (though we do still log to stdout)
    save_path : str or None, optional
        If a str, the path to the file to save the metamer object to. If
        None, we don't save the synthesis output (that's probably a bad
        idea)

    """
    logger, log_file = setup_logger(log_file)
    logger.info("Using seed %s" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("On device %s" % device)
    image = setup_image(image, device)
    model, figsize = setup_model(model_name, scaling, image, min_ecc, max_ecc)
    logger.info("Using model %s from %.02f degrees to %.02f degrees" % (model_name, min_ecc,
                                                                        max_ecc))
    logger.info("Using learning rate %s, loss_thresh %s, and max_iter %s" % (learning_rate,
                                                                             loss_thresh,
                                                                             max_iter))
    clamper = po.RangeClamper((0, 1))
    initial_image = torch.nn.Parameter(torch.rand_like(image, requires_grad=True, device=device,
                                                       dtype=torch.float32))
    metamer = po.synth.Metamer(image, model)
    if save_path is not None:
        save_progress = True
    else:
        save_progress = False
    matched_im, matched_rep = metamer.synthesize(clamper=clamper, store_progress=10,
                                                 learning_rate=learning_rate, max_iter=max_iter,
                                                 loss_thresh=loss_thresh, seed=seed,
                                                 initial_image=initial_image,
                                                 save_progress=save_progress, save_path=save_path)
    if save_path is not None:
        save(save_path, metamer, figsize)
    if log_file is not None:
        log_file.close()
