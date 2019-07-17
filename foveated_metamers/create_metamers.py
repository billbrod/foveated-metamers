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


def setup_logger(log_file):
    logger = logging.getLogger('create_metamers')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    if log_file is not None:
        log_file = open(log_file, 'w')
        logger.addHandler(logging.StreamHandler(log_file))
        # fh = logging.FileHandler(log_file)
        # fh.setLevel(logging.DEBUG)
        # logger.addHandler(fh)
    return logger, log_file


def main(model_name, scaling, image, seed=0, min_ecc=.5, max_ecc=15, learning_rate=1, max_iter=100,
         loss_thresh=1e-4, log_file=None, save_path=None):
    """create metamers!
    """
    logger, log_file = setup_logger(log_file)
    logger.info("Using seed %s" % seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("On device %s" % device)
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
    if model_name == 'RGC':
        model = po.simul.RetinalGanglionCells(scaling, image.shape, min_eccentricity=min_ecc,
                                              max_eccentricity=max_ecc)
        figsize = (17, 5)
    elif model_name == 'V1':
        model = po.simul.PrimaryVisualCortex(scaling, image.shape, min_eccentricity=min_ecc,
                                             max_eccentricity=max_ecc)
        figsize = (35, 11)
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    logger.info("Using model %s from %.02f degrees to %.02f degrees" % (model_name, min_ecc,
                                                                        max_ecc))
    logger.info("Using learning rate %s, loss_thresh %s, and max_iter %s" % (learning_rate,
                                                                             loss_thresh,
                                                                             max_iter))
    clamper = po.RangeClamper((0, 1))
    initial_image = torch.nn.Parameter(torch.rand_like(image, requires_grad=True, device=device,
                                                       dtype=torch.float32))
    metamer = po.synth.Metamer(image, model)
    matched_im, matched_rep = metamer.synthesize(clamper=clamper, save_representation=10,
                                                 save_image=10, learning_rate=learning_rate,
                                                 max_iter=max_iter, loss_thresh=loss_thresh,
                                                 initial_image=initial_image)
    # add back the center of the image. This class of models will do nothing to the center of the
    # image (they don't see the fovea) and so we do this to add the fovea back in. for some reason
    # ~ (invert) is not implemented for booleans in pytorch yet, so we do this instead.
    metamer_image = ((model.windows[0].sum(0) * matched_im) +
                     ((1 - model.windows[0].sum(0)) * image))
    if save_path is not None:
        logger.info("Saving at %s" % save_path)
        metamer.save(save_path, save_model_sparse=True)
        # save png of metamer
        metamer_path = op.splitext(save_path)[0] + "_metamer.png"
        logger.info("Saving metamer image at %s" % metamer_path)
        imageio.imwrite(metamer_path, metamer_image)
        video_path = op.splitext(save_path)[0] + "_synthesis.mp4"
        logger.info("Saving synthesis video at %s" % video_path)
        anim = metamer.animate(figsize)
        anim.save(video_path)
    if log_file is not None:
        log_file.close()
