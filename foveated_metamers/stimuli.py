"""code to assemble stimuli for running experiment
"""
import imageio
import warnings
import numpy as np
from skimage import util


def pad_image(image, pad_mode, save_path=None, constant_values=.5, **pad_kwargs):
    if isinstance(image, str):
        image = imageio.imread(image, as_gray=True)
    else:
        if image.ndim > 2:
            raise Exception("We need image to be grayscale!")
    if image.max() > 1:
        warnings.warn("Assuming image range is (0, 255)")
        image /= 255
    if pad_mode == 'constant':
        pad_kwargs['constant_values'] = constant_values
    image = util.pad(image, int(image.shape[0]/2), pad_mode, **pad_kwargs)
    if save_path is not None:
        imageio.imwrite(save_path, image)
    return image


def collect_metamers(image_paths, save_path=None):
    images = []
    for i in image_paths:
        images.append(imageio.imread(i, as_gray=True))
    # want our images to be indexed along the first dimension
    images = np.einsum('ijk -> kij', np.dstack(images)).astype(np.uint8)
    if save_path is not None:
        np.save(save_path, images)
    return images
