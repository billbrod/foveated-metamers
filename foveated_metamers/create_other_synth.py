#!/usr/bin/env python3

"""Synthesize other images for this project.

Largely, that means mixing two images to obtain a specific MSE.

"""

import torch
import imageio
from tqdm.auto import tqdm
import plenoptic as po
import pyrtools as pt
import numpy as np
import matplotlib.pyplot as plt
from . import distances
from . import create_metamers


def mix_images(base_image, image_to_mix, alpha, direction='L'):
    """Mix together two images on one horizontal half, with weight alpha.

    Note we do not add a bar or anything else here.

    Parameters
    ----------
    base_image, image_to_mix : torch.Tensor
        The two images to mix.
    alpha : float
        The weight to multiply by image_to_mix
    direction : {'L', 'R'}, optional
        Whether to add image_to_mix on left or right half.

    Returns
    -------
    mixed_image : torch.Tensor
        The mixed image.

    """
    img_half_width = image_to_mix.shape[-1] // 2
    mixed_image = base_image.clone()
    if direction == 'R':
        mixed_image[..., img_half_width:] += alpha*image_to_mix[..., img_half_width:]
    elif direction == 'L':
        mixed_image[..., :img_half_width] += alpha*image_to_mix[..., :img_half_width]
    else:
        raise Exception(f"Don't know how to handle direction {direction}")
    return mixed_image


def obj_func(base_image, image_to_mix, alpha, target_err, direction='L'):
    """Get mse and objective function value between base_image and the version mixed with image_to_mix.

    Note we do not add a bar or anything else here.

    Parameters
    ----------
    base_image, image_to_mix : torch.Tensor
        The two images to mix.
    alpha : float
        The weight to multiply by image_to_mix
    target_err : float
        The target MSE value.
    direction : {'L', 'R'}, optional
        Whether to add image_to_mix on left or right half.

    Returns
    -------
    mse : torch.Tensor
        The MSE between the base_image and the mixed version.
    obj_value : torch.Tensor
        The squared error between mse and target_err

    """
    mixed_image = mix_images(base_image, image_to_mix, alpha, direction)
    mse = torch.square(mixed_image - base_image).mean()
    return mse, torch.square(mse - target_err)


def find_alpha(base_image, image_to_mix, alpha, target_err, learning_rate,
               max_iter, direction):
    """Use SGD to find correct alpha value.

    Parameters
    ----------
    base_image : torch.Tensor
        The base image, to which we're adding alpha*image_to_mix on left or
        ride side.
    image_to_mix : torch.Tensor
        The other image, added to base_image.
    alpha : float
        The weight to multiply by image_to_mix.
    target_err : float
        The target MSE value.
    learning_rate : float
        Learning rate for SGD.
    max_iter : int
        Maximum number of iterations to perform.
    direction : {'L', 'R'}, optional
        Whether to add image_to_mix on left or right half.

    Returns
    -------
    alphas : list
        List of alpha values over synthesis.
    mses : list
        List of MSEs values over synthesis.
    objs : list
        List of objective function values over synthesis.

    """
    opt = torch.optim.SGD([alpha], learning_rate)
    pbar = tqdm(range(max_iter))
    objs = []
    mses = []
    alphas = []
    for i in pbar:
        mse, obj = obj_func(base_image, image_to_mix, alpha, target_err,
                            direction)
        objs.append(obj.item())
        mses.append(mse.item())
        alphas.append(alpha.item())
        if obj.isnan():
            break
        opt.zero_grad()
        obj.backward()
        opt.step()
        pbar.set_postfix(mse=mse.item(), alpha=alpha.item(), loss=f'{obj.item():.3e}')
    # get the final values
    mse, obj = obj_func(base_image, image_to_mix, alpha, target_err,
                        direction)
    objs.append(obj.item())
    mses.append(mse.item())
    alphas.append(alpha.item())
    return alphas, mses, objs


def main(base_image, image_to_mix, target_err, learning_rate, max_iter=100,
         direction='L', seed=None, save_path=None, bar_deg_size=2.,
         screen_size_deg=73.45, screen_size_pix=3840):
    """Determine alpha value for mixing two images.

    Parameters
    ----------
    base_image : torch.Tensor or str
        The base image, to which we're adding alpha*image_to_mix on left or
        ride side. If a string, we assume it's a path and will load it in.
    image_to_mix : torch.Tensor or str
        The other image, added to base_image. If a string, we assume it's
        either a type of noise, one of {'pink', 'white'}, or the path and will
        load it in.
    target_err : float
        The target MSE value.
    learning_rate : float
        Learning rate for SGD.
    max_iter : int, optional
        Maximum number of iterations to perform.
    direction : {'L', 'R'}, optional
        Whether to add image_to_mix on left or right half.
    seed : int or None, optional
        The number to use for initializing numpy and torch's random
        number generators. if None, we don't set the seed.
    save_path : str or None, optional
        If not None, path to save resulting mixed image at. Will also save a
        figure showing synthesis progress at `save_path.replace('.png',
        '_synth.svg')`
    bar_deg_size : float, optional
        Width of the bar, in degrees. Default matches experimental setup.
    screen_size_deg : float, optional
        Width of the screen, in degrees. Default matches experimental setup.
    screen_size_pix : float, optional
        Width of the screen, in pixels. Default matches experimental setup.

    Returns
    -------
    alpha : float
        The final alpha
    fig : plt.Figure
        Figure containing the synthesis progress plot.

    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # for this, we want the images to be between 0 and 255 (that's the range we
    # calculated MSE on)
    base_image = 255 * create_metamers.setup_image(base_image)
    if isinstance(image_to_mix, str) and image_to_mix in ['white', 'pink']:
        if image_to_mix == 'white':
            image_to_mix = torch.rand_like(base_image, dtype=torch.float32)
        elif image_to_mix == 'pink':
            # this `.astype` probably isn't necessary, but just in case
            image_to_mix = pt.synthetic_images.pink_noise(base_image.shape[-2:]).astype(np.float32)
            # need to rescale this so it lies between 0 and 1
            image_to_mix += np.abs(image_to_mix.min())
            image_to_mix /= image_to_mix.max()
            image_to_mix = torch.Tensor(image_to_mix).unsqueeze(0).unsqueeze(0)
        image_to_mix = 255* image_to_mix
    else:
        image_to_mix = 255 * create_metamers.setup_image(image_to_mix)
    bar_pix_size = int(bar_deg_size * (screen_size_pix / screen_size_deg))
    bar = distances._create_bar_mask(base_image.shape[1], bar_pix_size)
    base_image = distances._add_bar(base_image, bar)
    image_to_mix = distances._add_bar(image_to_mix, bar)
    alpha = torch.rand(1).squeeze().requires_grad_()
    alphas, mses, objs = find_alpha(base_image, image_to_mix, alpha,
                                    target_err, learning_rate, max_iter,
                                    direction)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, data, name) in enumerate(zip(axes, [objs, alphas, mses],
                                             ['Objective function', 'Alpha', 'MSE'])):
        if i == 0:
            ax.semilogy(data)
        else:
            ax.plot(data)
        if name == 'MSE':
            ax.axhline(target_err, c='k', linestyle='--')
        ax.set(xlabel='iteration', ylabel=name,
               title=f'Final value = {data[-1]}')
    if save_path is not None:
        mixed_image = mix_images(base_image, image_to_mix, alphas[-1], direction)
        mixed_image = po.to_numpy(mixed_image).squeeze()
        imageio.imwrite(save_path, mixed_image)
        fig.savefig(save_path.replace('.png', '_synth.svg'))
    return alphas[-1], fig
