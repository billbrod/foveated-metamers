#!/usr/bin/env python3
"""Code to fit the psychophysical curve to data
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plenoptic as po


def calculate_discriminability(scaling, proportionality_factor, critical_scaling):
    r"""Calculate disriminability at given scaling, for specified parameters.

    This comes from the Online Methods section of [1]_, equation 17.

    Parameters
    ----------
    scaling : torch.Tensor
        Scaling value(s) to calculate discriminability for.
    proportionality_factor : torch.Tensor or float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : torch.Tensor or float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.

    Returns
    -------
    discrim : torch.tensor
        discriminability ($d^2$ in [1]_) at each scaling value.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    discrim = torch.zeros_like(scaling)
    masks = [scaling <= critical_scaling, scaling > critical_scaling]
    vals = [torch.zeros_like(scaling),
            proportionality_factor * (1 - (critical_scaling**2 / scaling**2))]
    for m, v in zip(masks, vals):
        discrim[m] = v[m]
    return discrim


def proportion_correct_curve(scaling, proportionality_factor, critical_scaling):
    r"""Compute the proportion correct curve, as function of parameters.

    This comes from the Online Methods section of [1]_, equation 18.

    Parameters
    ----------
    scaling : torch.Tensor
        Scaling value(s) to calculate discriminability for.
    proportionality_factor : torch.Tensor or float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : torch.Tensor or float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.

    Returns
    -------
    proportion_correct : torch.tensor
        The proportion correct curve at each scaling value, as given by the
        parameter values

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    norm = torch.distributions.Normal(0, 1)
    discrim = calculate_discriminability(scaling, proportionality_factor,
                                         critical_scaling)
    return (norm.cdf(discrim / np.sqrt(2)) * norm.cdf(discrim / 2)
            + norm.cdf(-discrim / np.sqrt(2)) * norm.cdf(-discrim / 2))


def fit_psychophysical_parameters(scaling, proportion_correct, lr=.001,
                                  scheduler=True, max_iter=10000):
    """Fit the parameters of psychophysical curve for a single set of values.

    Parameters
    ----------
    scaling : torch.tensor
        The scaling values tested.
    proportion_correct : torch.tensor
        The proportion correct at each of those scaling values.
    lr : float, optional
        The learning rate for Adam optimizer.
    scheduler : bool, optional
        Whether to use the scheduler or not (reduces lr by half when loss
        appears to plateau).
    max_iter : int, optional
        The number of iterations to optimize for.

    Returns
    -------
    proportionality_factor : float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.
    losses : list of floats
        Loss on each iteration.
    proportionality_factor_history : list of floats
        proportionality_factor on each iteration.
    critical_scaling_history : list of floats
        critical_scaling on each iteration.

    """
    a_0 = torch.nn.Parameter(torch.rand(1))
    s_0 = torch.nn.Parameter(torch.rand(1))
    losses = []
    a_0s = []
    s_0s = []
    optimizer = torch.optim.Adam([a_0, s_0], lr=lr)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               'min', .5)
    pbar = tqdm(range(max_iter))
    for i in pbar:
        yhat = proportion_correct_curve(scaling, a_0, s_0)
        loss = torch.sum((proportion_correct - yhat)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(loss.item())
        pbar.set_postfix({'loss': loss.item(), 'a_0': a_0.item(), 's_0': s_0.item(),
                          'lr': optimizer.param_groups[0]['lr']})
        losses.append(loss.item())
        a_0s.append(a_0.item())
        s_0s.append(s_0.item())
    return a_0.detach(), s_0.detach(), losses, a_0s, s_0s


def plot_optimization_results(scaling, proportion_correct,
                              proportionality_factor, critical_scaling, loss,
                              proportionality_factor_history=[],
                              critical_scaling_history=[]):
    """Plot results of psychophysical curve fitting

    This doesn't quite work as I want right now -- need to test with e.g.,
    multiple bootstraps and find best way to make it work

    """
    if not hasattr(proportionality_factor, '__iter__'):
        proportionality_factor = [proportionality_factor]
    if not hasattr(critical_scaling, '__iter__'):
        critical_scaling = [critical_scaling]
    prop_corr_curves = []
    for i, (a_0, s_0) in enumerate(zip(proportionality_factor, critical_scaling)):
        tmp = proportion_correct_curve(scaling, a_0, s_0)
        prop_corr_curves.append(pd.DataFrame({'prop_corr': tmp, 'scaling':
                                              po.to_numpy(scaling), 'n': i}))
    prop_corr_curves = pd.concat(prop_corr_curves).reset_index(drop=True)
    n_plots = 2
    if proportionality_factor_history:
        n_plots += 1
    if critical_scaling_history:
        n_plots += 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    sns.lineplot(x='scaling', y='prop_corr', ax=axes[0], data=prop_corr_curves)
    axes[0].scatter(scaling, proportion_correct)
    axes[0].set(title='data and psychophysical curve', xlabel='Proportion correct')
    axes[1].semilogy(loss)
    axes[1].set(xlabel='iteration', ylabel='loss', title='Loss')
    if proportionality_factor_history:
        axes[2].plot(proportionality_factor_history)
        axes[2].set(xlabel='iteration', ylabel=r'proportionality_factor ($\alpha_0$)',
                    title='Parameter history')
    if critical_scaling_history:
        axes[3].plot(critical_scaling_history)
        axes[3].set(xlabel='iteration', ylabel=r'critical_scaling ($s_0$)',
                    title='Parameter history')
    fig.tight_layout()
    return fig
