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
                                  scheduler=True, max_iter=10000, seed=None,
                                  proportionality_factor_init_range=(1, 5),
                                  critical_scaling_init_range=(0, .5)):
    r"""Fit the parameters of psychophysical curve for a single set of values.

    This has trouble (unsurprisingly) if critical scaling falls too far outside
    the range of tested values. Also has issues if critical_scaling's
    initialization is too far outside of tested values (it seems less sensitive
    to proportionality_factor); default values seem to work pretty well.

    Loss is MSE between predicted proportion correct and
    ``proportion_correct``, plus a quadratic penalty on negative values of
    critical_scaling (``10*(critical_scaling)**2``, in order to make it the
    magnitude of that loss matter when compared against the proportion correct
    MSE)

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
    seed : int or None, optional
        Seed to set pytorch's RNG with (we don't set numpy's, because we don't
        use any numpy function)
    proportionality_factor_init_range : tuple of floats, optional
        Range of values for initialization of proportionality_factor parameter
        (uniform distribution on this interval). If initialized value is too
        small, seems to have trouble finding a good solution.
    critical_scaling_init_range : tuple of floats, optional
        Range of values for initialization of critical_scaling parameter
        (uniform distribution on this interval). If initialized value is too
        large, seems to have trouble finding a good solution. Can tell because
        neither parameter changes. Should probably be the same as the range of
        ``scaling``

    Returns
    -------
    proportionality_factor : torch.tensor
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : torch.tensor
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.
    losses : np.ndarray of floats
        Loss on each iteration.
    proportionality_factor_history : np.ndarray of floats
        proportionality_factor on each iteration.
    critical_scaling_history : np.ndarray of floats
        critical_scaling on each iteration.

    """
    if seed is not None:
        torch.manual_seed(seed)
    a_0 = torch.nn.Parameter(np.diff(proportionality_factor_init_range)[0]*torch.rand(1) +
                             np.min(proportionality_factor_init_range))
    s_0 = torch.nn.Parameter(np.diff(critical_scaling_init_range)[0]*torch.rand(1) +
                             np.min(critical_scaling_init_range))
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
        # MSE on proportion correct, plus a penalty on negative critical
        # scaling values
        loss = torch.sum((proportion_correct - yhat)**2)
        if s_0 < 0:
            # the sum is really unnecessary (s_0 will only ever be a single
            # value), but it makes sure that it's a 0d tensor, like loss
            loss += 10*(s_0 ** 2).sum()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step(loss.item())
        pbar.set_postfix({'loss': loss.item(), 'a_0': a_0.item(), 's_0': s_0.item(),
                          'lr': optimizer.param_groups[0]['lr']})
        losses.append(loss.item())
        a_0s.append(a_0.item())
        s_0s.append(s_0.item())
    return a_0.detach(), s_0.detach(), np.array(losses), np.array(a_0s), np.array(s_0s)


def plot_optimization_results(scaling, proportion_correct,
                              proportionality_factor, critical_scaling, loss,
                              proportionality_factor_history=[],
                              critical_scaling_history=[], fig=None,
                              plot_data=True, **plot_kwargs):
    r"""Plot results of psychophysical curve fitting.

    Creates 2 to 5 plot figure:
    1. Data and psychophysical curve
    2. Loss over iterations
    3. History of proportionality_factor values (if
       proportionality_factor_history is set)
    4. History of critical_scaling values (if
       critical_scaling_history is set)
    5. Scatter plot showing initial and final values of the two parameters
       (critical_scaling on x, proportionality_factor on y; if both histories
       are set)

    Intended to be called multiple time with, e.g., multiple bootstraps or
    optimization iterations. In order to do that, call this once with
    ``fig=None``, then pass the returned figure as the ``fig`` argument to
    subsequent calls (probably with ``plot_data=False``)

    Parameters
    ----------
    scaling : torch.tensor
        The scaling values tested.
    proportion_correct : torch.tensor
        The proportion correct at each of those scaling values.
    proportionality_factor : torch.tensor or float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. Currently only handle single values
        for this (i.e., one curve)
    critical_scaling : torch.tensor or float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study. Currently only handle single values
        for this (i.e., one curve)
    losses : np.ndarray of floats
        Loss on each iteration.
    proportionality_factor_history : np.ndarray of floats, optional
        proportionality_factor on each iteration.
    critical_scaling_history : np.ndarray of floats, optional
        critical_scaling on each iteration.
    fig : plt.Figure or None, optional
        If not None, the figure to plot on. We don't check number of axes or
        add any more, so make sure you've created enough.
    plot_data : bool, optional
        Whether to plot data on the first axes or just the psychophysical curve
    plot_kwargs :
        passed to each plotting funtion.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plots

    """
    plot_kwargs.setdefault('color', 'C0')
    prop_corr = proportion_correct_curve(scaling, proportionality_factor,
                                         critical_scaling)
    prop_corr_curves = pd.DataFrame({'prop_corr': prop_corr,
                                     'scaling': po.to_numpy(scaling)})
    n_plots = 2
    if np.any(proportionality_factor_history):
        n_plots += 1
    if np.any(critical_scaling_history):
        n_plots += 1
    if np.any(proportionality_factor_history) and np.any(critical_scaling_history):
        n_plots += 1
    if fig is None:
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    else:
        axes = fig.axes
    sns.lineplot(x='scaling', y='prop_corr', ax=axes[0], data=prop_corr_curves,
                 **plot_kwargs)
    if plot_data:
        data_kwargs = plot_kwargs.copy()
        data_kwargs.update({'color': 'k', 'label': 'data'})
        axes[0].scatter(scaling, proportion_correct, **data_kwargs)
    axes[0].set(title='Data and psychophysical curve', xlabel='scaling',
                ylabel='proportion correct')
    axes[1].semilogy(loss, **plot_kwargs)
    axes[1].set(xlabel='iteration', ylabel='loss', title='Loss')
    axes_idx = 2
    if np.any(proportionality_factor_history):
        axes[axes_idx].plot(proportionality_factor_history, **plot_kwargs)
        axes[axes_idx].set(xlabel='iteration',
                           ylabel=r'proportionality_factor ($\alpha_0$)',
                           title='Parameter history')
        axes_idx += 1
    if np.any(critical_scaling_history):
        axes[axes_idx].plot(critical_scaling_history, **plot_kwargs)
        axes[axes_idx].set(xlabel='iteration', ylabel=r'critical_scaling ($s_0$)',
                           title='Parameter history')
        axes_idx += 1
    if np.any(proportionality_factor_history) and np.any(critical_scaling_history):
        scatter_kwargs = plot_kwargs.copy()
        scatter_kwargs.pop('label', '')
        axes[axes_idx].plot(critical_scaling_history[[0, -1]],
                            proportionality_factor_history[[0, -1]], '-',
                            **scatter_kwargs)
        axes[axes_idx].scatter(critical_scaling_history[0],
                               proportionality_factor_history[0],
                               label='initial',
                               facecolor='none', **scatter_kwargs)
        axes[axes_idx].scatter(critical_scaling_history[-1],
                               proportionality_factor_history[-1],
                               label='final',
                               **scatter_kwargs)
        axes[axes_idx].set(xlabel=r'critical_scaling ($s_0$)',
                           ylabel=r'proportionality_factor ($\alpha_0$)',
                           title='Initial and final parameter values')
        # don't make the legend bigger than it needs to be
        if axes[axes_idx].legend_ is None:
            axes[axes_idx].legend()
        axes_idx += 1
    fig.tight_layout()
    return fig


def multi_plot_optimization_results(scaling, proportion_correct, results,
                                    **plot_kwargs):
    """Plot multiple optimization results.

    This calls ``plot_optimization_results`` multiple times and is intended to
    be used when you've, e.g., optimized multiple time on the same data and
    want to compare outcomes.

    Parameters
    ----------
    scaling : torch.tensor
        The scaling values tested.
    proportion_correct : torch.tensor
        The proportion correct at each of those scaling values.
    results : list
        List of results, as returned by ``fit_psychological_parameters``
    plot_kwargs :
        passed to each plotting funtion.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plots


    """
    plot_kwargs.update({'color': 'C0', 'label': 0})
    fig = plot_optimization_results(scaling, proportion_correct,
                                    *results[0], **plot_kwargs)
    for i, res in enumerate(results[1:]):
        plot_kwargs.update({'color': f'C{i+1}', 'label': i+1})
        plot_optimization_results(scaling, proportion_correct, *res,
                                  fig=fig, plot_data=False,
                                  **plot_kwargs)
    fig.axes[0].legend()
    return fig
