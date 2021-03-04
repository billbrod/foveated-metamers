#!/usr/bin/env python3
"""Code to fit the psychophysical curve to data."""
import torch
# this is a wrapper and drop-in replacement for multiprocessing:
# https://pytorch.org/docs/stable/multiprocessing.html
import torch.multiprocessing as mp
import inspect
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def calculate_discriminability(scaling, proportionality_factor, critical_scaling):
    r"""Calculate disriminability at given scaling, for specified parameters.

    This comes from the Online Methods section of [1]_, equation 17.

    Parameters
    ----------
    scaling : torch.Tensor or np.ndarray
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
    if not isinstance(scaling, torch.Tensor):
        scaling = torch.tensor(scaling)
    vals = proportionality_factor * (1 - (critical_scaling**2 / scaling**2))
    # this has to be non-negative
    return vals.clamp(min=0)


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
    # we use the fact that norm.cdf(-x) = 1 - norm.cdf(x) to speed up the
    # following, which is equivalent to: norm.cdf(d/sqrt(2)) * norm.cdf(d/2) +
    # norm.cdf(-d/sqrt(2)) * norm.cdf(-d/2)
    norm_cdf_sqrt_2 = norm.cdf(discrim / np.sqrt(2))
    norm_cdf_2 = norm.cdf(discrim / 2)
    return norm_cdf_sqrt_2 * norm_cdf_2 + (1-norm_cdf_sqrt_2) * (1-norm_cdf_2)


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
    critical_scaling (``10*(critical_scaling)**2``, in order to make its
    magnitude matter when compared against the proportion correct MSE)

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
    result : pd.DataFrame
        DataFrame with 6 columns and ``max_iter`` rows, containing the
        proportionality_factor, critical_scaling (both constant across
        iterations), iteration, loss, proportionality_factor_history,
        critical_scaling_history.

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
            # value), but it makes sure that it's a 0d tensor, like loss. the
            # 10* is necessary to make it have enough weight to matter.
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
    result = pd.DataFrame({'proportionality_factor': a_0.item(),
                           'critical_scaling': s_0.item(),
                           'loss': np.array(losses),
                           'proportionality_factor_history': np.array(a_0s),
                           'critical_scaling_history': np.array(s_0s),
                           'iteration': np.arange(max_iter)})
    return result


def multi_fit_psychophysical_parameters(kwargs, use_multiproc=True,
                                        n_processes=None, identifiers=[]):
    """Run fit_psychophysical_parameters multiple times, optionally using multiproc.

    Simple helper function to fit the psychophysical parameters multiple times,
    using either multiproc or a simple for loop.

    Parameters
    ----------
    kwargs : list
        List of dictionaries to pass to ``fit_psychophysical_parameters``. We
        simply iterate through them and unpack on the call.
    use_multiproc : bool, optional
        Whether to use multiprocessing to parallelize across cores or not.
    n_processes : int or None, optional
        If ``use_multiproc`` is True, how many processes to run in parallel. If
        None, will be equal to ``os.cpu_count()``. If ``use_multiproc`` is
        False, this is ignored.
    identifiers : list, optional
        If not empty, list of dictionaries with same length as kwargs. Contains
        key: value pairs to add to the results from each run to identify them
        (e.g., different seeds, bootstrap numbers).

    Returns
    -------
    results : pd.DataFrame
        DataFrame containing all the results. we add the values from args as
        additional columns, to differentiate among them.

    """
    if len(identifiers) > 0 and len(identifiers) != len(kwargs):
        raise Exception("identifiers must be the same length as kwargs, but "
                        f'got {len(identifiers)} and {len(kwargs)}!')
    if use_multiproc:
        # multiprocessing.pool does not allow for kwargs, so we have to
        # construct the args tuple. this is difficult because we don't know
        # whether the kwargs we've been passed has all the args or not. to
        # solve that, we grab the function signature (in order)...
        defaults = OrderedDict(inspect.signature(fit_psychophysical_parameters).parameters)
        # and iterate through the arguments for each dictionary in kwargs. if
        # that dictionary has the corresponding key, grab it; else, grab the
        # value from defaults
        args = [[kwarg.get(k, v.default) for k, v in defaults.items()]
                for kwarg in kwargs]
        # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork for why
        # we use spawn
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_processes) as pool:
            results = pool.starmap_async(fit_psychophysical_parameters, args)
            results = results.get()
    else:
        results = []
        for kwarg in kwargs:
            results.append(fit_psychophysical_parameters(**kwarg))
    for (iden, result) in zip(identifiers, results):
        for (k, v) in iden.items():
            result[k] = v
    results = pd.concat(results)
    return results


def plot_optimization_results(data, result, hue=None, fig=None, plot_data=True,
                              plot_mean=False, **plot_kwargs):
    r"""Plot results of psychophysical curve fitting.

    Creates 5 plot figure:
    1. Data and psychophysical curve
    2. Loss over iterations
    3. History of proportionality_factor values
    4. History of critical_scaling values
    5. Scatter plot showing initial and final values of the two parameters
       (critical_scaling on x, proportionality_factor on y)

    Intended to be called multiple time with, e.g., multiple bootstraps or
    optimization iterations. In order to do that, call this once with
    ``fig=None``, then pass the returned figure as the ``fig`` argument to
    subsequent calls (probably with ``plot_data=False``)

    Parameters
    ----------
    data : pd.DataFrame
        data DataFrame containing columns for scaling and proportion correct.
    result : pd.DataFrame
        results DataFrame, as returned by ``fit_psychophysical_parameters``.
        can be from multiple fits, in which case hue must be set.
    hue : str or None, optional
        The variable in result to facet the hue on (we don't facet data)
    fig : plt.Figure or None, optional
        If not None, the figure to plot on. We don't check number of axes or
        add any more, so make sure you've created enough.
    plot_data : bool, optional
        Whether to plot data on the first axes or just the psychophysical curve
    plot_mean : bool, optional
        Whether to plot the (bootstrapped) mean across hue as well as the
        individual values. If so, we plot the individual values with a reduced
        alpha. This increases the amount of time this takes by a fair amount
        (roughly 3x; due to bootstrapping).
    plot_kwargs :
        passed to each plotting funtion.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plots

    """
    alpha = 1
    if plot_mean:
        alpha = .5
    if fig is None:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    else:
        axes = fig.axes
    prop_corr_curves = []
    scaling = data.scaling.unique()
    if hue is None:
        gb = [(None, result)]
    else:
        gb = result.groupby(hue)
    for n, g in gb:
        if g.critical_scaling.nunique() > 1 or g.proportionality_factor.nunique() > 1:
            raise Exception("For a given hue value, need a single solution!")
        prop_corr = proportion_correct_curve(scaling,
                                             g.proportionality_factor.unique()[0],
                                             g.critical_scaling.unique()[0])
        prop_corr = pd.DataFrame({'proportion_correct': prop_corr,
                                  'scaling': scaling, hue: n})
        prop_corr_curves.append(prop_corr)
    prop_corr_curves = pd.concat(prop_corr_curves)
    sns.lineplot(x='scaling', y='proportion_correct', hue=hue, ax=axes[0],
                 data=prop_corr_curves, alpha=alpha, **plot_kwargs)
    if plot_mean:
        sns.lineplot(x='scaling', y='proportion_correct', ax=axes[0],
                     data=prop_corr_curves, legend=False, color='k',
                     **plot_kwargs)
    if plot_data:
        data_kwargs = plot_kwargs.copy()
        data_kwargs.update({'color': 'k', 'label': 'data'})
        sns.lineplot(x='scaling', y='proportion_correct', ax=axes[0], data=data,
                     marker='o', err_style='bars', linestyle='', **data_kwargs)
    axes[0].set(title='Data and psychophysical curve')
    sns.lineplot(x='iteration', y='loss', hue=hue, data=result, ax=axes[1],
                 legend=False, alpha=alpha, **plot_kwargs)
    if plot_mean:
        sns.lineplot(x='iteration', y='loss', data=result, ax=axes[1],
                     legend=False, color='k', **plot_kwargs)
    axes[1].set(title='Loss')
    sns.lineplot(x='iteration', y='proportionality_factor_history', hue=hue,
                 data=result, ax=axes[2], legend=False, alpha=alpha, color='k',
                 **plot_kwargs)
    if plot_mean:
        sns.lineplot(x='iteration', y='proportionality_factor_history',
                     data=result, ax=axes[2], legend=False, color='k',
                     **plot_kwargs)
    axes[2].set(title='Parameter history')
    sns.lineplot(x='iteration', y='critical_scaling_history', hue=hue, data=result,
                 ax=axes[3], legend=False, alpha=alpha, **plot_kwargs)
    if plot_mean:
        sns.lineplot(x='iteration', y='critical_scaling_history', color='k',
                     data=result, ax=axes[3], legend=False, **plot_kwargs)
    axes[3].set(title='Parameter history')
    reduced = pd.concat([result.groupby(hue).first().reset_index(),
                         result.groupby(hue).last().reset_index()])
    sns.lineplot(x='critical_scaling_history', y='proportionality_factor_history',
                 hue=hue, data=reduced, ax=axes[4], alpha=alpha, legend=False,
                 **plot_kwargs)
    sns.scatterplot(x='critical_scaling_history',
                    y='proportionality_factor_history', hue=hue,
                    style='iteration', data=reduced, ax=axes[4],
                    markers=['.', 'o'], alpha=alpha, **plot_kwargs)
    if plot_mean:
        # just plot the final one
        reduced = result.groupby(hue).last().reset_index()
        s_0 = np.percentile([reduced.critical_scaling.sample(frac=1, replace=True).mean()
                             for _ in range(1000)], [2.5, 50, 97.5])
        a_0 = np.percentile([reduced.proportionality_factor.sample(frac=1, replace=True).mean()
                             for _ in range(1000)], [2.5, 50, 97.5])
        ellipse = mpl.patches.Ellipse((s_0[1], a_0[1]), s_0[2] - s_0[0],
                                      a_0[2] - a_0[0], facecolor='k', alpha=.5,
                                      zorder=10)
        axes[4].add_patch(ellipse)
    axes[4].set(title='Initial and final parameter values')
    fig.tight_layout()
    return fig
