#!/usr/bin/env python3
"""Code related to simulating data for curve fitting."""
import torch
import numpy as np
import pandas as pd
from . import curve_fit
import plenoptic as po


def _plot_true_params(ax, proportionality_factor, critical_scaling):
    r"""Plot the true parameters on the parameter plot.

    We plot the real values as a black square and update the legend so it
    includes that square. This only makes sense for simulation, since otherwise
    we don't know what the true values are.

    Parameters
    ----------
    ax : axis
        The axis containing the parameter plot (the 5th and final one from
        ``curve_fit.plot_optimization_results``)

    Returns
    -------
    ax :
        The axis with the plot

    """
    ax.scatter(critical_scaling, proportionality_factor, marker='s',
               label='true value', color='k')
    # this new legend doesn't look great, but it works
    ax.legend()
    return ax


def test_optimization(proportionality_factor=5, critical_scaling=.2,
                      scaling=torch.logspace(-1, -.3, steps=9), n_opt=10,
                      use_multiproc=True, n_processes=None, lr=.001,
                      scheduler=True, max_iter=10000):
    r"""Test whether fit_psychophysical_parameters works.

    This simulates data with the specified parameters (for the given scaling
    values) and runs ``curve_fit.fit_psychophysical_parameters`` ``n_opt``
    times to see how good our optimization procedure is.

    It looks like, should run optimization multiple times, because sometimes it
    fails, and for ~5000 iterations per. When it fails, it's obvious
    (psychophysical curve is way off, loss fairly high, and parameter values
    haven't really moved much during optimization) and seems to be because the
    parameters were initialized too far from the actual values. It also has
    trouble if the critical_scaling is too far outside the examined scaling
    values.

    Creates final plot summarizing results, as well as dataframe summarizing
    the results.

    We set seeds for reproducibility (so multiple calls of this function will
    give the same result; to test more, increase ``n_opt``).

    Parameters
    ----------
    proportionality_factor : float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. Currently only handle single values
        for this (i.e., one curve)
    critical_scaling : float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study. Currently only handle single values
        for this (i.e., one curve)
    scaling : torch.tensor, optional
        The scaling values to test. Default corresponds roughly to V1 tested
        values.
    n_opt : int, optional
        Number of times to run optimization.
    use_multiproc : bool, optional
        Whether to use multiprocessing to parallelize across cores or not.
    n_processes : int or None, optional
        If ``use_multiproc`` is True, how many processes to run in parallel. If
        None, will be equal to ``os.cpu_count()``. If ``use_multiproc`` is
        False, this is ignored.
    lr : float, optional
        The learning rate for Adam optimizer.
    scheduler : bool, optional
        Whether to use the scheduler or not (reduces lr by half when loss
        appears to plateau).
    max_iter : int, optional
        The number of iterations to optimize for.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plots
    params : pd.DataFrame
        DataFrame containing the final parameter values from each optimization,
        as well as the true values.
    data : pd.DataFrame
        DataFrame containing the scaling and proportion_correct that results
        from the true parameters; this is the data we were trying to fit.

    """
    metadata = {'max_iter': max_iter, 'lr': lr, 'scheduler': scheduler,
                'n_opt': n_opt,
                'proportionality_factor_true': proportionality_factor,
                'critical_scaling_true': critical_scaling}
    simul_prop_corr = curve_fit.proportion_correct_curve(scaling,
                                                         proportionality_factor,
                                                         critical_scaling)
    data = pd.DataFrame({'scaling': scaling,
                         'proportion_correct': simul_prop_corr})
    kwargs = [{'scaling': scaling, 'proportion_correct': simul_prop_corr, 'lr':
               lr, 'scheduler': scheduler, 'max_iter': max_iter, 'seed': s}
              for s in range(n_opt)]
    results = curve_fit.multi_fit_psychophysical_parameters(kwargs,
                                                            use_multiproc,
                                                            n_processes,
                                                            [{'seed': s} for s
                                                             in range(n_opt)])
    fig = curve_fit.plot_optimization_results(data, results, hue='seed',
                                              plot_mean=True)
    _plot_true_params(fig.axes[-1], proportionality_factor, critical_scaling)
    results = results.drop_duplicates('seed')[['seed', 'critical_scaling',
                                               'proportionality_factor']]
    for k, v in metadata.items():
        results[k] = v
        data[k] = v
    return fig, results, data


def test_num_trials(num_trials, num_bootstraps, proportionality_factor=5,
                    critical_scaling=.2, scaling=torch.logspace(-1, -.3, steps=9),
                    use_multiproc=True, n_processes=None,
                    lr=.001, scheduler=True, max_iter=10000):
    r"""Test how many trials we need to be confident in our parameter estimates.

    We generate the true proportion correct at each scaling value for the
    specified parameters, then simulate ``num_trials`` psychophysical trials at
    each scaling value by sampling that many times from a Bernoulli
    distribution with that probability. We then bootstrap the proportion
    correct ``num_bootstraps`` times, running
    ``curve_fit.fit_psychophysical_parameters`` on the resulting data each
    time.

    Creates final plot summarizing results, as well as dataframe summarizing
    the results.

    Parameters
    ----------
    num_trials : int
        The number of trials to have per scaling value.
    num_bootstraps : int
        The number of times to bootstrap each scaling value (from the available
        trials)
    proportionality_factor : float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. Currently only handle single values
        for this (i.e., one curve)
    critical_scaling : float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study. Currently only handle single values
        for this (i.e., one curve)
    scaling : torch.tensor, optional
        The scaling values to test. Default corresponds roughly to V1 tested
        values.
    use_multiproc : bool, optional
        Whether to use multiprocessing to parallelize across cores or not.
    n_processes : int or None, optional
        If ``use_multiproc`` is True, how many processes to run in parallel. If
        None, will be equal to ``os.cpu_count()``. If ``use_multiproc`` is
        False, this is ignored.
    lr : float, optional
        The learning rate for Adam optimizer.
    scheduler : bool, optional
        Whether to use the scheduler or not (reduces lr by half when loss
        appears to plateau).
    max_iter : int, optional
        The number of iterations to optimize for.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plots
    params : pd.DataFrame
        DataFrame containing the final parameter values from each optimization,
        as well as the true values.
    data : pd.DataFrame
        DataFrame containing the scaling and proportion_correct from each
        bootstrap, as well as from the true parameters.

    """
    metadata = {'max_iter': max_iter, 'lr': lr, 'scheduler': scheduler,
                'num_trials': num_trials, 'num_bootstraps': num_bootstraps,
                'proportionality_factor_true': proportionality_factor,
                'critical_scaling_true': critical_scaling}
    true_prop_corr = curve_fit.proportion_correct_curve(scaling,
                                                        proportionality_factor,
                                                        critical_scaling)
    distribs = [torch.distributions.Bernoulli(p) for p in true_prop_corr]
    responses = torch.stack([d.sample((num_trials,)) for d in distribs])
    bootstrapped = []
    for b in range(num_bootstraps):
        samp = np.random.randint(num_trials, size=(len(scaling), num_trials))
        bootstraps = torch.zeros_like(responses)
        for j, idx in enumerate(samp):
            bootstraps[j] = responses[j, idx]
        bootstrapped.append(bootstraps)
    data = pd.concat([pd.DataFrame({'scaling': po.to_numpy(scaling),
                                    'proportion_correct': po.to_numpy(b.mean(1)),
                                    'bootstrap_num': i}) for i, b in
                      enumerate(bootstrapped)])
    kwargs = [{'scaling': scaling, 'proportion_correct': b.mean(1), 'lr':
               lr, 'scheduler': scheduler, 'max_iter': max_iter}
              for b in bootstrapped]
    results = curve_fit.multi_fit_psychophysical_parameters(kwargs, use_multiproc,
                                                            n_processes,
                                                            [{'bootstrap_num': b}
                                                             for b in range(num_bootstraps)])
    fig = curve_fit.plot_optimization_results(data, results,
                                              hue='bootstrap_num',
                                              plot_mean=False)
    _plot_true_params(fig.axes[-1], proportionality_factor, critical_scaling)
    fig.suptitle(f'{num_trials} trials', size='xx-large')
    fig.subplots_adjust(top=.88)
    results = results.drop_duplicates('bootstrap_num')[['bootstrap_num', 'critical_scaling',
                                                        'proportionality_factor']]
    # add the true values for data
    data = data.append(pd.DataFrame({'bootstrap_num': 'true_value',
                                     'scaling': scaling,
                                     'proportion_correct': true_prop_corr}))
    for k, v in metadata.items():
        results[k] = v
        data[k] = v
    return fig, results, data
