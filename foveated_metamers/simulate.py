#!/usr/bin/env python3
"Code related to simulating data for curve fitting."
import torch
# this is a wrapper and drop-in replacement for multiprocessing:
# https://pytorch.org/docs/stable/multiprocessing.html
import torch.multiprocessing as mp
from . import curve_fit


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

    Creates final plot summarizing results (by calling
    ``curve_fit.multi_plot_optimization_results``)

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
    proportionality_factor : torch.tensor
        The final proportionality_factor values from each optimization
    critical_scaling : torch.tensor
        The final critical_scaling values from each optimization

    """
    simul_prop_corr = curve_fit.proportion_correct_curve(scaling,
                                                         proportionality_factor,
                                                         critical_scaling)
    seeds = list(range(n_opt))
    if use_multiproc:
        args = [(scaling, simul_prop_corr, lr, scheduler, max_iter, s) for s in
                seeds]
        # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork for why
        # we use spawn
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_processes) as pool:
            result = pool.starmap_async(curve_fit.fit_psychophysical_parameters, args)
            result = result.get()
    else:
        result = []
        for s in seeds:
            result.append(curve_fit.fit_psychophysical_parameters(scaling,
                                                                  simul_prop_corr,
                                                                  lr,
                                                                  scheduler,
                                                                  max_iter, s))
    fig = curve_fit.multi_plot_optimization_results(scaling, simul_prop_corr,
                                                    result)
    fig.axes[-1].scatter(critical_scaling, proportionality_factor, marker='s',
                         label='true value', color='k')
    # make sure we only grab artist for one initial, one final, and the true
    # value
    dots = [fig.axes[-1].collections[i] for i in [0, 1, -1]]
    fig.axes[-1].legend(dots, [d.get_label() for d in dots])
    a_0 = torch.cat([r[0] for r in result])
    s_0 = torch.cat([r[1] for r in result])
    return fig, a_0, s_0
