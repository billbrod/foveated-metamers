#!/usr/bin/env python3
"""Misc plotting functions."""
import numpy as np
import matplotlib.pyplot as plt


def _jitter_data(data, jitter):
    """Optionally jitter data some amount.

    jitter can be None / False (in which case no jittering is done), a number
    (in which case we add uniform noise with a min of -jitter, max of jitter),
    or True (in which case we do the uniform thing with min/max of -.1/.1)

    based on seaborn.linearmodels._RegressionPlotter.scatter_data

    """
    if jitter is None or jitter is False:
        return data
    else:
        if jitter is True:
            jitter = .1
        return data + np.random.uniform(-jitter, jitter, len(data))


def is_numeric(s):
    """Check whether data s is numeric.

    s should be something that can be converted to an array: list, Series,
    array, column from a DataFrame, etc

    this is based on the function
    seaborn.categorical._CategoricalPlotter.infer_orient.is_not_numeric

    Parameters
    ----------
    s :
        data to check

    Returns
    -------
    is_numeric : bool
        whether s is numeric or not

    """
    try:
        np.asarray(s, dtype=np.float)
    except ValueError:
        return False
    return True


def _map_dataframe_prep(data, x, y, estimator, x_jitter, x_dodge, x_order,
                        ci=68):
    """Prepare dataframe for plotting.

    Several of the plotting functions are called by map_dataframe and
    need a bit of prep work before plotting. These include:
    - computing the central trend
    - computing the CIs
    - jittering, dodging, or ordering the x values

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the info to plot
    x : str
        which column of data to plot on the x-axis
    y : str
        which column of data to plot on the y-axis
    estimator : callable, optional
        what function to use for estimating central trend of the data
    x_jitter : float, bool, or None, optional
        whether to jitter the data along the x-axis. if None or False,
        don't jitter. if a float, add uniform noise (drawn from
        -x_jitter to x_jitter) to each point's x value. if True, act as
        if x_jitter=.1
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same
        x-values (or are categorical), we can jitter the data along the
        x-axis, but we can also "dodge" it, which operates
        deterministically. x_dodge should be either a single float or an
        array of the same shape as x (we will dodge by calling `x_data =
        x_data + x_dodge`). if None, we don't dodge at all. If True, we
        dodge as if x_dodge=.01
    x_order: np.array or None, optional
        the order to plot x-values in. If None, don't reorder
    ci : int, optinoal
        The width fo the CI to draw (in percentiles)

    Returns
    -------
    x_data : np.array
        the x data to plot
    plot_data : np.array
        the y data of the central trend
    plot_cis : np.array
        the y data of the CIs
    x_numeric : bool
        whether the x data is numeric or not (used to determine if/how
        we should update the x-ticks)

    """
    plot_data = data.groupby(x)[y].agg(estimator)
    ci_vals = [50 - ci/2, 50 + ci/2]
    plot_cis = [data.groupby(x)[y].agg(np.percentile, val) for val in ci_vals]
    if x_order is not None:
        plot_data = plot_data.reindex(x_order)
        plot_cis = [p.reindex(x_order) for p in plot_cis]
    x_data = plot_data.index
    # we have to check here because below we'll end up making things
    # numeric
    x_numeric = is_numeric(x_data)
    if not x_numeric:
        x_data = np.arange(len(x_data))
    x_data = _jitter_data(x_data, x_jitter)
    # at this point, x_data could be an array or the index of a
    # dataframe. we want it to be an array for all the following calls,
    # and this try/except forces that
    try:
        x_data = x_data.values
    except AttributeError:
        pass
    if x_dodge is not None:
        if x_dodge is True:
            x_dodge = .01
        x_data = x_data + x_dodge
    return x_data, plot_data, plot_cis, x_numeric


def scatter_ci_dist(x, y, ci=68, x_jitter=None, join=False,
                    estimator=np.median, draw_ctr_pts=True, ci_mode='lines',
                    ci_alpha=.2, size=5, x_dodge=None, **kwargs):
    """Plot center points and specified CIs, for use with map_dataframe.

    based on seaborn.linearmodels.scatterplot. CIs are taken from a
    distribution in this function. Therefore, it's assumed that the
    values being passed to it are values from a bootstrap distribution.

    by default, this draws the 68% confidence interval. to change this,
    change the ci argument. for instance, if you only want to draw the
    estimator point, pass ci=0

    Parameters
    ----------
    x : str
        which column of data to plot on the x-axis
    y : str
        which column of data to plot on the y-axis
    ci : int, optinoal
        The width fo the CI to draw (in percentiles)
    x_jitter : float, bool, or None, optional
        whether to jitter the data along the x-axis. if None or False,
        don't jitter. if a float, add uniform noise (drawn from
        -x_jitter to x_jitter) to each point's x value. if True, act as
        if x_jitter=.1
    join : bool, optional
        whether to connect the central trend of the data with a line or
        not.
    estimator : callable, optional
        what function to use for estimating central trend of the data,
        as plotted if either draw_ctr_pts or join is True.
    draw_ctr_pts : bool, optional
        whether to draw the center points (as given by estimator).
    ci_mode : {'lines', 'fill'}, optional
        how to draw the CI. If 'lines', we draw lines for the CI. If
        'fill', we shade the region of the CI, with alpha given by
        ci_alpha
    ci_alpha : float, optional
        the alpha value for the CI, if ci_mode=='fill'
    size : float, optional
        Diameter of the markers, in points. (Although plt.scatter is
        used to draw the points, the size argument here takes a "normal"
        markersize and not size^2 like plt.scatter, following how it's
        done by seaborn.stripplot).
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same
        x-values (or are categorical), we can jitter the data along the
        x-axis, but we can also "dodge" it, which operates
        deterministically. x_dodge should be either a single float or an
        array of the same shape as x (we will dodge by calling `x_data =
        x_data + x_dodge`). if None, we don't dodge at all. If True, we
        dodge as if x_dodge=.01
    kwargs :
        must contain data. Other expected keys:
        - ax: the axis to draw on (otherwise, we grab current axis)
        - x_order: the order to plot x-values in. Otherwise, don't
          reorder
        everything else will be passed to the scatter, plot, and
        fill_between functions called (except label, which will not be
        passed to the plot or fill_between function call that draws the
        CI, in order to make any legend created after this prettier)

    Returns
    -------
    dots, lines, cis :
        The handles for the center points, lines connecting them (if
        join=True), and CI lines/fill. this is returned for better
        control over what shows up in the legend.

    """
    data = kwargs.pop('data')
    ax = kwargs.pop('ax', plt.gca())
    x_order = kwargs.pop('x_order', None)
    x_data, plot_data, plot_cis, x_numeric = _map_dataframe_prep(data, x, y,
                                                                 estimator,
                                                                 x_jitter,
                                                                 x_dodge,
                                                                 x_order, ci)
    if draw_ctr_pts:
        # scatter expects s to be the size in pts**2, whereas we expect
        # size to be the diameter, so we convert that (following how
        # it's handled by seaborn's stripplot)
        dots = ax.scatter(x_data, plot_data.values, s=size**2, **kwargs)
    else:
        dots = None
    if join is True:
        lines = ax.plot(x_data, plot_data.values, **kwargs)
    else:
        lines = None
    # if we attach label to the CI, then the legend may use the CI
    # artist, which we don't want
    kwargs.pop('label', None)
    if ci_mode == 'lines':
        for x, (ci_low, ci_high) in zip(x_data, zip(*plot_cis)):
            cis = ax.plot([x, x], [ci_low, ci_high], **kwargs)
    elif ci_mode == 'fill':
        cis = ax.fill_between(x_data, plot_cis[0].values, plot_cis[1].values,
                              alpha=ci_alpha, **kwargs)
    else:
        raise Exception(f"Don't know how to handle ci_mode {ci_mode}!")
    # if we do the following when x is numeric, things get messed up.
    if (x_jitter is not None or x_dodge is not None) and not x_numeric:
        ax.set(xticks=range(len(plot_data)),
               xticklabels=plot_data.index.values)
    return dots, lines, cis


def map_flat_line(x, y, data, linestyles='--', colors='k', ax=None, **kwargs):
    """Plot a flat line across every axis in a FacetGrid.

    For use with seaborn's map_dataframe, this will plot a horizontal or
    vertical line across all axes in a FacetGrid.

    Parameters
    ----------
    x, y : str, float, or list of floats
        One of these must be a float (or list of floats), one a str. The str
        must correspond to a column in the mapped dataframe, and we plot the
        line from the minimum to maximum value from that column. If the axes
        x/ylim looks very different than these values (and thus we assume this
        was a seaborn categorical plot), we instead map from 0 to
        data[x/y].nunique()-1
    The float
        corresponds to the x/y value of the line; if a list, then we plot
        multiple lines.
    data : pd.DataFrame
        The mapped dataframe
    linestyles, colors : str, optional
        The linestyles and colors to use for the plotted lines.
    ax : axis or None, optional
        The axis to plot on. If None, we grab current axis.
    kwargs :
        Passed to plt.hlines / plt.vlines.

    Returns
    -------
    lines : matplotlib.collections.LineCollection
        Artists for the plotted lines

    """
    if ax is None:
        ax = plt.gca()
    # we set color with the colors kwarg, don't want to confuse it.
    kwargs.pop('color')
    if isinstance(x, str):
        xmin, xmax = data[x].min(), data[x].max()
        # then this looks like a categorical plot
        if (ax.get_xlim()[-1] - xmax) / xmax > 5:
            xmin = 0
            xmax = data[x].nunique()-1
        lines = ax.hlines(y, xmin, xmax, linestyles=linestyles, colors=colors,
                          **kwargs)
    elif isinstance(y, str):
        ymin, ymax = data[y].min(), data[y].max()
        # then this looks like a categorical plot
        if (ax.get_ylim()[-1] - ymax) / ymax > 5:
            ymin = 0
            ymax = data[y].nunique()-1
        lines = ax.vlines(x, ymin, ymax, linestyles=linestyles, colors=colors,
                          **kwargs)
    else:
        raise Exception("One of x or y must be a str corresponding to a "
                        "column in the dataframe!")
    return lines
