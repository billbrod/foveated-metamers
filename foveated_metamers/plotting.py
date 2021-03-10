#!/usr/bin/env python3
"""Misc plotting functions."""
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    estimator : callable
        what function to use for estimating central trend of the data
    x_jitter : float, bool, or None
        whether to jitter the data along the x-axis. if None or False,
        don't jitter. if a float, add uniform noise (drawn from
        -x_jitter to x_jitter) to each point's x value. if True, act as
        if x_jitter=.1
    x_dodge : float, None, or bool
        to improve visibility with many points that have the same
        x-values (or are categorical), we can jitter the data along the
        x-axis, but we can also "dodge" it, which operates
        deterministically. x_dodge should be either a single float or an
        array of the same shape as x (we will dodge by calling `x_data =
        x_data + x_dodge`). if None, we don't dodge at all. If True, we
        dodge as if x_dodge=.01
    x_order: np.array or None
        the order to plot x-values in. If None, don't reorder
    ci : int or 'hdi', optinoal
        The width of the CI to draw (in percentiles). If 'hdi', data must
        contain a column with 'hdi', which contains 50 and two other values
        giving the median and the endpoints of the HDI.

    Returns
    -------
    x_data : np.array
        the x data to plot
    plot_data : pd.Series
        the y data of the central trend
    plot_cis : list of pd.Series
        the y data of the CIs
    x_numeric : bool
        whether the x data is numeric or not (used to determine if/how
        we should update the x-ticks)

    """
    if ci == 'hdi':
        plot_data = data.set_index(x).query("hdi==50")[y]
        ci_vals = list(data.hdi.unique())
        ci_vals.remove(50)
        assert len(ci_vals) == 2, "should only have median and two endpoints for HDI!"
        plot_cis = [data.set_index(x).query("hdi==@val")[y] for val in ci_vals]
    else:
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
                    ci_alpha=.2, size=5, x_dodge=None, all_labels=None,
                    like_pointplot=False, **kwargs):
    """Plot center points and specified CIs, for use with map_dataframe.

    based on seaborn.linearmodels.scatterplot. CIs are taken from a
    distribution in this function. Therefore, it's assumed that the values
    being passed to it are values from a bootstrap distribution or a Bayesian
    posterior. seaborn.pointplot draws the CI on the *mean* (or other central
    tendency) of a distribution, not the distribution itself.

    by default, this draws the 68% confidence interval. to change this,
    change the ci argument. for instance, if you only want to draw the
    estimator point, pass ci=0

    Parameters
    ----------
    x : str
        which column of data to plot on the x-axis
    y : str
        which column of data to plot on the y-axis
    ci : int or 'hdi', optinoal
        The width of the CI to draw (in percentiles). If 'hdi', data must
        contain a column with 'hdi', which contains 50 and two other values
        giving the median and the endpoints of the HDI.
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
    all_labels : list or None, optional
        To dodge across hue, set `all_labels=list(df[hue].unique())` when
        calling `map_dataframe`. This will allow us to dodge each hue level by
        a consistent amount across plots, and make sure the data is roughly
        centered.
    like_pointplot: bool, optional
        If True, we tweak the aesthetics a bit (right now, just size of points
        and lines) to look more like seaborn's pointplot. Good when there's
        relatively little data. If True, this overrides the size option.
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
    if all_labels is not None and 'label' in kwargs.keys():
        orig_x_dodge = .01 if x_dodge is True else x_dodge
        x_dodge = all_labels.index(kwargs['label']) * orig_x_dodge
        x_dodge -= orig_x_dodge * ((len(all_labels)-1) / 2)
    if like_pointplot:
        # copying from how seaborn.pointplot handles this, because they look nicer
        lw = mpl.rcParams["lines.linewidth"] * 1.8
        # annoyingly, scatter and plot interpret size / markersize differently:
        # for plot, it's roughly the area, whereas for scatter it's the
        # diameter. In the following, we can use either; we use the sqrt of the
        # value here so it has same interpretation as the size parameter.
        warnings.warn(f"with like_pointplot, overriding user-specified size {size}")
        size = np.sqrt(np.pi * np.square(lw) * 2)
    else:
        # use default
        lw = mpl.rcParams['lines.linewidth']
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
        lines = ax.plot(x_data, plot_data.values, linewidth=lw,
                        markersize=size, **kwargs)
    else:
        lines = None
    # if we attach label to the CI, then the legend may use the CI
    # artist, which we don't want
    kwargs.pop('label', None)
    if ci_mode == 'lines':
        for x, (ci_low, ci_high) in zip(x_data, zip(*plot_cis)):
            cis = ax.plot([x, x], [ci_low, ci_high], linewidth=lw, **kwargs)
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
    if isinstance(x, str) and not isinstance(y, str):
        try:
            xmin, xmax = data[x].min(), data[x].max()
        except KeyError:
            # it looks like the above works with catplot / related functions
            # (i.e., when seaborn thought the data was categorical), but not
            # when it's relplot / related functions (i.e., when seaborn thought
            # data was numeric). in that case, the columns have been renamed to
            # 'x', 'y', etc.
            xmin, xmax = data['x'].min(), data['x'].max()
        # then this looks like a categorical plot
        if (ax.get_xlim()[-1] - xmax) / xmax > 5:
            xmin = 0
            xmax = data[x].nunique()-1
        lines = ax.hlines(y, xmin, xmax, linestyles=linestyles, colors=colors,
                          **kwargs)
    elif isinstance(y, str) and not isinstance(x, str):
        try:
            ymin, ymax = data[y].min(), data[y].max()
        except KeyError:
            # it looks like the above works with catplot / related functions
            # (i.e., when seaborn thought the data was categorical), but not
            # when it's relplot / related functions (i.e., when seaborn thought
            # data was numeric). in that case, the columns have been renamed to
            # 'x', 'y', etc.
            ymin, ymax = data['y'].min(), data['y'].max()
        # then this looks like a categorical plot
        if (ax.get_ylim()[-1] - ymax) / ymax > 5:
            ymin = 0
            ymax = data[y].nunique()-1
        lines = ax.vlines(x, ymin, ymax, linestyles=linestyles, colors=colors,
                          **kwargs)
    else:
        raise Exception("Exactly one of x or y must be a string!")
    return lines


def get_log_ax_lims(vals, base=10):
    """Get good limits for log-scale axis.

    Since several plotting functions have trouble automatically doing this.

    Parameters
    ----------
    vals : array_like
        The values plotted on the axis.
    base : float, optional
        The base of the axis

    Returns
    -------
    lims : tuple
        tuple of floats for the min and max value. Both will be of the form
        base**i, where i is the smallest / largest int larger / smaller than
        all values in vals.

    """
    i_min = 50
    while base**i_min > vals.min():
        i_min -= 1
    i_max = -50
    while base**i_max < vals.max():
        i_max += 1
    return base**i_min, base**i_max


def title_experiment_summary_plots(g, expt_df, summary_text, comparison='ref',
                                   post_text=''):
    """Handle suptitle for FacetGrids summarizing experiment.

    We want to handle the suptitle for these FacetGrids in a standard way:

    - Add suptitle that describes contents (what subjects, sessions,
      comparison)

    - Make sure it's visible

    Currently, used by: run_length_plot, compare_loss_and_performance_plot,
    performance_plot

    Parameters
    ----------
    g : sns.FacetGrid
        FacetGrid containing the figure.
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    summary_text : str
         String summarizing what's shown in the plot, such as "Performance" or
         "Run length". Will go at beginning of suptitle.
    comparison : {'ref', 'met'}, optional
        Whether this comparison is between metamers and reference images
        ('ref') or two metamers ('met').
    post_text : str, optional
        Text to put at the end of the suptitle, e.g., info on how to interpret
        the plot.

    Returns
    -------
    g : sns.FacetGrid
        The modified FacetGrid.

    """
    if expt_df.subject_name.nunique() > 1:
        subj_str = 'all subjects'
    else:
        subj_str = expt_df.subject_name.unique()[0]
    if 'session_number' not in expt_df.columns or expt_df.session_number.nunique() > 1:
        sess_str = 'all sessions'
    else:
        sess_str = f'session {expt_df.session_number.unique()[0]:02d}'
    comp_str = {'ref': 'reference images', 'met': 'other metamers'}[comparison]
    # got this from https://stackoverflow.com/a/36369238/4659293
    n_rows = g.fig.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]
    # we want to add some newlines at end of title, based on number of rows, to
    # make sure there's enough space
    end_newlines = ''
    if n_rows > 1:
        end_newlines += '\n'
    if n_rows > 3:
        end_newlines += '\n'
    g.fig.suptitle(f"{summary_text} for {subj_str}, {sess_str}."
                   f" Comparing metamers and {comp_str}. {post_text}{end_newlines}",
                   va='bottom')
    return g


def lineplot_like_pointplot(data, x, y, col=None, row=None, hue=None, ci=95,
                            col_wrap=None, ax=None, **kwargs):
    """Make a lineplot that looks like pointplot

    Pointplot looks nicer than lineplot for data with few points, but it
    assumes the data is categorical. This makes a lineplot that looks like
    pointplot, but can handle numeric data.

    Two modes here:

    - relplot: if ax is None, we call relplot, which allows for row and col to
      be set and creates a whole figure

    - lineplot: if ax is either an axis to plot on or `'map'` (in which case we
      grab `ax=plt.gca()`), then we call lineplot and create a single axis.
      Obviously, col and row can't be set, but you also have to be carefulf or
      how hue is mapped across multiple facets

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to plot
    x, y, col, row, hue : str or None
        The columns of data to plot / facet on those dimensions.
    ci : int, optional
        What size confidence interval to draw
    col_wrap : int or None
        How many columns before wrapping
    ax : {None, matplotlib axis, 'map'}, optional
        If None, we create a new figure using relplot. If axis, we plot on that
        axis using lineplot. If `'map'`, we grab the current axis and plot on
        that with lineplot.
    kwargs :
        passed to relplot / lineplot

    Returns
    -------
    g : FacetGrid or axis
        In relplot mode, a FacetGrid; in lineplot mode, an axis

    """
    # copying from how seaborn.pointplot handles this, because they look nicer
    lw = mpl.rcParams["lines.linewidth"] * 1.8
    # annoyingly, scatter and plot interpret size / markersize differently: for
    # plot, it's roughly the area, whereas for scatter it's the diameter. so
    # the following (which uses plot), should use sqrt of the value that gets
    # used in pointplot (which uses scatter). I also added an extra factor of
    # sqrt(2) (by changing the 2 to a 4 in the sqrt below), which looks
    # necessary
    ms = np.sqrt(np.pi * np.square(lw) * 4)
    if ax is None:
        if col is not None:
            col_order = kwargs.pop('col_order', sorted(data[col].unique()))
        else:
            col_order = None
        g = sns.relplot(x=x, y=y, data=data, kind='line', style=hue, col=col,
                        row=row, hue=hue, markers=data[hue].nunique()*['o'],
                        dashes=False, err_style='bars', ci=ci,
                        col_order=col_order, col_wrap=col_wrap, linewidth=lw,
                        markersize=ms, err_kws={'linewidth': lw}, **kwargs)
    else:
        if isinstance(ax, str) and ax == 'map':
            ax = plt.gca()
        ax = sns.lineplot(x=x, y=y, data=data, style=hue, hue=hue,
                          markers=data[hue].nunique()*['o'], dashes=False,
                          err_style='bars', ci=ci, linewidth=lw, markersize=ms,
                          err_kws={'linewidth': lw}, **kwargs)
        g = ax
    return g
