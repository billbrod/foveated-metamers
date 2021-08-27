#!/usr/bin/env python3
"""Misc plotting functions."""
import numpy as np
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path as op
import yaml
from . import mcmc, other_data
import scipy
import copy
import itertools
from collections import OrderedDict


def get_palette(col, col_unique=None, as_dict=True):
    """Get palette for column.

    Parameters
    ----------
    col : {'subject_name', 'model', 'scaling', 'cell_type', str}
        The column to return the palette for. If we don't have a particular
        palette picked out, the palette will contain the strings 'C0', 'C1',
        etc, which use the default palette.
    col_unique : list or None, optional
        The list of unique values in col, in order to determine how many
        elements in the palette. If None, we use seaborn's default
    as_dict : bool, optional
        Whether to return the palette as a dictionary or not.

    Returns
    -------
    pal : dict, list, or seaborn ColorPalette.
        palette to pass to plotting function

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        config = yaml.safe_load(f)
    psychophys_vars = config['PSYCHOPHYSICS']
    if col_unique is None:
        col_nunique = None
    else:
        col_nunique = len(col_unique)
    if col == 'subject_name':
        all_vals = psychophys_vars['SUBJECTS']
        pal = sns.color_palette('deep', len(all_vals))
    elif col == 'model':
        all_vals = [config['RGC']['model_name'], config['V1']['model_name']]
        if len(col_unique) == 1 and col_unique[0] in all_vals:
            pass
        elif sorted(col_unique) != sorted(all_vals):
            all_vals = ['Retina', 'V1']
            if len(col_unique) == 1 and col_unique[0] in all_vals:
                pass
            elif sorted(col_unique) != sorted(all_vals):
                raise Exception(f"Don't know what to do with models {col_unique}")
        assert len(all_vals) == 2, "Currently only support 2 model values"
        pal = sns.color_palette('BrBG', 3)
        pal = [pal[0], pal[-1]]
    elif col == 'scaling':
        # unlike others, we don't force this palette to be consistent across
        # possible values of scaling (and hence we don't have an all_vals)
        scaling_vals = sorted([c for c in col_unique if c != 'ref_image'])
        # we want the color to get lighter as scaling gets larger
        pal = sns.color_palette('Reds_r', len(scaling_vals))
        pal = dict(zip(scaling_vals, pal))
        pal['ref_image'] = 'k'
    elif col == 'cell_type':
        all_vals = ['parasol', 'midget']
        if len(col_unique) == 1 and col_unique[0] in all_vals:
            pass
        elif sorted(col_unique) != sorted(all_vals):
            all_vals = ['Retina', 'V1']
            if len(col_unique) == 1 and col_unique[0] in all_vals:
                pass
            elif sorted(col_unique) != sorted(all_vals):
                raise Exception(f"Don't know what to do with models {col_unique}")
        assert len(all_vals) == 2, "Currently only support 2 model values"
        pal = sns.color_palette('BrBG', 3)
        pal = [(*pal[0], .5), (*pal[0], 1)]
    elif col == 'image_name':
        all_vals = [i.replace('_symmetric', '') for i in config['IMAGE_NAME']['ref_image']]
        pal = sns.color_palette('husl', len(all_vals))
        all_vals = sorted(all_vals, key=lambda x: get_order('image_name').index(x))
        pal = dict(zip(all_vals, pal))
    else:
        if col_nunique is None:
            col_nunique = 10
        else:
            all_vals = col_unique
        pal = [f'C{i}' for i in range(col_nunique)]
    if as_dict and not isinstance(pal, dict):
        pal = dict(zip(sorted(all_vals), pal))
    return pal


def get_style(col, col_unique, as_dict=True):
    """Get style for plotting.

    Parameters
    ----------
    col : {'trial_type', 'mcmc_model_type', list containing both elements}
        The column to return the palette for. If we don't have a particular
        style picked out, we raise an exception. If it's a list containing both
        elements, we will combine the two style mappings.
    col_unique : list, optional
        The list of unique values in col, in order to determine elements to
        map. If col is a list of elements, this must be a list of list in the
        same order.
    as_dict : bool, optional
        Whether to return the palette as a dictionary or not.

    Returns
    -------
    style_dict : dict
        dict to unpack for plotting functions

    """
    def _combine_dicts(style_dict, style_key):
        """Combine dicts for multiple style args."""
        to_return = {}
        default_val = {'dashes_dict': '', 'markers': 'o',
                       'marker_adjust': {}}[style_key]
        dicts = itertools.product(*[v[style_key].items() for v in style_dict.values()])
        for item in dicts:
            # each of these will be a separate key in our final dictionary
            key = tuple([v[0] for v in item])
            if len(key) == 1:
                key = key[0]
            # ignore the default_val
            val = [v[1] for v in item if v[1] != default_val]
            # ... unless that's all there is
            if len(val) == 0:
                val = default_val
            # if there's a single thing that's not the default_val, we use that
            elif len(val) == 1:
                val = val[0]
            # else try and resolve conflicts
            else:
                if style_key != 'marker_adjust':
                    raise Exception(f"Don't know how to handle more than one value for {style_key}! {val}")
                marker_dict = {}
                for v in val:
                    for k, v_ in v.items():
                        if k not in marker_dict:
                            marker_dict[k] = v_
                        elif k == 'marker':
                            if marker_dict[k] == 'o':
                                marker_dict[k] = v_
                            elif v_ == 'o':
                                pass
                            else:
                                raise Exception(f"Don't know how to handle more than one value for {style_key} "
                                                f"key {k}! {v_} and {marker_dict[k]}")
                val = marker_dict
            to_return[key] = val
        return to_return

    if isinstance(col, str):
        col = [col]
        col_unique = [col_unique]
    style_dict = OrderedDict()
    for col_val, uniq in zip(col, col_unique):
        if col_val == 'trial_type':
            all_vals = ['metamer_vs_reference', 'metamer_vs_metamer',
                        'metamer_vs_metamer-natural', 'metamer_vs_metamer-downsample',
                        'metamer_vs_reference-natural']
            if any([c for c in uniq if c not in all_vals]):
                if all([c.startswith('trial_type_') for c in uniq]):
                    # this is the only exception we allow, which comes from
                    # simulated data
                    all_vals = uniq
                else:
                    raise Exception("Got unsupported value for "
                                    f"col='trial_type', {uniq}")
            dashes_dict = dict(zip(all_vals, ['', (2, 2), (2, 2), (2, 2), '']))
            dashes_dict = {k: v for k, v in dashes_dict.items() if k in uniq}
            # this (setting marker in marker_adjust and also below in the marker
            # dict) is a hack to allow us to determine which markers correspond to
            # which style level, which we can't do otherwise (they have no label)
            marker_adjust = {all_vals[1]:
                             {'fc': 'w', 'ec': 'original_fc', 'ew': 'lw',
                              's': 'total_unchanged', 'marker': 'o'},
                             all_vals[2]:
                             {'fc': 'w', 'ec': 'original_fc', 'ew': 'lw',
                              's': 'total_unchanged', 'marker': 's'},
                             all_vals[3]:
                             {'fc': 'w', 'ec': 'original_fc', 'ew': 'lw',
                              's': 'total_unchanged', 'marker': 'D'},
                             # this doesn't change the facecolor
                             all_vals[4]:
                             {'fc': None, 'ec': 'original_fc', 'ew': 'lw',
                              's': 'total_unchanged', 'marker': 's'},
                             }
            marker_adjust.update({c: {} for c in all_vals[:1]})
            marker_adjust = {k: v for k, v in marker_adjust.items() if k in uniq}
            markers = dict(zip(all_vals, ['o', 'v', '<', '>', '^']))
            markers = {k: v for k, v in markers.items() if k in uniq}
        elif col_val == 'mcmc_model_type':
            all_vals = ['unpooled', 'partially-pooled']
            marker_adjust = {c: {'marker': m} for c, m in
                             zip(all_vals, ['v', 'o'])}
            dashes_dict = dict(zip(all_vals, len(all_vals)*['']))
            markers = dict(zip(all_vals, len(all_vals)*['o']))
        else:
            raise Exception(f"Currently only support col='trial_type' or "
                            "'mcmc_model_type' but got {col_val}")
        style_dict[col_val] = {'dashes_dict': dashes_dict,
                               'marker_adjust': marker_adjust,
                               'markers': markers}
    overall_dict = {}
    overall_dict['dashes_dict'] = _combine_dicts(style_dict, 'dashes_dict')
    overall_dict['markers'] = _combine_dicts(style_dict, 'markers')
    overall_dict['marker_adjust'] = _combine_dicts(style_dict, 'marker_adjust')
    return overall_dict


def get_order(col, col_unique=[]):
    """Get order to plot something in.

    For rows, cols, or x.

    Parameters
    ----------
    col : {'image_name', str}
        The column to return the palette for. If we don't have a particular
        order picked out, will just sort col_unique.
    col_unique : list, optional
        List of unique values in col.

    Returns
    -------
    col_order : list
        List giving plot order.

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        config = yaml.safe_load(f)
    img_sets = config['PSYCHOPHYSICS']['IMAGE_SETS']
    img_order = (sorted(img_sets['all']) + sorted(img_sets['A']) +
                 sorted(img_sets['B']))
    img_order = [i.replace('symmetric_', '').replace('_range-.05,.95_size-2048,2600', '')
                 for i in img_order]
    if col == 'image_name':
        col_order = img_order
    else:
        col_order = sorted(col_unique)
    return col_order


def myLogFormat(y, pos):
    """formatter that only shows the required number of decimal points

    this is for use with log-scaled axes, and so assumes that everything greater than 1 is an
    integer and so has no decimal points

    to use (or equivalently, with `axis.xaxis`):
    ```
    from matplotlib import ticker
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    ```

    modified from https://stackoverflow.com/a/33213196/4659293
    """
    # Find the number of decimal places required
    if y < 1:
        # because the string representation of a float always starts "0."
        decimalplaces = len(str(y)) - 2
    else:
        decimalplaces = 0
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def _marker_adjust(axes, marker_adjust, label_map):
    """Helper function to adjust markers after the fact. Pretty hacky.

    Parameters
    ----------
    axes : list
        List of axes containing the markers to adjust
    marker_adjust : dict
        Dictionary with keys identifying the style level and values describing
        how to adjust the markers. Can contain the following keys: fc, ec, ew,
        s, marker (to adjust those properties). If a property is None or not
        included, won't adjust. In addition to the standard values those
        properties can take, can also take the following: - ec: 'original_fc'
        (take the original marker facecolor) - ew: 'lw' (take the linewidth) -
        s: 'total_unchanged' (adjust marker size so that, after changing the
        edge width, the overall size will not change)
    label_map : dict
        seaborn unfortunately doesn't label these markers, so we need to know
        how to match the marker to the style level. this maps the style level
        to the marker used when originally creating the plot, allowing us to
        identify the style level.

    Returns
    -------
    artists : dict
        dictionary between the style levels and marker properties, for creating
        a legend.

    """
    def _adjust_one_marker(line, fc=None, ec=None, ew=None, s=None, marker=None):
        original_fc = line.get_mfc()
        lw = line.get_lw()
        original_ms = line.get_ms() + line.get_mew()
        if ec == 'original_fc':
            ec = original_fc
        if ew == 'lw':
            ew = lw
        if s == 'total_unchanged':
            s = original_ms - ew
        if fc is not None:
            line.set_mfc(fc)
        if ec is not None:
            line.set_mec(ec)
        if ew is not None:
            line.set_mew(ew)
        if s is not None:
            line.set_ms(s)
        if marker is not None:
            line.set_marker(marker)
        return {'marker': line.get_marker(), 'mec': line.get_mec(),
                'mfc': line.get_mfc(), 'ms': line.get_ms(), 'mew': line.get_mew()}

    artists = {}
    for ax in axes:
        for line in ax.lines:
            if line.get_marker() and line.get_marker() != 'None':
                label = label_map[line.get_marker()]
                artists[label] = _adjust_one_marker(line, **marker_adjust[label])
    return artists


def _remap_image_names(df):
    """Prepare image names for plotting.

    This function remaps the image names in df to drop the parts that are
    duplicated across images ('symmetric_' and
    '_range_-.05,.95_size-2048,2600'), as well as grabbing the correct order

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the behavioral results or psychophysical curve
        fits to those results.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with image_name column remapped, if appropriate

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        config = yaml.safe_load(f)
    # this is the "full" version of the image names (including range and size),
    # as well as the "trimmed" one (without those two parts). we check each
    # separately, and can't have images in both sets
    all_imgs_both = [config['IMAGE_NAME']['ref_image'],
                     config['DEFAULT_METAMERS']['image_name']]
    remapped = False
    for all_imgs in all_imgs_both:
        # if we have down sampled images, we want to do same thing as the normal case
        if any([i.replace('_downsample-2', '') in all_imgs for i in df.image_name.unique()]):
            # while still gathering data, will not have all images in the df.
            # Adding these blank lines gives us blank subplots in the performance
            # plot, so that each image is in the same place
            extra_ims = [i for i in all_imgs if i.replace('_ran', '_downsample-2_ran') not in df.image_name.unique()
                         and i not in df.image_name.unique()]
            df = df.copy().append(pd.DataFrame({'image_name': extra_ims}), True)
            # strip out the parts of the image name that are consistent across
            # images
            df.image_name = df.image_name.apply(lambda x: x.replace('_symmetric', '').replace('_range-.05,.95_size-2048,2600', '').replace('_downsample-2', ''))
            assert df.image_name.nunique() == 20, "Something went wrong, don't have all images!"
            if not remapped:
                remapped = True
            else:
                raise Exception("Can't remap image names twice!")
        # for simulated data, our image names will just be 'image-00' (and we'll
        # have a variable number of them), and so we don't want to do the following
        else:
            if remapped:
                continue
    return df


def _psychophysical_curve_ticks(df, axes, logscale_xaxis=False, height=5,
                                col=None):
    """Set ticks appropriately for psychophysical curve plots.

    This assumes scaling is on the x-axis (named exactly that) and proportion
    correct is on the y-axis (name unconstrained). This sets major yticks at .5
    and 1, as well as minor ones at .4, .6, .7, .8, .9. For xticks, we try to
    determine a reasonable number, based on the size of the axis and the plot,
    and make sure they're equally-spaced.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the scaling values plotted.
    axes : list
        1d list (or array) containing the axes with the plots
    logscale_xaxis : bool, optional
        If True, we logscale the x-axis. Else, it's a linear scale.
    height : float, optional
        Height of each axis
    col : str or None, optional
        What is facetted along the plots columns

    """
    # it's difficult to come up with good tick values. this finds a somewhat
    # reasonable number of ticks in reasonable locations, reducing the number
    # if the axis is small or the font is large
    xmin = np.round(df.scaling.min() - .004, 2)
    xmax = np.round(df.scaling.max(), 2)
    nticks = 12
    if height < 6:
        nticks /= 2
    if mpl.rcParams['font.size'] > 15 or height < 2:
        nticks /= 2
        # don't want to overlap the labels on adjacent columns
        if col is not None:
            xmax -= (xmax-xmin)/10
    if logscale_xaxis:
        for ax in axes:
            ax.set_xscale('log', base=2)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: f"{y:.03f}"))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # if we make this an integer, our ticks go up by factors of 2, which
        # makes them a bit easier to interpret
        xtick_spacing = np.round((np.log2(xmax) - np.log2(xmin)) / (nticks-1))
        xticks = [2**(np.log2(xmin)+i*xtick_spacing) for i in range(int(nticks+1))]
        nticks = int(nticks+1)
        while xticks[-1] < xmax/2:
            xticks += [2**(np.log2(xmin)+nticks*xtick_spacing)]
            nticks += 1
        minor_xticks = [xticks[i]+(xticks[i+1]-xticks[i])/2
                        for i in range(nticks-1)]
    else:
        xtick_spacing = np.round((xmax - xmin) / (nticks-1), 2)
        xticks = [xmin+i*xtick_spacing for i in range(int(nticks+1))]
        minor_xticks = []
    for ax in axes:
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([.5, 1]))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([.4, .6, .7, .8, .9]))
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_xticks))


def _add_legend(df, g=None, fig=None, hue=None, style=None, palette={},
                final_markers={}, dashes_dict={}, legend_content='full'):
    """Add legend, making use of custom hue and style.

    Since we modify the markers after the fact, we can't rely on seaborn's
    built-in legend-generator. Instead, we create our own, based on how seaborn
    does it, showing both hue and style in separable manner.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the plotted data
    g : sns.FacetGrid
        FacetGrid containing the plots. One of this or fig must be set (not
        both.)
    fig : plt.Figure
        Figure containing the plots. One of this or g must be set (not both)
    hue, style : str or None, optional
        The columns in df that were uesd to facet hue and style, respectively.
    palette : dict, optional
        Must be non-empty if hue is not None. Dictionary between levels of hue
        variable and colors.
    final_markers : dict, optional
        Dictionary between levels of style variable and marker options, as
        returned by plotting._marker_adjust(). Unlike dashes_dict and palette,
        can be empty even if style is set (in which case we use default values).
    dashes_dict : dict, optional
        Must be non-empty if style is not None. Dictionary between levels of
        style variable and dash options, as passed to sns.relplot() or related
        functions
    legend_content : {'full', 'brief'}, optional
        Whether to include all hue levels or a sample of evenly spaced values
        (only if hue is numeric). NOTE: 'brief' currently only affects hue
        levels, all style levels are always included.

    """
    artists = {}
    if g is not None:
        ax = g.axes.flatten()[0]
    else:
        ax = fig.axes[0]
    lw = mpl.rcParams["lines.linewidth"] * 1.8
    if hue is not None:
        try:
            sorted_hue = get_order(hue, df[hue].unique())
        except TypeError:
            sorted_hue = sorted(df[hue].unique(), key=lambda x: 1e32 if isinstance(x, str) else x)
        if legend_content == 'brief':
            sorted_hue = [h for h in sorted_hue if is_numeric(h)]
            # definitely want the first and last value, and then three more,
            # roughly evenly spaced
            sorted_hue = (sorted_hue[:-1:len(sorted_hue)//4] + [sorted_hue[-1]])
            sorted_hue += [h for h in sorted_hue if not is_numeric(h)]
        artists[hue] = ax.scatter([], [], s=0)
        for hue_val in sorted_hue:
            if isinstance(hue_val, float) and np.isnan(hue_val):
                continue
            artists[hue_val] = ax.plot([], [], color=palette[hue_val],
                                       lw=lw)[0]
    if style is not None:
        if isinstance(style, str):
            style = [style]
        for sty in style:
            artists[sty] = ax.scatter([], [], s=0)
            sty_unique = [s for s in df[sty].unique() if isinstance(s, str) or
                          not np.isnan(s)]
            for style_val in sorted(sty_unique):
                if isinstance(style_val, float) and np.isnan(style_val):
                    continue
                style_key = [k for k in final_markers.keys() if style_val in k]
                # we want to grab the corresponding style_key. the following
                # should work
                if len(style_key) > 1:
                    style_key = [k for k in style_key if k == style_val][0]
                else:
                    style_key = style_key[0]
                markers = {k: v for k, v in final_markers[style_key].items()}
                markers['mec'] = 'k'
                markers['mfc'] = 'k' if markers['mfc'] != 'w' else 'w'
                markers['mew'] = lw
                artists[style_val] = ax.plot([], [], color='k', lw=lw,
                                             dashes=dashes_dict.get(style_key, []),
                                             **markers)[0]
    if artists:
        if g is not None:
            # in order for add_legend() to work, _hue_var and hue_names both
            # need to be None (otherwise, it will default to using them, even
            # when passing the artists dict)
            g._hue_var = None
            g.hue_names = None
            g.add_legend(artists)
        else:
            fig.legend(list(artists.values()), list(artists.keys()),
                       frameon=False, bbox_to_anchor=(.95, .5),
                       bbox_transform=fig.transFigure, loc='center left',
                       borderaxespad=0)


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
        plot_data = data.query("hdi==50").groupby(x)[y].agg(estimator)
        # remove np.nan and 50
        ci_vals = [v for v in data.hdi.unique() if not np.isnan(v) and v !=50]
        assert len(ci_vals) == 2, "should only have median and two endpoints for HDI!"
        plot_cis = [data.query("hdi==@val").groupby(x)[y].agg(estimator)
                    for val in ci_vals]
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
                    like_pointplot=False, style=None, dashes_dict={},
                    markers={}, **kwargs):
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
    style : str or None, optional
        columns from data to map along the style dimension
    dashes_dict : dict, optional
        dictionary mapping between style levels and args to pass as `dashes` to
        ax.plot.
    markers : dict, optional
        dictionary mapping between style levels and args to pass as `marker` to
        ax.scatter.
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
        size = np.sqrt(np.pi * np.square(lw) * 4)
    else:
        # use default
        lw = mpl.rcParams['lines.linewidth']
    data = kwargs.pop('data')
    ax = kwargs.pop('ax', plt.gca())
    x_order = kwargs.pop('x_order', None)
    if style is not None:
        data = data.groupby(style)
    else:
        data = [(None, data)]
    dots, lines, cis = [], [], []
    for n, d in data:
        if ci == 'hdi' and np.isnan(d.hdi.unique()).all():
            # then this is data that doesn't have any HDI, and so we don't want
            # to plot it with this function
            continue
        x_data, plot_data, plot_cis, x_numeric = _map_dataframe_prep(d, x, y,
                                                                     estimator,
                                                                     x_jitter,
                                                                     x_dodge,
                                                                     x_order,
                                                                     ci)
        dashes = dashes_dict.get(n, '')
        marker = copy.deepcopy(markers.get(n, mpl.rcParams['scatter.marker']))
        if draw_ctr_pts:
            if isinstance(marker, dict):
                # modify the dict (which is the kind that would be passed to
                # _marker_adjust) so it works with scatter.
                # for scatter, ew is called lw
                marker['lw'] = marker.pop('ew', None)
                if marker['lw'] == 'lw':
                    marker['lw'] = lw
                marker.pop('s', None)
                if marker.get('ec', None) == 'original_fc':
                    marker['ec'] = kwargs['color']
                # scatter expects s to be the size in pts**2, whereas we expect
                # size to be the diameter, so we convert that (following how
                # it's handled by seaborn's stripplot)
                dots.append(ax.scatter(x_data, plot_data.values, s=size**2,
                                       #want this to appear above the CI lines
                                       zorder=100, **kwargs, **marker))
            else:
                # scatter expects s to be the size in pts**2, whereas we expect
                # size to be the diameter, so we convert that (following how
                # it's handled by seaborn's stripplot)
                dots.append(ax.scatter(x_data, plot_data.values, s=size**2,
                                       marker=marker, **kwargs))
        else:
            dots.append(None)
        if join is True:
            lines.append(ax.plot(x_data, plot_data.values, linewidth=lw,
                                 markersize=size, dashes=dashes, **kwargs))
        else:
            lines.append(None)
        # if we attach label to the CI, then the legend may use the CI
        # artist, which we don't want
        kwargs.pop('label', None)
        if ci_mode == 'lines':
            for x_, (ci_low, ci_high) in zip(x_data, zip(*plot_cis)):
                cis.append(ax.plot([x_, x_], [ci_low, ci_high], linewidth=lw,
                                   **kwargs))
        elif ci_mode == 'fill':
            cis.append(ax.fill_between(x_data, plot_cis[0].values, plot_cis[1].values,
                                       alpha=ci_alpha, **kwargs))
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


def title_experiment_summary_plots(g, expt_df, summary_text, post_text=''):
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
    if expt_df.dropna().image_name.nunique() > 1:
        image_str = 'all images'
    elif expt_df.image_name.dropna().nunique() == 1:
        image_str = expt_df.image_name.dropna().unique()[0]
    else:
        image_str = ''
    if 'session_number' not in expt_df.columns or expt_df.session_number.nunique() > 1:
        sess_str = 'all sessions'
    else:
        sess_str = f'session {int(expt_df.session_number.unique()[0]):02d}'
    # we can have nans because of how we add blank rows to make sure each image
    # is represented
    model_name = ' and '.join([m.split('_')[0] for m in expt_df.model.dropna().unique()])
    if expt_df.trial_type.nunique() > 1 and expt_df.trial_type.unique()[0].startswith('metamer'):
        comparison = 'both'
    elif expt_df.trial_type.unique()[0].endswith('metamer'):
        comparison = 'met'
    elif expt_df.trial_type.unique()[0].endswith('reference'):
        comparison = 'ref'
    else:
        comparison = None
    comp_str = {'ref': 'reference images', 'met': 'other metamers',
                'both': 'both reference and other metamer images'}.get(comparison, "simulated")
    # got this from https://stackoverflow.com/a/36369238/4659293
    n_rows, n_cols = g.fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
    # we want to add some newlines at end of title, based on number of rows, to
    # make sure there's enough space
    end_newlines = ''
    break_newlines = ''
    if n_cols < 3:
        break_newlines += '\n'
    if n_rows > 1:
        end_newlines += '\n\n'
    if n_rows > 3:
        end_newlines += '\n'
    if n_rows > 10:
        end_newlines += '\n\n'
    g.fig.suptitle(f"{summary_text} for {subj_str}, {sess_str}, {image_str}.{break_newlines}"
                   f" Comparing {model_name} metamers to {comp_str}. {post_text}{end_newlines}",
                   va='bottom')
    return g


def _label_and_title_psychophysical_curve_plot(g, df, summary_text, ci=None, hdi=None):
    """Label and title plot.

    Does some changes to titles and labels to make the plots pretty.

    Parameters
    ----------
    g : sns.FacetGrid
        FacetGrid containing the plot
    df : pd.DataFrame
        Dataframe contanining the plotted data
    summary_text : str
        String summarizing what's shown in the plot, such as "Performance" or
        "Run length". Will go at beginning of suptitle.
    ci : None or int
        The CI plotted. One of ci or hdi must be None, one must be not-None.
    hdi : None or float
        The CI plotted. One of ci or hdi must be None, one must be not-None.
    
    """
    if ci is not None:
        ci_txt = f"{ci}% CI"
    elif hdi is not None:
        ci_txt = f"{hdi*100}% HDI"
    g.set_ylabels(f'Proportion correct (with {ci_txt})')
    g.set_xlabels('Scaling')
    g.set(ylim=(.3, 1.05))
    title_experiment_summary_plots(g, df, summary_text)
    g.set_titles('{col_name}')
    axes = g.axes.flatten()
    # got this from https://stackoverflow.com/a/36369238/4659293
    n_rows, n_cols = axes[0].get_subplotspec().get_gridspec().get_geometry()
    y_idx = n_cols * ((n_rows-1)//2)
    if n_rows % 2 == 0:
        yval = 0
    else:
        yval = .5
    x_idx = -((n_cols+1)//2)
    if n_cols % 2 == 0:
        xval = 0
    else:
        xval = .5
    # first axis will always have a ylabel and last one will always have an
    # xlabel
    ylabel = axes[0].get_ylabel()
    xlabel = axes[-1].get_xlabel()
    g.set(xlabel='', ylabel='')
    g.fig.subplots_adjust(hspace=.2, wspace=.1, top=1)
    axes[y_idx].set_ylabel(ylabel, y=yval, ha='center')
    axes[x_idx].set_xlabel(xlabel, x=xval, ha='center')


def fit_psychophysical_curve(x, y, hue=None, style=None, pal={}, dashes_dict={},
                             data=None, to_chance=False, **kwargs):
    """Fit psychophysical curve to mean data, intended for visualization only.

    For use with seaborn's FacetGrid.map_dataframe

    Parameters
    ----------
    x, y, hue, style : str
        columns from data to map along the x, y, hue, style dimensions
    pal : dict, optional
        dictionary mapping between hue levels and colors. if non-empty, will
        override color set in kwargs
    dashes_dict : dict, optional
        like pal, dictionary mapping between style levels and args to pass as
        `dashes` to ax.plot.
    data : pd.DataFrame or None, optional
        the data to plot
    to_chance : bool, optional
        if True, we extend the plotted x-values until y-values reach chance. If
        False, we just plot the included xvals.
    kwargs :
        passed along to plt.plot.

    """
    # copying from how seaborn.pointplot handles this, because they look nicer
    lw = mpl.rcParams["lines.linewidth"] * 1.8
    kwargs.setdefault('linewidth', lw)
    default_color = kwargs.pop('color')
    if hue is not None:
        data = data.groupby(hue)
    elif 'hue' in data.columns and not data.hue.isnull().all():
        data = data.groupby('hue')
    else:
        data = [(None, data)]
    for n, d in data:
        color = pal.get(n, default_color)
        if style is not None:
            d = d.groupby(style)
        elif 'style' in d.columns and not d['style'].isnull().all():
            d = d.groupby('style')
        else:
            d = [(None, d)]
        for m, g in d:
            dashes = dashes_dict.get(m, '')
            try:
                g = g.groupby(x)[y].mean()
            except KeyError:
                # it looks like the above works with catplot / related functions
                # (i.e., when seaborn thought the data was categorical), but not
                # when it's relplot / related functions (i.e., when seaborn thought
                # data was numeric). in that case, the columns have been renamed to
                # 'x', 'y', etc.
                g = g.groupby('x').y.mean()
            try:
                popt, _ = scipy.optimize.curve_fit(mcmc.proportion_correct_curve, g.index.values,
                                                   g.values, [5, g.index.min()],
                                                   # both params must be non-negative
                                                   bounds=([0, 0], [np.inf, np.inf]),
                                                   maxfev=1000)
            except ValueError:
                # this happens when this gets called on an empty facet, in which
                # case we just want to exit out
                continue
            xvals = g.index.values
            yvals = mcmc.proportion_correct_curve(xvals, *popt)
            if to_chance:
                while yvals.min() > .5:
                    xvals = np.concatenate(([xvals.min()-xvals.min()/10], xvals))
                    yvals = mcmc.proportion_correct_curve(xvals, *popt)
            ax = kwargs.pop('ax', plt.gca())
            ax.plot(xvals, yvals, dashes=dashes, color=color, **kwargs)


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
    kwargs.setdefault('dashes', False)
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
        g = sns.relplot(x=x, y=y, data=data, kind='line', col=col, row=row,
                        hue=hue, marker='o', err_style='bars',
                        ci=ci, col_order=col_order, col_wrap=col_wrap,
                        linewidth=lw, markersize=ms, err_kws={'linewidth': lw},
                        **kwargs)
    else:
        if isinstance(ax, str) and ax == 'map':
            ax = plt.gca()
        ax = sns.lineplot(x=x, y=y, data=data, hue=hue, marker='o',
                          err_style='bars', ci=ci, linewidth=lw, markersize=ms,
                          err_kws={'linewidth': lw}, **kwargs)
        g = ax
    return g


def _setup_facet_figure(df, col=None, row=None, col_order=None, row_order=None,
                        aspect=1, height=2.5, gridspec_kw={},
                        rotate_xticklabels=False, **kwargs):
    """Setup figure for custom facetting.

    Goal is to have a figure we can plot different facets on, similar to how
    sns.FacetGrid works but with slightly more control.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data we'll plot.
    col, row : str or None, optional
        variables to plot on axes or facet along.
    col_order, row_order : list or None, optional
        order to plot those variables in. If None, we'll use the default.
    aspect : float, optional
        aspect ratio of the subplots
    height : float, optional
        height of the subplots
    gridspec_kw : dict, optional
        dictionary defining the figure's gridspec.
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not.
    kwargs :
        passed to plt.subplots

    Returns
    -------
    fig : plt.Figure
        figure to plot on
    axes : np.ndarray
        2d array of axes (index into with row, column) from figure
    cols, rows : list
        list of values in col, row (respectively) in order to plot them. If col
        or row is None, the corresponding list will be empty.
    x_order : list or None
        order to plot x in

    """
    if col_order is None:
        col_order = get_order(col, df[col].unique())
    if row_order is None:
        row_order = get_order(row, df[row].unique())
    if col is not None:
        cols = sorted(df[col].unique(), key=lambda x: col_order.index(x))
    else:
        cols = []
    if row is not None:
        rows = sorted(df[row].unique(), key=lambda x: row_order.index(x))
    else:
        rows = []
    if 'hspace' not in gridspec_kw:
        hspace = .15
        if rotate_xticklabels and len(rows) > 1:
            hspace += .1
        gridspec_kw['hspace'] = hspace
    fig, axes = plt.subplots(nrows=max(1, len(rows)), ncols=max(1, len(cols)),
                             figsize=(aspect*len(cols)*height,
                                      len(rows)*height,), squeeze=False,
                             gridspec_kw=gridspec_kw, **kwargs)
    return fig, axes, cols, rows


def _prep_labels(df, hue=None, style=None, col=None, row=None):
    """Prepare labels for custom legend.

    In order to create our custom legend for our custom facetting, need some
    extra info.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data we'll plot.
    hue, style, col, row : str or None, optional
        variables to plot on axes or facet along.

    Returns
    -------
    label : str or None
        Whether we'll be using style, hue, both or neither to label data for
        the legend. Intended to be passed to _facetted_scatter_ci_dist.
    all_labels : list
        List of possible labels for the legend. Intended to be passed to
        _facetted_scatter_ci_dist.

    """
    if hue is not None and (style is None or style == col or style == row):
        all_labels = list(df[hue].unique())
        label = 'hue'
    elif style is not None and hue is None:
        all_labels = list(df[style].unique())
        label = 'style'
    elif style is not None and hue is not None:
        try:
            all_labels = [n for n, _ in df.groupby([style, hue])]
            gb_style = [style]
        except ValueError:
            # then style is a list
            gb_style = [s for s in style if s != row and s != col]
            all_labels = [n for n, _ in df.groupby([*gb_style, hue])]
        label = 'both'
    else:
        all_labels = [None]
        label = None
    return label, all_labels


def _facetted_scatter_ci_dist(data, x, y, hue=None, style=None, x_order=None,
                              label=None, all_labels=[None], x_dodge=None,
                              marker_adjust={}, palette={},
                              rotate_xticklabels=False, xlabel="", ylabel="",
                              title_str="", color='k', ax=None):
    """Use scatter_ci_dist, with style and hue, on a specific axis.

    Meant for use with our "custom facetting", this handles much of the
    boilerplate stuff (properly arranging style and hue, assembling
    final_markers dict, titling / labeling).

    NOTE: this is intended for use with a *single* axis, but allows multiple
    hue/ style values.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    x, y : str
        Column in data with the info to plot along x and y.
    hue, style : str or None, optional
        Column in data with the info to facet along hue or style.
    x_order : list or None, optional
        Order to plot the x values in. If None, use natural sort.
    label : str or None
        Whether we'll be using style, hue, both or neither to label data for
        the legend.
    all_labels : list
        List of possible labels for the legend.
    x_dodge : float, None, or bool, optional
        to improve visibility with many points that have the same x-values (or
        are categorical), we can dodge the data along the x-axis,
        deterministically shifting it. If a float, x_dodge is the amount we
        shift each level of hue by; if None, we don't dodge at all; if True, we
        dodge as if x_dodge=.01
    marker_adjust : dict, optional
        Dictionary with keys identifying the style level and values describing
        how to adjust the markers. Can contain the following keys: fc, ec, ew,
        s, marker (to adjust those properties). If a property is None or not
        included, won't adjust. In addition to the standard values those
        properties can take, can also take the following: - ec: 'original_fc'
        (take the original marker facecolor) - ew: 'lw' (take the linewidth) -
        s: 'total_unchanged' (adjust marker size so that, after changing the
        edge width, the overall size will not change)
    palette : dict, optional
        Dictionary mapping levels of hue to color. If a level is not found in
        palette, will use `color`.
    rotate_xticklabels : bool or int, optional
        whether to rotate the x-axis labels or not. if True, we rotate
        by 25 degrees. if an int, we rotate by that many degrees. if
        False, we don't rotate.
    xlabel, ylabel, title_str : str, optional
        Labels for the x-axis, y-axis, and axis title, respectively. If empty,
        will not add.
    color : str or tuple, optional
        Any color that can be interpreted by matplotlib. The default color if a
        hue level is not found in `palette`
    ax : axis or None
        The axis to plot on. If None, we grab current axis.

    Returns
    -------
    final_markers : dict
        dict specifying the mapping between style level and plot style. You
        should aggregate these over multiple calls to this function, then pass
        to `plotting._add_legend`

    """
    if ax is None:
        ax = plt.gca()
    if hue is None:
        hue_gb = [(None, data)]
    else:
        hue_gb = data.groupby(hue)
    if rotate_xticklabels is True:
        rotate_xticklabels = 25
    final_markers = {}
    for n, g in hue_gb:
        if style is None:
            style_gb = [(None, g)]
        else:
            style_gb = g.groupby(style)
        for m, h in style_gb:
            if label == 'hue':
                lab = n
            elif label == 'style':
                lab = m
            elif label == 'both':
                if isinstance(style, list) and len(style) > 1:
                    lab_m = [m_ for m_ in m if
                             any([m_ in l for l in all_labels])]
                    lab = (*lab_m, n)
                else:
                    lab = (m, n)
            else:
                lab = label
            dots, _, _ = scatter_ci_dist(x, y, x_dodge=x_dodge,
                                         estimator=np.mean,
                                         all_labels=all_labels,
                                         like_pointplot=True, ci='hdi',
                                         markers=marker_adjust,
                                         style=style, x_order=x_order,
                                         color=palette.get(n, color), data=h,
                                         label=lab, ax=ax)
            if m is not None:
                fc = dots[0].get_facecolor()[0]
                if all(fc == 1):
                    fc = 'w'
                marker_dict = {'marker': marker_adjust[m].get('marker', 'o'),
                               'mew': dots[0].get_lw()[0], 'mfc': fc,
                               'ms': np.sqrt(dots[0].get_sizes()[0]),
                               'mec': dots[0].get_edgecolor()[0]}
                final_markers[m] = marker_dict
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title_str:
        ax.set_title(title_str)
    if rotate_xticklabels:
        labels = ax.get_xticklabels()
        if labels:
            ax.set_xticklabels(labels, rotation=rotate_xticklabels,
                               ha='right')
    return final_markers


def _draw_triangle(ax, top_pt, side_length=.1, rotation_angle=0, **kwargs):
    """Draw equilateral triangle on axis to point something out.

    Will not change axes limits.

    Parameters
    ----------
    ax : axis
        The axis to add the triangle to.
    top_pt : tuple
        The (x, y) tuple (in data coordinates) of the "top point" of the
        triangle. This is the value the triangle will be pointing to.
    side_length : float, optional
        The length of the triangle (in axes coordinates)
    rotation_angle : float, optional
        The angle (in radians) of the triangles rotation. 0 gives an angle with
        its top_pt pointing towards the top of the axis and the opposite side
        parallel to the x-axis.
    kwargs :
        Passed to mpl.patches.Polygon

    Returns
    -------
    xy : np.ndarray
        2d array with 3 values giving the x, y values of the three triangle
        points (in axes coordinates)

    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # this is the ratio between the x length and y length (so >1 means it's
    # wider than it is tall). we use it to correct for any stretching, because
    # otherwise, if the axis is wider than it is tall, axis coordinate of .1 in
    # the x direction is larger than axis coordinate of .1 in the y direction.
    aspect = np.divide(*ax.bbox.size)
    if rotation_angle > np.pi*2 or rotation_angle < -np.pi*2:
        raise Exception("rotation_angle should be in radians between -2pi and"
                        f" 2pi, but got {rotation_angle}!")
    # this converts the top pt from data coordinates to axes coordinates. this
    # makes it easier to be consistent with sizing (e.g., to have all sizes the
    # same length even when axis is logscaled)
    x3, y3 = (ax.transScale+ax.transLimits).transform(top_pt)
    x1 = x3 - side_length/aspect * np.cos(rotation_angle + np.pi/3)
    y1 = y3 - side_length * np.sin(rotation_angle + np.pi/3)
    x2 = x1 + side_length/aspect * np.cos(rotation_angle)
    y2 = y1 + side_length * np.sin(rotation_angle)
    xy = np.array([(x3, y3), (x1, y1), (x2, y2)])
    triangle = mpl.patches.Polygon(xy, transform=ax.transAxes, **kwargs)
    ax.add_patch(triangle)
    ax.set(xlim=xlim, ylim=ylim)
    return xy


def _draw_rectangle(ax, top_pts, height=.1, **kwargs):
    """Draw rectangle on axis to show range.

    Will not change axes limits.

    Parameters
    ----------
    ax : axis
        The axis to add the triangle to.
    top_pts : List
        List of (x, y) tuples (in data coordinates) of the "top points" of the
        rectangle.
    height : float, optional
        The height of the rectangle (in axes coordinates)
    kwargs :
        Passed to mpl.patches.Polygon

    Returns
    -------
    xy : np.ndarray
        2d array with 4 values giving the x, y values of the three triangle
        points (in axes coordinates)

    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # this converts the top pt from data coordinates to axes coordinates. this
    # makes it easier to be consistent with sizing (e.g., to have all sizes the
    # same length even when axis is logscaled)
    x1, y1 = (ax.transScale+ax.transLimits).transform(top_pts[0])
    x2, y2 = (ax.transScale+ax.transLimits).transform(top_pts[1])
    assert y1 == y2, "two points in top_pts should have same y value!"
    if x2 < x1:
        x_, y_ = x2, y2
        x2, y2 = x1, y1
        x1, y1 = x_, y_
    rectangle = mpl.patches.Rectangle((x1, y1-height), x2-x1, height,
                                      transform=ax.transAxes, **kwargs)
    ax.add_patch(rectangle)
    ax.set(xlim=xlim, ylim=ylim)
    xy = np.array([(x1, y1), (x2, y2), (x1, y1-height), (x2, y2-height)])
    return xy


def add_physiological_scaling_arrows(ax, side_length=.05, midget_rgc=True,
                                     parasol_rgc=True, v1=True, **kwargs):
    """Draw arrows pointing out physiological scaling values.

    These are point estimates, just taken from looking at figures, the more
    accurate version is `add_physiological_scaling_bars`.

    Parameters
    ----------
    ax : axis
        The axis to add the triangle to.
    side_length : float, optional
        The length of the triangle (in axes coordinates)
    midget_rgc, parasol_rgc, v1 : bool, optional
        Whether to include the arrows for midget RGC, parasol RGC, and V1
        physiological scaling, respectively.
    kwargs :
        Passed to mpl.patches.Polygon. Cannot contain color.

    """
    pal = get_palette('model', ['Retina', 'V1'])
    vals = {'midget_rgc': ['Retina', .01, 'Midget'],
            'parasol_rgc': ['Retina', .03, 'Parasol'],
            'v1': ['V1', .25, 'V1']}
    # this is axes coordinates
    triangle_height = np.sqrt(side_length**2 - (side_length/2)**2)
    for k, v in vals.items():
        if not eval(k):
            continue
        model, scaling, label = v
        xy = _draw_triangle(ax, (scaling, .5), side_length, color=pal[model])
        txt = ax.text(xy[0][0], xy[0][1] - triangle_height - triangle_height/4,
                      label, ha='center', va='top', transform=ax.transAxes)
        # get_window_extent() is in display units, and transAxes.inverted()
        # puts it back into axes coordinates, so a negative values mean its
        # outside the axes.
        while ax.transAxes.inverted().transform(txt.get_window_extent())[0, 0] < .005:
            xy[0][0] += .01
            txt.set_visible(False)
            txt = ax.text(xy[0][0], xy[0][1] - triangle_height - triangle_height/4,
                          label, ha='center', va='top', transform=ax.transAxes)


def add_physiological_scaling_bars(ax, inf_data, bar_height=.04,
                                   midget_rgc=True, parasol_rgc=True, v1=True,
                                   **kwargs):
    """Draw bars showing range of physiological scaling values.

    We get the range of V1 values from using WebPlotDigitzer on Figure 1G of
    the Wallis, 2019 paper. For RGC, we fit our own line to data from Dacey
    1992 paper.

    Parameters
    ----------
    ax : axis
        The axis to add the triangle to.
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by
        `other_data.run_phys_scaling_inference`.
    bar_height : float, optional
        The height of the bar to draw (in axes coordinates).
    midget_rgc, parasol_rgc, v1 : bool, optional
        Whether to include the bars for midget RGC, parasol RGC, and V1
        physiological scaling, respectively.
    kwargs :
        Passed to mpl.patches.Polygon. Cannot contain color.

    """
    pal = get_palette('model', ['Retina', 'V1'])
    pal.update(get_palette('cell_type', ['midget', 'parasol']))
    pal = {k.lower(): v for k, v in pal.items()}
    df = other_data.inf_data_to_df(inf_data, 'parameters', hdi=True,
                                   query_str="distribution=='posterior'")
    vals = {'v1': [(.125, .265), 'V1']}
    for n in ['midget', 'parasol']:
        g = df.query("cell_type==@n")
        vals[n+'_rgc'] = [(g.query("variable=='diameter_slope'").value.min(),
                           g.query("variable=='diameter_slope'").value.max()),
                          n.capitalize()]
    for k, v in vals.items():
        if not eval(k):
            continue
        top_pts, label = v
        xy = _draw_rectangle(ax, [(p, .5) for p in top_pts], bar_height,
                             color=pal[label.lower()])
        txt = ax.text(xy[0][0] + (xy[1][0] - xy[0][0])/2,
                      xy[-1][1] - bar_height/4,
                      label, ha='center', va='top', transform=ax.transAxes)
        # get_window_extent() is in display units, and transAxes.inverted()
        # puts it back into axes coordinates, so a negative values mean its
        # outside the axes.
        while ax.transAxes.inverted().transform(txt.get_window_extent())[0, 0] < .005:
            xy[0][0] += .01
            txt.set_visible(False)
            txt = ax.text(xy[0][0] + (xy[1][0] - xy[0][0])/2,
                          xy[-1][1] - bar_height/4,
                          label, ha='center', va='top', transform=ax.transAxes)


def _spectra_dataset_to_dataframe(spectra, data='sf'):
    """Convert spectra xarray dataset to pandas dataframe.

    Parameters
    ----------
    spectra : xarray.Dataset
        Dataset containing the spectra for synthesized metamers and our natural
        reference images.
    data : {'sf', 'orientation'}, optional
        Whether to grab the spatial frequency or orientation info

    Returns
    -------
    df : pd.DataFrame

    """
    if data == 'sf':
        cols = ['freq_n']
    elif data == 'orientation':
        cols = ['orientation_slice', 'samples']
    else:
        raise Exception("data must be one of {'sf', 'orientation'} but "
                        f"got {data}")
    df = spectra[f'ref_image_{data}_amplitude'].to_dataframe().reset_index()
    met_df = spectra[f'metamer_{data}_amplitude'].to_dataframe().reset_index()
    # give the ref image rows dummy values
    df['scaling'] = 'ref_image'
    df['seed_n'] = 0
    df = df.melt(cols + ['image_name', 'model', 'scaling', 'seed_n',
                         'trial_type'],
                 var_name='image_type', value_name=f'{data}_amplitude')
    met_df = met_df.melt(cols + ['image_name', 'model', 'scaling', 'seed_n',
                                 'trial_type'],
                         var_name='image_type', value_name=f'{data}_amplitude')
    df = pd.concat([df, met_df])
    df.image_type = df.image_type.apply(lambda x: x.replace(f'_{data}_amplitude', ''))
    return df
