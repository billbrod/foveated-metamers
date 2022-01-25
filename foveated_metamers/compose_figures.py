#!/usr/bin/env python3
"""functions to put the final touches on figures for publication
"""
import re
from svgutils import compose
from . import style
import matplotlib as mpl
import tempfile
import matplotlib.pyplot as plt


def calc_scale(created_in='matplotlib'):
    """Convert to correct size in pixels.

    There's a bizarre issue that makes it so the size of the figure saved out
    from matplotlib as an svg is not the same as the SVG element created here.
    this wouldn't be an issue (because SVGs scale nicely and all that) except
    for the font size. So, here's the issue:

    - matplotlib saves out svgs, specifying the size in the header in pts,
      converting from inches to pts with a hard-coded dpi of 72 (see definition
      of class FigureCanvasSVG, defind in backend_svg.py)

    - svgutils loads in svgs, but assumes the values in the height/width
      definitions are in pixels, not points, and that pixels have a hard-coded
      dpi of 90

    Therefore, in order to make sure everything is the right size, we need to
    scale all SVG figures up by 90/72

    weirdly, ImageMagick (via identify on the command line) correctly does the
    conversion

    Similarly for inkscape, except it uses a dpi of 96 for pixels.

    Parameters
    ----------
    created_in : {'matplotlib', 'svgutils', 'inkscape'}, optional
       which program created this svg, which determines the dpi used: 90 for
       svgutils, 72 for matplotlib, and 96 for inkscape.

    Returns
    -------
    scale : float

    """
    dpi = {'matplotlib': 72, 'svgutils': 72, 'inkscape': 96}[created_in]
    return 90/dpi


class SVG(compose.SVG):
    """SVG from file.

    This is the same as svgutils.compose.SVG, except we automatically scale it
    appropriately (see docstring of calc_scale() for details)

    Parameters
    ----------
    fname : str
       full path to the file
    created_in : {'matplotlib', 'svgutils', 'inkscape'}, optional
       which program created this svg, which determines the dpi used: 90 for
       svgutils, 72 for matplotlib, and 96 for inkscape.

    """

    def __init__(self, fname=None, created_in='matplotlib'):
        super().__init__(fname)
        self.scale(calc_scale(created_in))


def _convert_to_pix(val):
    """Convert value into pixels to make our lives easier."""
    if not isinstance(val, str):
        return val
    else:
        v, units = re.findall('([\d\.]+)([a-z]+)', val)[0]
        # svgutils can't convert out of inches or pts, so we do that ourselves
        # (it supposedly can do pts, but it says there is pt per inch? which is
        # just totally wrong)
        if units == 'in':
            return float(v) * 90
        elif units == 'pt':
            return float(v) * (90/72)
        else:
            return compose.Unit(val).to('px').value


def _create_tmp_rectangle(height=1.7, width=6.5, **kwargs):
    """Create a svg in a temporary file that just contains a single rectangle.

    Parameters
    ----------
    height, width : float
        height and width of rectangle, specified in inches.
    kwargs :
        passed to mpl.patches.Rectangle

    Returns
    -------
    filename : str
        path to the temporary file

    """
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.set_visible(False)
    rect = mpl.patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 **kwargs)
    fig.add_artist(rect)
    name = tempfile.NamedTemporaryFile().name + '.svg'
    fig.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    return name


def model_schematic(schematic_fig, contour_fig_large, contour_figs_small,
                    fig_size='full', context='paper'):
    """Add window contours to the model schematic figure.

    Parameters
    ----------
    schematic_fig, contour_fig_large, contour_figs_small : str
        Paths to the model schematic figure and two versions of the window
        contour figure, one which will be displayed relatively large and one
        that will be a bit less than half that size (and should also have a
        white background). contour_figs_small can be a single str or a list of 4
        strs, in which case we'll use each of those separately.
    fig_size : {'full', 'half'}, optional
        We have two different versions of this figure, one full-width, one half.
        This specifies which we're making.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    Returns
    -------
    fig : svgutils.compose.Figure
        Figure containing composed plots

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils',
                                                     fig_size)
    figure_width = _convert_to_pix(figure_width)
    schematic_fig = SVG(schematic_fig, 'inkscape')
    if fig_size == 'full':
        scales = [.06, .0255]
        positions = [(209, 51), (417+8, 302-15), (417, 302),
                     (388+8, 362-15), (388, 362)]
    elif fig_size == 'half':
        scales = [.0395, .0151]
        positions = [(160, 17), (245.5+5.5, 172.5-8.5), (245.5, 172.5),
                     (228+5.5, 209-8.5), (228, 209)]
    if isinstance(contour_figs_small, str):
        contour_figs_small = [contour_figs_small] * 4
    return compose.Figure(
        figure_width, schematic_fig.height * calc_scale('inkscape'),
        schematic_fig,
        SVG(contour_fig_large).scale(scales[0]).move(*positions[0]),
        *[SVG(fig).scale(scales[1]).move(*positions[i+1])
          for i, fig in enumerate(contour_figs_small)],
    )


def metamer_comparison(metamer_fig, labels, cutout_fig=False,
                       natural_seed_fig=False, context='paper'):
    """Add text labeling model metamer scaling values.

    Parameters
    ----------
    metamer_fig : str
        Path to the metamer comparison figure.
    labels : list
        List of strings or floats to label plots with. If floats, we assume
        these are scaling values and add "Scaling = " to each of them. If
        strings, we label as is.
    cutout_fig : bool, optional
        Whether this is the nocutout (False) or cutout (True) version of the
        metamer comparison figure, which changes where we place the labels.
    natural_seed_fig : bool, optional
        Whether this is the natural-seed version of this fig or not, which
        changes how we place the labels.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    Returns
    -------
    fig : svgutils.compose.Figure
        Figure containing composed plots

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    metamer_fig = SVG(metamer_fig, 'inkscape')
    metamer_move = [0, 0]
    figure_height = metamer_fig.height * calc_scale('inkscape')
    # font_size is for panel labels and so too large for what we want here --
    # we want two-thirds or five-ninths the size (converting from 18pt to 12pt
    # or 10pt, respectively)
    font_size = float(text_params.pop('size').replace('pt', ''))
    labels = [f'Scaling = {val}' if isinstance(val, float) else val
              for val in labels]
    if not cutout_fig:
        font_size = _convert_to_pix(f'{font_size*2/3}pt')
        txt_move = [[120, 240], [380, 240], [120, 485], [395, 485]]
    else:
        font_size = _convert_to_pix(f'{font_size*5/9}pt')
        txt_move = [[100, 168], [380, 168], [100, 338], [380, 338],
                    [100, 508], [380, 508]]
    # this has 6 subplots, and we want a label above each of them
    if natural_seed_fig:
        # want to shift the metamer figure down a little bit so there's room
        # for labels on top row.
        figure_height += 20
        metamer_move[1] += 20
        # +20 to account for the extra 20px added above, -170 because we want
        # to move everything up a row.
        txt_move = [[mv[0], mv[1]+20-170] for mv in txt_move]
        # change the x value because they're longer than the scaling labels
        txt_move = [[mv[0]-offset, mv[1]] for mv, offset
                    in zip(txt_move, [10]+[53]*5)]
    return compose.Figure(
        figure_width, figure_height,
        metamer_fig.move(*metamer_move),
        *[compose.Text(val, *mv, size=font_size, **text_params)
          for val, mv in zip(labels, txt_move)],
    )


def performance_metamer_comparison_small(performance_fig, metamer_fig,
                                         scaling_vals, rectangle_colors,
                                         context='paper'):
    """Combine performance and small metamer performance comparison figs.

    Parameters
    ----------
    performance_fig, metamer_fig : str
        Paths to the performance and small metamer comparison figures.
    scaling_vals : list
        List of strings or floats, the scaling values to use for labeling the
        figure.
    rectangle_colors : list
        List of two colors for the rectangle edges
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    Returns
    -------
    fig : svgutils.compose.Figure
        Figure containing composed plots

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    # we do this to set the line thickness
    params, _ = style.plotting_style(context, 'matplotlib', 'full')
    plt.style.use(params)
    # little bit of extra space for the borders
    figure_width = _convert_to_pix(figure_width)+3
    metamer_fig = SVG(metamer_fig, 'inkscape')
    performance_fig = SVG(performance_fig, 'matplotlib')
    # font_size is for panel labels and so too large for the titles -- we want
    # five-ninths the size (converting from 18pt to 10pt)
    font_size = float(text_params.pop('size').replace('pt', ''))
    title_font_size = _convert_to_pix(f'{font_size*5/9}pt')
    txt_move = [(60, 333), (197+50, 333), (2*197+50, 333)]
    text = ['Target image'] + [f'Scaling = {val}' for val in scaling_vals]
    rects = [SVG(_create_tmp_rectangle(ec=c, fill=False,
                                       lw=2*params['lines.linewidth']),
                 'matplotlib') for c in rectangle_colors]
    return compose.Figure(
        figure_width, 2.1 * metamer_fig.height * calc_scale('inkscape'),
        performance_fig.move(0, 10),
        compose.Text('A', 0, 25, size=font_size, **text_params),
        metamer_fig.move(0, 340),
        compose.Text('B', 0, 330, size=font_size, **text_params),
        *[compose.Text(txt, *mv, size=title_font_size, **text_params)
          for txt, mv in zip(text, txt_move)],
        rects[0].move(1, 339),
        rects[1].move(1, 339+160),
    )


def combine_one_ax_figs(figs, context='paper'):
    """Combine one-ax figures showing performance separately per subject or image.

    Parameters
    ----------
    figs : list
        Lists of paths to the four figures to combine (one per model by trial type)
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    Returns
    -------
    fig : svgutils.compose.Figure
        Figure containing composed plots

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    figs = [SVG(fig, 'matplotlib') for fig in figs]
    # create rectangles to block out the labels we don't want to see
    ylabel_rect = SVG(_create_tmp_rectangle(figure_width+20, .1, fc='w', ec='none'),
                      'matplotlib')
    xlabel_rect = SVG(_create_tmp_rectangle(.2, figure_width, fc='w', ec='none'),
                      'matplotlib')
    # font_size is for panel labels and so too large for the titles
    font_size = float(text_params.pop('size').replace('pt', ''))
    return compose.Figure(
        # height needs room for xaxis label and titles as well
        figure_width, figure_width+35,
        # want to go through this backwards so the first figures are on top
        *[fig.move(9 + i // 2 * (figure_width/2 - 10), 15 + i % 2 * figure_width/2)
          for i, fig in enumerate(figs)][::-1],
        ylabel_rect.move(9+figure_width/2+5, 15),
        xlabel_rect.move(9, figure_width/2+15),
        compose.Text('Luminance model', 109, 15, size=font_size,
                     **text_params),
        compose.Text('Energy model', 109+figure_width/2+10+10, 15,
                     size=font_size, **text_params),
        compose.Text('Original vs Synth: white noise', 13, figure_width/2-20,
                     size=font_size, **text_params).rotate(270, 13, figure_width/2-20),
        compose.Text('Synth vs Synth: white noise', 13, figure_width-25,
                     size=font_size, **text_params).rotate(270, 13, figure_width-25),
    )


def performance_comparison(performance_fig, param_fig, context='paper'):
    """Combine performance figure with parameter one for comparison.

    Parameters
    ----------
    performance_fig, param_fig : str
        Paths to the performance and parameter figures, respectively.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    Returns
    -------
    fig : svgutils.compose.Figure
        Figure containing composed plots

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    # little bit of extra space for the borders
    return compose.Figure(
        figure_width+10, figure_width+30,
        SVG(performance_fig).move(0, 25),
        SVG(param_fig).move(0, figure_width/2),
        compose.Text('A', 0, 25, **text_params),
        compose.Text('B', 0, figure_width/2+25, **text_params),
    )
