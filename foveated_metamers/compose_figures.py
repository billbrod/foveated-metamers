#!/usr/bin/env python3
"""functions to put the final touches on figures for publication
"""
import re
from svgutils import compose
from . import style


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


def model_schematic(schematic_fig, contour_fig_large, contour_figs_small,
                    save_path, fig_size='full', context='paper'):
    """Add window contours to the model schematic figure.

    Parameters
    ----------
    schematic_fig, contour_fig_large, contour_figs_small : str
        Paths to the model schematic figure and two versions of the window
        contour figure, one which will be displayed relatively large and one
        that will be a bit less than half that size (and should also have a
        white background). contour_figs_small can be a single str or a list of 4
        strs, in which case we'll use each of those separately.
    save_path : str
        path to save the composed figure at
    fig_size : {'full', 'half'}, optional
        We have two different versions of this figure, one full-width, one half.
        This specifies which we're making.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

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
        scales = [.04, .0151]
        positions = [(162, 17), (247.5+5.5, 146.5-8.5), (247.5, 146.5),
                     (230+5.5, 183-8.5), (230, 183)]
    if isinstance(contour_figs_small, str):
        contour_figs_small = [contour_figs_small] * 4
    compose.Figure(
        figure_width, schematic_fig.height * calc_scale('inkscape'),
        schematic_fig,
        SVG(contour_fig_large).scale(scales[0]).move(*positions[0]),
        *[SVG(fig).scale(scales[1]).move(*positions[i+1])
          for i, fig in enumerate(contour_figs_small)],
    ).save(save_path)


def metamer_comparison(metamer_fig, scaling_vals, save_path, cutout_fig=False,
                       context='paper'):
    """Add text labeling model metamer scaling values.

    Parameters
    ----------
    metamer_fig : str
        Path to the metamer comparison figure.
    scaling_vals : list
        List of strings or floats, the scaling values to use for labeling the
        figure.
    save_path : str
        path to save the composed figure at
    cutout_fig : bool, optional
        Whether this is the nocutout (False) or cutout (True) version of the
        metamer comparison figure, which changes where we place the scaling
        labels.
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    metamer_fig = SVG(metamer_fig, 'inkscape')
    # font_size is for panel labels and so too large for what we want here --
    # we want two-thirds or five-ninths the size (converting from 18pt to 12pt
    # or 10pt, respectively)
    font_size = float(text_params.pop('size').replace('pt', ''))
    if not cutout_fig:
        font_size = _convert_to_pix(f'{font_size*2/3}pt')
        txt_move = [(120, 240), (380, 240), (120, 485), (395, 485)]
    else:
        font_size = _convert_to_pix(f'{font_size*5/9}pt')
        txt_move = [(100, 168), (380, 168), (100, 338), (380, 338),
                    (100, 508), (380, 508)]
    compose.Figure(
        figure_width, metamer_fig.height * calc_scale('inkscape'),
        metamer_fig,
        *[compose.Text(f'Scaling = {val}', *mv, size=font_size,
                       **text_params)
          for val, mv in zip(scaling_vals, txt_move)],
    ).save(save_path)


def performance_metamer_comparison_small(performance_fig, metamer_fig,
                                         scaling_vals, save_path,
                                         context='paper'):
    """Add text labeling model metamer scaling values.

    Parameters
    ----------
    performance_fig, metamer_fig : str
        Paths to the performance and small metamer comparison figures.
    scaling_vals : list
        List of strings or floats, the scaling values to use for labeling the
        figure.
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    metamer_fig = SVG(metamer_fig, 'inkscape')
    performance_fig = SVG(performance_fig, 'matplotlib')
    # font_size is for panel labels and so too large for the titles -- we want
    # five-ninths the size (converting from 18pt to 10pt)
    font_size = float(text_params.pop('size').replace('pt', ''))
    title_font_size = _convert_to_pix(f'{font_size*5/9}pt')
    txt_move = [(60, 333), (197+50, 333), (2*197+50, 333)]
    text = ['Target image'] + [f'Scaling = {val}' for val in scaling_vals]
    compose.Figure(
        figure_width, 2.1 * metamer_fig.height * calc_scale('inkscape'),
        performance_fig.move(0, 10),
        compose.Text('A', 0, 25, size=font_size, **text_params),
        metamer_fig.move(0, 340),
        compose.Text('B', 0, 330, size=font_size, **text_params),
        *[compose.Text(txt, *mv, size=title_font_size, **text_params)
          for txt, mv in zip(text, txt_move)],
    ).save(save_path)
