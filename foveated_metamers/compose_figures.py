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


def model_schematic(schematic_fig, contour_fig_large, contour_fig_small,
                    save_path, context='paper'):
    """Add window contours to the model schematic figure.

    Parameters
    ----------
    schematic_fig, contour_fig_large, contour_fig_small : str
        Paths to the model schematic figure and two versions of the window
        contour figure, one which will be displayed relatively large and one
        that will be a bit less than half that size (and should also have a
        white background).
    save_path : str
        path to save the composed figure at
    context : {'paper', 'poster'}, optional
        plotting context that's being used for this figure (as in
        seaborn's set_context function). if poster, will scale things up. Note
        that, for this figure, only paper has really been checked

    """
    text_params, figure_width = style.plotting_style(context, 'svgutils', 'full')
    figure_width = _convert_to_pix(figure_width)
    schematic_fig = SVG(schematic_fig, 'inkscape')
    compose.Figure(
        figure_width, schematic_fig.height * calc_scale('inkscape'),
        schematic_fig,
        SVG(contour_fig_large).scale(.06).move(209, 51),
        SVG(contour_fig_small).scale(.0255).move(417+8, 302-15),
        SVG(contour_fig_small).scale(.0255).move(417, 302),
        SVG(contour_fig_small).scale(.0255).move(388+8, 362-15),
        SVG(contour_fig_small).scale(.0255).move(388, 362),
    ).save(save_path)


def metamer_comparison(metamer_fig, scaling_vals, save_path, context='paper'):
    """Add text labeling model metamer scaling values.

    Parameters
    ----------
    metamer_fig : str
        Path to the metamer comparison figure.
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
    # font_size is for panel labels and so too large for what we want here --
    # we want two thirds the size
    font_size = float(text_params.pop('size').replace('pt', ''))
    font_size = _convert_to_pix(f'{font_size*2/3}pt')
    compose.Figure(
        figure_width, metamer_fig.height * calc_scale('inkscape'),
        metamer_fig,
        compose.Text(f'Scaling = {scaling_vals[0]}', 120, 240, size=font_size,
                     **text_params),
        compose.Text(f'Scaling = {scaling_vals[1]}', 395, 240, size=font_size,
                     **text_params),
        compose.Text(f'Scaling = {scaling_vals[2]}', 120, 485, size=font_size,
                     **text_params),
        compose.Text(f'Scaling = {scaling_vals[3]}', 395, 485, size=font_size,
                     **text_params),
    ).save(save_path)
