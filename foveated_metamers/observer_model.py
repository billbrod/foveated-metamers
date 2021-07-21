"""Observer model to explain our behavioral data.

This started with the PooledV1 model (found in extra_packages/plenoptic_part in
this repo and used to synthesize the model metamers used in this experiment).
This model was then changed to use the not-downsampled and tightframe version
of the steerable pyramid, in order to remove the necessity of the normalizing
across spatial scales. This entailed a bit of restructuring, so that this
should be functionally the same as PooledV1 in what it computes, interacting
with it is a bit different. This should hopefully be simpler, with unnecessary
options removed.

We then added on extra bells and whistles in order to
improve the predictive ability of the observer model.

"""
import torch
import warnings
import pyrtools as pt
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
import plenoptic as po
import sys
import os.path as op
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..',
                        'extra-packages', 'pooling-windows'))
from pooling import PoolingWindows
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..',
                        'extra-packages', 'plenoptic_part'))
from plenoptic_part.tools.display import clean_up_axes, update_stem, clean_stem_plot


class ObserverModel(nn.Module):
    r"""Observer model based on an pooled energy model of V1.

    This just models V1 as containing complex cells and a representation
    of the mean luminance. For the complex cells, we take the outputs of
    the complex steerable pyramid and takes the complex modulus of them
    (that is, squares, sums, and takes the square root across the real
    and imaginary parts; this is a phase-invariant measure of the local
    magnitude). The mean luminance representation is the same as that
    computed by the PooledRGC model.

    Note that we will calculate the minimum eccentricity at which the area of
    the windows at half-max exceeds one pixel (based on ``scaling``,
    ``img_res`` and ``max_eccentricity``). We will not throw an Exception if
    this value is below ``min_eccentricity``, however. We instead print a
    warning to alert the user and use this value as ``min_eccentricity`` when
    creating the plots. In order to see what this value was, see
    ``self.calculated_min_eccentricity_degrees``

    We can optionally cache the windows tensor we create, if ``cache_dir`` is
    not None. In that case, we'll also check to see if appropriate cached
    windows exist before creating them and load them if they do. The path we'll
    use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_w-1_gaussian.pt``. We'll cache each scale separately,
    changing the img_res (and potentially min_eccentricity) values in that save
    path appropriately.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows.
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
    order : int, optional
        The Gaussian derivative order used for the steerable
        filters. Default value is 3.  Note that to achieve steerability
        the minimum number of orientation is ``order`` + 1, and is used
        here (that's currently all we support, though could extend
        fairly simply)
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
    order : int, optional
        The Gaussian derivative order used for the steerable
        filters. Default value is 3.  Note that to achieve steerability
        the minimum number of orientation is ``order`` + 1, and is used
        here (that's currently all we support, though could extend
        fairly simply)
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    state_dict_reduced : dict
        A dictionary containing those attributes necessary to initialize
        the model, plus a 'model_name' field which the ``load_reduced``
        method uses to determine which model constructor to call. This
        is used for saving/loading the models, since we don't want to
        keep the (very large) representation and intermediate steps
        around. To save, use ``self.save_reduced(filename)``, and then
        load from that same file using the class method
        ``po.simul.PooledVentralStream.load_reduced(filename)``
    window_width_degrees : dict
        Dictionary containing the widths of the windows in
        degrees. There are four keys: 'radial_top', 'radial_full',
        'angular_top', and 'angular_full', corresponding to a 2x2 for
        the widths in the radial and angular directions by the 'top' and
        'full' widths (top is the width of the flat-top region of each
        window, where the window's value is 1; full is the width of the
        entire window). Each value is a list containing the widths for
        the windows in different eccentricity bands. To visualize these,
        see the ``plot_window_sizes`` method.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        ``plot_window_sizes`` method.
    n_polar_windows : int
        The number of windows we have in the polar angle dimension
        (within each eccentricity band)
    n_eccentricity_bands : int
        The number of eccentricity bands in our model
    calculated_min_eccentricity_degrees : list
        List of floats (one for each scale) that contain
        ``calc_min_eccentricity()[0]``, that is, the minimum
        eccentricity (in degrees) where the area of the window at
        half-max exceeds one pixel (based on the scaling, size of the
        image in pixels and in degrees).
    calculated_min_eccentricity_pixels : list
        List of floats (one for each scale) that contain
        ``calc_min_eccentricity()[1]``, that is, the minimum
        eccentricity (in pixels) where the area of the window at
        half-max exceeds one pixel (based on the scaling, size of the
        image in pixels and in degrees).
    central_eccentricity_degrees : np.ndarray
        A 1d array with shape ``(self.n_eccentricity_bands,)``, each
        value gives the eccentricity of the center of each eccentricity
        band of windows (in degrees).
    central_eccentricity_pixels : list
        List of 1d arrays (one for each scale), each with shape
        ``(self.n_eccentricity_bands,)``, each value gives the
        eccentricity of the center of each eccentricity band of windows
        (in degrees).
    window_approx_area_degrees : dict
        Dictionary containing the approximate areas of the windows, in
        degrees. There are three keys: 'top', 'half', and 'full',
        corresponding to which width we used to calculate the area (top
        is the width of the flat-top region of each window, where the
        window's value is 1; full is the width of the entire window;
        half is the width at half-max). To get this approximate area, we
        multiply the radial and angular widths against each other and
        then by pi/4 to get the area of the regular ellipse that has
        those widths (our windows are elongated, so this is probably an
        under-estimate). To visualize these, see the
        ``plot_window_areas`` method
    window_approx_area_pixels : list
        List of dictionaries containing the approximate areasof the
        windows in pixels; each entry in the list corresponds to the
        areas for a different scale, as in ``windows``. See above for
        explanation of the dictionaries. To visualize these, see the
        ``plot_window_areas`` method.
    deg_to_pix : list
        List of floats containing the degree-to-pixel conversion factor
        at each scale
    cache_dir : str or None
        If str, this is the directory where we cached / looked for
        cached windows tensors
    cache_paths : list
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    scales : list
        List of the scales in the model, from fine to coarse. Used for
        synthesizing in coarse-to-fine order

    """

    def __init__(self, scaling, img_res, num_scales=4, order=3,
                 min_eccentricity=.5, max_eccentricity=15, cache_dir=None):
        super().__init__()
        self.PoolingWindows = PoolingWindows(scaling, img_res,
                                             min_eccentricity,
                                             # we always want 1 scale for
                                             # pooling windows, because we use
                                             # the not-downsampled pyramid
                                             max_eccentricity, num_scales=1,
                                             cache_dir=cache_dir,
                                             window_type='gaussian',
                                             std_dev=1)
        for attr in ['n_polar_windows', 'n_eccentricity_bands', 'scaling',
                     'state_dict_reduced', 'window_width_pixels',
                     'window_width_degrees', 'min_eccentricity',
                     'max_eccentricity', 'cache_dir', 'deg_to_pix',
                     'window_approx_area_degrees', 'window_approx_area_pixels',
                     'cache_paths', 'calculated_min_eccentricity_degrees',
                     'calculated_min_eccentricity_pixels',
                     'central_eccentricity_pixels',
                     'central_eccentricity_degrees', 'img_res']:
            setattr(self, attr, getattr(self.PoolingWindows, attr))
        # remove the keys from state_dict_reduced that are not used to
        # initialize ObserverModel
        for k in ['transition_region_width', 'window_type', 'std_dev']:
            self.state_dict_reduced.pop(k)
        self._spatial_masks = {}
        self.state_dict_reduced.update({'order': order,
                                        'model_name': 'ObserverModel',
                                        'num_scales': num_scales})
        self.num_scales = num_scales
        self.order = order
        self.complex_steerable_pyramid = po.simul.Steerable_Pyramid_Freq(
            img_res, self.num_scales, self.order, is_complex=True,
            downsample=False, tight_frame=True)
        self.scales = ['mean_luminance'] + list(range(num_scales))[::-1]

    def forward(self, image, scales=[]):
        r"""Generate the V1 representation of an image.

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's ``scales`` attribute (ints
            up to self.num_scales-1, the str 'mean_luminance'). Can contain a
            single value or multiple values. If it's an int or float, we
            include all orientations from that scale.

        Returns
        -------
        representation : torch.Tensor
            A 3d tensor containing the averages of the
            'complex cell responses', that is, the squared and summed
            outputs of the complex steerable pyramid.

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        if image.shape[1] != 1:
            raise Exception("Haven't thought about how to handle images with"
                            " more than 1 channel!")
        if not scales:
            scales = self.scales
        # initialize these with empty tensors so that torch.cat runs
        # successfully even if they don't get updated.
        mean_complex_cell_responses = torch.tensor([])
        mean_luminance = torch.tensor([])
        if any([i in self.complex_steerable_pyramid.scales for i in scales]):
            # because self.scales never includes residual_highpass and
            # residual_lowpass, we never have the residuals in pyr_coeffs.
            pyr_coeffs = self.complex_steerable_pyramid(image, scales)
            complex_cell_responses, pyr_info = self.complex_steerable_pyramid.convert_pyr_to_tensor(pyr_coeffs)
            self._pyr_info = pyr_info
            # to get the energy, we just square and take the absolute value
            # (since this is a complex tensor, this is equivalent to summing
            # across the real and imaginary components). the if statement
            # avoids the residuals
            complex_cell_responses = complex_cell_responses.pow(2).abs()
            mean_complex_cell_responses = self.PoolingWindows(complex_cell_responses)
        if 'mean_luminance' in scales:
            mean_luminance = self.PoolingWindows(image)
        return torch.cat([mean_complex_cell_responses, mean_luminance], dim=1)

    def _gen_spatial_masks(self, n_angles=4):
        r"""Generate spatial masks.

        Create and return masks that allow us to specifically select values
        from model's representation that correspond to different regions of the
        image. See ``summarize_representation()`` for an example of how to use
        them

        Parameters
        ----------
        n_angles : int, optional
            The number of angular regions to subdivide the image
            into. By default, splits it into quadrants

        Returns
        -------
        masks : dict
            dictionary with a key for each (scale, angle_i) that
            contains a tensor, same shape as the representations at that
            scale, which is a boolean mask selecting the values that
            correspond to that angular region

        """
        masks = {}
        ecc = torch.ones_like(self.PoolingWindows.ecc_windows[0], dtype=int)
        angles = torch.zeros_like(self.PoolingWindows.angle_windows[0], dtype=int)
        for j in range(n_angles):
            angles[j*angles.shape[0]//4:(j+1)*angles.shape[0]//4] = j
        windows = torch.einsum('ahw,ehw->ea', angles, ecc)
        for j, val in enumerate(sorted(windows.unique())):
            masks[f'region_{j}'] = (windows == val).flatten()
        return masks

    def to(self, *args, do_windows=True, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            do_windows (:class: `bool`): whether to also call
                `self.PoolingWindows.to()` with the specified args and kwargs.

        """
        self.complex_steerable_pyramid.to(*args, **kwargs)
        if do_windows:
            self.PoolingWindows.to(*args, **kwargs)
        for k, v in self._spatial_masks.items():
            self._spatial_masks[k] = v.to(*args, **kwargs)
        nn.Module.to(self, *args, **kwargs)

    def plot_windows(self, ax=None, contour_levels=None, colors='r',
                     subset=True, **kwargs):
        r"""Plot the pooling windows on an image.

        This is just a simple little helper to plot the pooling windows
        on an existing axis. The use case is overlaying this on top of
        the image we're pooling (as returned by ``pyrtools.imshow``),
        and so we require an axis to be passed

        Any additional kwargs get passed to ``ax.contour``

        Parameters
        ----------
        ax : matplotlib.pyplot.axis or None, optional
            The axis to plot the windows on. If None, will create a new
            figure with 1 axis
        contour_levels : None, array-like, or int, optional
            The ``levels`` argument to pass to ``ax.contour``. From that
            documentation: "Determines the number and positions of the
            contour lines / regions. If an int ``n``, use ``n`` data
            intervals; i.e. draw ``n+1`` contour lines. The level
            heights are automatically chosen. If array-like, draw
            contour lines at the specified levels. The values must be in
            increasing order". If None, will plot the contour that gives
            the first intersection (.5 for raised-cosine windows,
            self.window_max_amplitude * np.exp(-.25/2) (half a standard
            deviation away from max) for gaussian windows), as this is
            the easiest to see.
        colors : color string or sequence of colors, optional
            The ``colors`` argument to pass to ``ax.contour``. If a
            single character, all will have the same color; if a
            sequence, will cycle through the colors in ascending order
            (repeating if necessary)
        subset : bool, optional
            If True, will only plot four of the angle window
            slices. This is to save time and memory. If False, will plot
            all of them

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis with the windows

        """
        return self.PoolingWindows.plot_windows(ax, contour_levels, colors, subset, **kwargs)

    def summarize_window_sizes(self):
        r"""Summarize window sizes.

        This function returns a dictionary summarizing the window sizes
        at the minimum and maximum eccentricity. Let ``min_window`` be
        the window whose center is closest to ``self.min_eccentricity``
        and ``max_window`` the one whose center is closest to
        ``self.max_eccentricity``. We find its center, FWHM (in the
        radial direction), and approximate area (at half-max) in
        degrees. We do the same in pixels, for each scale.

        Returns
        -------
        sizes : dict
            dictionary with the keys described above, summarizing window
            sizes. all values are scalar floats

        """
        return self.PoolingWindows.summarize_window_sizes()

    def plot_window_widths(self, units='degrees', figsize=(5, 5), jitter=.25):
        r"""Plot the widths of the windows, in degrees or pixels.

        We plot the width of the window in both angular and radial direction,
        as well as showing the 'half' and 'full' widths (full is the width of
        the entire window; half is the width at the half-max value, which is
        what corresponds to the scaling value)

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the width and the window location will be presented in the
            same unit).
        figsize : tuple, optional
            The size of the figure to create
        jitter : float or None, optional
            Whether to add a little bit of jitter to the x-axis to
            separate the radial and angular widths. There are only two
            values we separate, so we don't add actual jitter, just move
            one up by the value specified by jitter, the other down by
            that much (we use the same value at each eccentricity)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        fig = self.PoolingWindows.plot_window_widths(units, 0, figsize, jitter)
        # since ObserverModel always uses Gaussian windows, the top width is
        # always 0 and so we remove it.
        for c in fig.axes[0].containers:
            if 'top' in c.get_label():
                c.remove()
        # recreate the legend
        fig.axes[0].legend(loc='upper left')
        return fig

    def plot_window_areas(self, units='degrees', figsize=(5, 5)):
        r"""Plot the approximate areas of the windows, in degrees or pixels.

        We plot the approximate area of the window, calculated using 'half' and
        'full' widths (full is the width of the entire window; half is the
        width at the half-max value, which is what corresponds to the scaling
        value). To get the approximate area, we multiply the radial width
        against the corresponding angular width, then divide by pi / 4.

        The half area shown here is what we use to compare against a
        threshold value in the ``calc_min_eccentricity()`` in order to
        determine what the minimum eccentricity where the windows
        contain more than 1 pixel.

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the area and the window location will be presented in the
            same unit).
        figsize : tuple, optional
            The size of the figure to create

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        fig = self.PoolingWindows.plot_window_areas(units, 0, figsize)
        # since ObserverModel always uses Gaussian windows, the top area is
        # always 0 and so we remove it.
        for c in fig.axes[0].containers:
            if c.get_label() == 'top':
                c.remove()
        # recreate the legend
        fig.axes[0].legend(loc='upper left')
        return fig

    def save_reduced(self, file_path):
        r"""Save the relevant parameters to make saving/loading more efficient.

        This saves self.state_dict_reduced, which contains the
        attributes necessary to initialize the model plus a 'model_name'
        key, which the ``load_reduced`` method uses to determine which
        model constructor to call

        Parameters
        ----------
        file_path : str
            The path to save the model object to

        """
        torch.save(self.state_dict_reduced, file_path)

    @classmethod
    def load_reduced(cls, file_path):
        r"""Load from the dictionary saved by ``save_reduced``.

        Parameters
        ----------
        file_path : str
            The path to load the model object from
        """
        state_dict_reduced = torch.load(file_path)
        return cls.from_state_dict_reduced(state_dict_reduced)

    @classmethod
    def from_state_dict_reduced(cls, state_dict_reduced):
        r"""Initialize model from ``state_dict_reduced``.

        Parameters
        ----------
        state_dict_reduced : dict
            The reduced state dict to load
        """
        state_dict_reduced = state_dict_reduced.copy()
        model_name = state_dict_reduced.pop('model_name')
        # want to remove class if it's here
        state_dict_reduced.pop('class', None)
        if model_name == 'ObserverModel':
            return ObserverModel(**state_dict_reduced)
        else:
            raise Exception("Don't know how to handle model_name %s!" % model_name)

    def summarize_representation(self, data, summary_func='mse', by_angle=False):
        r"""Summarize representation by key and (optionally) quadrant.

        This takes data and summarizes it within each key of the
        dictionary. The intended use case is to get the mean-squared
        error (by passing ``data=metamer.representation_error()``)
        within each orientation and scale. With ``by_angle=True``, also
        breaks it down by quadrant.

        Parameters
        ----------
        data : torch.Tensor
            The data to convert. Should look like the output, with the exact
            same structure
        summary_func : {'mse', 'l2', callable}, optional
            the function to use to for summarizing the
            representation. If 'mse', we'll square and average; if 'l2',
            we'll use the L2-norm; else, should be a callable that can
            take a tensor and returns a scalar
        by_angle : bool, optional
            whether to further breakdown representation by angle. If
            False, will just have a single value per key in
            representation. If True, keys will be (k, 'region_{i}'),
            where goes from 0 to 3 and represents quadrants, starting
            from bottom right. Note that, if this is the first time
            doing this, we'll need to create spatial masks to do this,
            which will massively increase the amount of memory used
            (because we create copies of the PoolingWindows at the
            coarsest scale to do so).

        Returns
        -------
        summarized : dict
            dictionary containing keys from representation (or
            representation and region, see above) with values giving the
            corresponding summarized representation

        """
        if not self._spatial_masks and by_angle:
            self._spatial_masks = self._gen_spatial_masks()
        if summary_func == 'mse':
            summary_func = lambda x: torch.pow(x, 2).mean()
        elif summary_func == 'l2':
            summary_func = lambda x: torch.norm(x, 2)
        if by_angle:
            keys = [f'region_{i}' for i in range(4)]
        else:
            keys = ['whole_image']
        # This is slightly complicated because summary_func takes a tensor and
        # returns a scalar -- if we could also pass a dim arg, this would be
        # simpler, but we don't want to restrict ourselves that way.
        summarized = {k: torch.empty(data.shape[:-1]) for k in keys}
        for i, batch_d in enumerate(data):
            for j, channel_d in enumerate(batch_d):
                if by_angle:
                    for k in range(4):
                        mask = self._spatial_masks[f'region_{k}']
                        summarized[f'region_{k}'][i, j] = summary_func(channel_d[mask]).item()
                else:
                    summarized['whole_image'][i, j] = summary_func(channel_d).item()
        return summarized

    @classmethod
    def _get_title(cls, title_list, idx, default_title):
        r"""Helper function for dealing with the default way we handle title.

        We have a couple possible ways of handling title in these
        plotting functions, so this helper function consolidates
        that.

        When picking a title, we know we'll either have a list of titles
        or a single None.

        - If it's None, we want to just use the default title.

        - If it's a list, pick the appropriate element of the list

          - If that includes '|' (the pipe character), then append the
            default title on the other side of the pipe

        Then return

        Parameters
        ----------
        title_list : list or None
            A list of strs or Non
        idx : int
            An index into title_list. Can be positive or negative.
        default_title : str
            The title to use if title_list is None or to include if
            title_list[idx] includes a pipe

        Returns
        -------
        title : str
            The title to use
        """
        try:
            title = title_list[idx]
            if '|' in title:
                if title.index('|') == 0:
                    # then assume it's at the beginning
                    title = default_title + ' ' + title
                else:
                    # then assume it's at the end
                    title += ' ' + default_title
        except TypeError:
            # in this case, title is None
            title = default_title
        return title

    def update_plot(self, axes, data, batch_idx=0):
        r"""Update the information in our representation plot.

        This is used for creating an animation of the representation
        over time. In order to create the animation, we need to know how
        to update the matplotlib Artists, and this provides a simple way
        of doing that. It relies on the fact that we've used
        ``plot_representation`` to create the plots we want to update
        and so know that they're stem plots.

        We take the axes containing the representation information (note
        that this is probably a subset of the total number of axes in
        the figure, if we're showing other information, as done by
        ``Metamer.animate``), grab the representation from plotting and,
        since these are both lists, iterate through them, updating as we
        go.

        We can optionally accept a data argument, in which case it should look
        just like the representation or output of this model.

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Parameters
        ----------
        axes : list
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        data : torch.Tensor
            The data to show on the plot. Should look like model's
            representation with the exact same structure (e.g., as returned by
            ``metamer.representation_error()`` or another instance of this
            class).
        batch_idx : int, optional
            Which index to take from the batch dimension

        Returns
        -------
        stem_artists : list
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data = data[batch_idx]
        for ax, d in zip(axes, data):
            sc = update_stem(ax.containers[0], d)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists

    def _plot_helper(self, n_rows, n_cols, figsize=(25, 15),
                     ax=None, title=None, batch_idx=0):
        r"""Helper function for plotting that takes care of a lot of the standard stuff.

        Parameters
        ----------
        n_rows : int
            The number of rows in the (sub-)figure we're creating
        n_cols : int
            The number oc columns in the (sub-)figure we're creating
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If list or None, this does nothing to it. If
            a str, we turn it into a list with ``(n_rows*n_cols)``
            entries, all identical and the same as the user-pecified
            title
        batch_idx : int, optional
            Which index to take from the batch dimension

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing our subplots
        gs : matplotlib.gridspec.GridSpec
            The GridSpec object to use for creating subplots. You should
            use it with ``fig`` to add subplots by indexing into it,
            like so: ``fig.add_subplot(gs[0, 1])``
        title : list or None
            If title was None or a list, we did nothing to it. If it was
            a str, we made sure its length is (n_rows * n_cols)

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
            fig = ax.figure
        if isinstance(title, str):
            # then this is a single str, so we'll make it the same on
            # every subplot
            title = (n_rows * n_cols) * [title]
        return fig, gs, title

    def plot_representation(self, data, figsize=(25, 15), ylim=None, ax=None,
                            title=None, batch_idx=0):
        r"""Plot the representation of the V1 model.

        Since our PooledV1 model has more statistics than the
        PooledRGC model, this is a much more complicated
        plot. We end up creating a grid, showing each band and scale
        separately, and then a separate plot, off to the side, for the
        mean pixel intensity.

        Despite this complication, we can still take an ``ax`` argument
        to plot on some portion of a figure. We make use of matplotlib's
        powerful ``GridSpec`` to arrange things to our liking.

        Each plot has a small break in the data to show where we've
        moved out to the next eccentricity ring.

        Note that this looks better when it's wider than it is tall
        (like the default figsize suggests)

        Parameters
        ----------
        data : torch.Tensor
            The data to plot. Should look like model output, with same
            structure.
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ylim : tuple, False, or None
            If a tuple, the y-limits to use for this plot. If None, we use
            the default, slightly adjusted so that the minimum is 0. If
            False, we do nothing.
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If a list of strings, must be the same length
            as ``data``, and each value will be put above
            the subplots. If a str, the same title will be put above
            each subplot. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If it includes a '|' (pipe), then we'll
            append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        n_cols = self.num_scales + 1
        n_rows = max(self.order + 1, 1)
        col_multiplier = 2
        col_offset = 1
        fig, gs, title_list = self._plot_helper(2*n_rows, 2*n_cols, figsize,
                                                ax, title, batch_idx)
        axes = []
        for i, d in enumerate(po.to_numpy(data[batch_idx])):
            scale = i // (self.order+1)
            band = i % (self.order+1)
            if scale < self.num_scales:
                t = self._get_title(title_list, i, f"scale {scale}, band {band}")
                ax = fig.add_subplot(gs[int(2*band):int(2*(band+1)), int(col_multiplier*scale):
                                        int(col_multiplier*(scale+col_offset))])
                ax = clean_stem_plot(d, ax, t, ylim)
                axes.append(ax)
            else:
                t = self._get_title(title_list, i, 'mean luminance')
                # middle row and last column
                ax = fig.add_subplot(gs[n_rows-1:n_rows+1, 2*(n_cols-1):])
                ax = clean_stem_plot(d, ax, t, ylim)
                axes.append(ax)
        return fig, axes

    def plot_representation_image(self, data, figsize=(22, 11), ax=None,
                                  title=None, batch_idx=0, vrange='auto1',
                                  zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows.

        Our representation is composed of pooled energy at several different
        scales and orientations, plus the pooled mean pixel intensity. In order
        to visualize these as images, we take each statistic, multiply it by
        the pooling windows, then sum across windows, as in
        ``PooledRGC.plot_representation_image``. We also sum across
        orientations at the same scale, so that we end up with an image for
        each scale, plus one for mean pixel intensity.

        Parameters
        ----------
        data : torch.Tensor
            The data to plot. Else, should look like model's output, with the
            exact same structure (e.g., as returned by
            ``metamer.representation_error()`` or another instance of this
            class).
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If a list of strings, must be the same length as
            ``data``, and each value will be put above the subplots. If a str,
            the same title will be put above each subplot. If None, we use the
            default choice, which specifies the scale and orientation of each
            plot (and the mean intensity). If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension
        vrange : str or tuple, optional
            The vrange value to pass to pyrtools.imshow
        zoom : float or int, optional
            If a float, must be castable to an int or an inverse power
            of 2. The amount to zoom the images in/out by.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        n_cols = max(self.num_scales, 1)
        n_rows = 2
        fig, gs, title_list = self._plot_helper(n_rows, n_cols, figsize, ax,
                                                title, batch_idx)
        titles = []
        axes = []
        imgs = []
        # project expects a 3d tensor
        data = self.PoolingWindows.project(data)[batch_idx]
        for i in self.scales:
            if isinstance(i, str):
                continue
            titles.append(self._get_title(title_list, i, "scale %s" % i))
            img = torch.zeros_like(data[0])
            for j in range(self.order+1):
                img += data[i*(self.order+1)+j]
            ax = fig.add_subplot(gs[0, int(i)])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            imgs.append(img)
            axes.append(ax)
        ax = fig.add_subplot(gs[1, 0])
        ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'],
                           ['x', 'y'])
        axes.append(ax)
        titles.append(self._get_title(title_list, len(self.scales)-1, 'mean_luminance'))
        imgs.append(data[-1])
        vrange, cmap = pt.tools.display.colormap_range(imgs, vrange)
        for ax, img, t, vr in zip(axes, imgs, titles, vrange):
            po.imshow(img.unsqueeze(0).unsqueeze(0), ax=ax, vrange=vr,
                      cmap=cmap, title=t, zoom=zoom)
        return fig, axes
