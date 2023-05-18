"""functions for ventral stream perceptual models, as seen in Freeman and Simoncelli, 2011

"""
import torch
import warnings
import pyrtools as pt
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
import plenoptic as po
from ..tools.display import clean_up_axes, update_stem, clean_stem_plot
from ..tools.optim import zscore_stats
import sys
import os.path as op
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', '..', 'pooling-windows'))
from pooling import PoolingWindows


class PooledVentralStream(nn.Module):
    r"""Generic class that sets up scaling windows

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    This just generates the pooling windows necessary for these models,
    given a small number of parameters. One tricky thing we do is
    generate a set of scaling windows for each scale (appropriately)
    sized. For example, the V1 model will have 4 scales, so for a 256 x
    256 image, the coefficients will have shape (256, 256), (128, 128),
    (64, 64), and (32, 32). Therefore, we need windows of the same size
    (could also up-sample the coefficient tensors, but since that would
    need to happen each iteration of the metamer synthesis,
    pre-generating appropriately sized windows is more efficient).

    We will calculate the minimum eccentricity at which the area of the
    windows at half-max exceeds one pixel at each scale. For scales
    beyond the first however, we will not throw an Exception if this
    value is below ``min_eccentricity``. We instead print a warning to
    alert the user and use this value as ``min_eccentricity`` when
    creating the plots. In order to see what this value was, see
    ``self.calculated_min_eccentricity_degrees``

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_t-{transition_region_width}.pt``. We'll cache
    each scale separately, changing the img_res (and potentially
    min_eccentricity) values in that save path appropriately.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows.
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    num_scales : int, optional
        The number of scales to generate masks for. For the RGC model,
        this should be 1, otherwise should match the number of scales in
        the steerable pyramid.
    transition_region_width : float, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    moments : list, optional
        Subset of 2, 3, 4. Which moments (other than the mean) to include in
        this model's representation.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    PoolingWindows : plenoptic.simulate.PoolingWindows
        A pooling windows object which contains the windows we use to
        pool our model's summary statistics across the image.
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
        degrees. There are six keys, corresponding to a 2x2 for the
        widths in the radial and angular directions by the 'top',
        'half', and 'full' widths (top is the width of the flat-top
        region of each window, where the window's value is 1; full is
        the width of the entire window; half is the width at
        half-max). Each value is a list containing the widths for the
        windows in different eccentricity bands. To visualize these, see
        the ``plot_window_widths`` method.
    window_width_pixels : list
        List of dictionaries containing the widths of the windows in
        pixels; each entry in the list corresponds to the widths for a
        different scale, as in ``windows``. See above for explanation of
        the dictionaries. To visualize these, see the
        ``plot_window_widths`` method.
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
    cached_paths : list
        List of strings, one per scale, taht we either saved or loaded
        the cached windows tensors from
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.simul.non_linearities.generate_norm_stats``. If this is an
        empty dict, we don't normalize the model.
    to_normalize : list
        List of attributes that we want to normalize by whitening

    """
    def __init__(self, scaling, img_res, min_eccentricity=.5,
                 max_eccentricity=15, num_scales=1, transition_region_width=.5,
                 cache_dir=None, window_type='cosine', std_dev=None,
                 normalize_dict={}, moments=[]):
        super().__init__()
        self.PoolingWindows = PoolingWindows(scaling, img_res, min_eccentricity, max_eccentricity,
                                             num_scales, cache_dir, window_type,
                                             transition_region_width, std_dev)
        for attr in ['n_polar_windows', 'n_eccentricity_bands', 'scaling', 'state_dict_reduced',
                     'transition_region_width', 'window_width_pixels', 'window_width_degrees',
                     'min_eccentricity', 'max_eccentricity', 'cache_dir', 'deg_to_pix',
                     'window_approx_area_degrees', 'window_approx_area_pixels', 'cache_paths',
                     'calculated_min_eccentricity_degrees', 'calculated_min_eccentricity_pixels',
                     'central_eccentricity_pixels', 'central_eccentricity_degrees', 'img_res',
                     'window_type', 'std_dev']:
            setattr(self, attr, getattr(self.PoolingWindows, attr))
        self.normalize_dict = normalize_dict
        self.to_normalize = []
        self.state_dict_reduced['normalize_dict'] = normalize_dict
        self.state_dict_reduced['moments'] = moments
        self.num_scales = 1
        self._spatial_masks = {}
        test_moments = [i for i in moments]
        # these are the only allowed moments
        for i in [2, 3, 4]:
            if i in test_moments:
                test_moments.remove(i)
        if len(test_moments) > 0:
            raise Exception(f"Only acceptable values for moments are [2, 3, 4], but got other values {test_moments}!``")
        self._moments = moments

    def _gen_spatial_masks(self, n_angles=4):
        r"""Generate spatial masks

        Create and return masks that allow us to specifically select
        values from ``self.representation`` that correspond to different
        regions of the image. See ``summarize_representation()`` for an
        example of how to use them

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
        for i in range(self.num_scales):
            ecc = torch.ones_like(self.PoolingWindows.ecc_windows[i], dtype=int)
            angles = torch.zeros_like(self.PoolingWindows.angle_windows[i], dtype=int)
            for j in range(n_angles):
                angles[j*angles.shape[0]//4:(j+1)*angles.shape[0]//4] = j
            windows = torch.einsum('ahw,ehw->ea', angles, ecc)
            for j, val in enumerate(sorted(windows.unique())):
                masks[(i, f'region_{j}')] = (windows == val).flatten()
        return masks

    def _calculate_moments(self, image, second=True, third=True,
                           fourth=True, pooled_mean=None):
        """Calculate weighted moments of image.

        Note that we're computing nth root of the first 4 moments, not the
        central or standardized moments (which are more common and go by the
        names variance, skew, and kurtosis; these involve subtracting off the
        mean and dividing by the variance raised to some power, respectively).
        We use the plain-old moments to avoid issues with negative values
        (which we can't take the root of) or blowing up when variance is small.

        We take the root so these values scale appropriately with the values of
        image.

        Note that, when matching the moments (e.g., in metamer synthesis),
        matching all of these plain-old moments is equivalent to matching the
        standardized versions (though this isn't the case if you wanted to use
        this model for e.g., computing perceptual distance). This isn't the
        case if you don't match all moments up to N: if you only match the
        fourth moment, that is *not* equivalent to matching the kurtosis.

        Parameters
        ----------
        image : torch.Tensor
            The 4d tensor to compute the weighted moments of.
        second, third, fourth : bool, optional
            Whether to compute the second, third, and fourth moments.
        pooled_mean : None or torch.Tensor, optional
            If a tensor, the weighted average of image (so we can avoid
            re-calculating it, and speed this up). If None, we compute it
            ourselves.

        Returns
        -------
        moments : dict
            Dict of tensors, containing the calculated moments.

        """
        moments = {}
        # clamp with min=0 to avoid NaNs. see
        # https://discuss.pytorch.org/t/incorrect-pow-function/62735/3 for why
        # this applies even to the third-root
        if second:
            moments['second'] = self.PoolingWindows(image.pow(2)).clamp(min=0).pow(1/2)
        if third:
            moments['third'] = self.PoolingWindows(image.pow(3)).clamp(min=0).pow(1/3)
        if fourth:
            moments['fourth'] = self.PoolingWindows(image.pow(4)).clamp(min=0).pow(1/4)
        return moments

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

        Returns:
            Module: self
        """
        if do_windows:
            self.PoolingWindows.to(*args, **kwargs)
        for k, v in self._spatial_masks.items():
            self._spatial_masks[k] = v.to(*args, **kwargs)
        for k, v in self.normalize_dict.items():
            if isinstance(v, dict):
                for l, w in v.items():
                    self.normalize_dict[k][l] = w.to(*args, **kwargs)
            else:
                self.normalize_dict[k] = v.to(*args, **kwargs)
        nn.Module.to(self, *args, **kwargs)
        return self

    def plot_windows(self, ax=None, contour_levels=None, colors='r', subset=True, **kwargs):
        r"""plot the pooling windows on an image.

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
        r"""Summarize window sizes

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

    def plot_window_widths(self, units='degrees', scale_num=0, figsize=(5, 5), jitter=.25):
        r"""plot the widths of the windows, in degrees or pixels

        We plot the width of the window in both angular and radial
        direction, as well as showing the 'top', 'half', and 'full'
        widths (top is the width of the flat-top region of each window,
        where the window's value is 1; full is the width of the entire
        window; half is the width at the half-max value, which is what
        corresponds to the scaling value)

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes)

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the width and the window location will be presented in the
            same unit).
        scale_num : int, optional
            Which scale window we should plot
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
        return self.PoolingWindows.plot_window_widths(units, scale_num, figsize, jitter)

    def plot_window_areas(self, units='degrees', scale_num=0, figsize=(5, 5)):
        r"""plot the approximate areas of the windows, in degrees or pixels

        We plot the approximate area of the window, calculated using
        'top', 'half', and 'full' widths (top is the width of the
        flat-top region of each window, where the window's value is 1;
        full is the width of the entire window; half is the width at the
        half-max value, which is what corresponds to the scaling
        value). To get the approximate area, we multiply the radial
        width against the corresponding angular width, then divide by pi
        / 4.

        The half area shown here is what we use to compare against a
        threshold value in the ``calc_min_eccentricity()`` in order to
        determine what the minimum eccentricity where the windows
        contain more than 1 pixel.

        We plot this as a stem plot against eccentricity, showing the
        windows at their central eccentricity

        If the unit is 'pixels', then we also need to know which
        ``scale_num`` to plot (the windows are created at different
        scales, and so come in different pixel sizes)

        Parameters
        ----------
        units : {'degrees', 'pixels'}, optional
            Whether to show the information in degrees or pixels (both
            the area and the window location will be presented in the
            same unit).
        scale_num : int, optional
            Which scale window we should plot
        figsize : tuple, optional
            The size of the figure to create

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot

        """
        return self.PoolingWindows.plot_window_areas(units, scale_num, figsize)

    def save_reduced(self, file_path):
        r"""save the relevant parameters to make saving/loading more efficient

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
        r"""load from the dictionary saved by ``save_reduced``

        Parameters
        ----------
        file_path : str
            The path to load the model object from
        """
        state_dict_reduced = torch.load(file_path)
        return cls.from_state_dict_reduced(state_dict_reduced)

    @classmethod
    def from_state_dict_reduced(cls, state_dict_reduced):
        r"""initialize model from ``state_dict_reduced``

        Parameters
        ----------
        state_dict_reduced : dict
            The reduced state dict to load
        """
        state_dict_reduced = state_dict_reduced.copy()
        model_name = state_dict_reduced.pop('model_name')
        # want to remove class if it's here
        state_dict_reduced.pop('class', None)
        if model_name == 'RGC':
            # RGC doesn't accept moments as argument
            state_dict_reduced.pop('moments', None)
            return PooledRGC(**state_dict_reduced)
        elif model_name == 'V1':
            return PooledV1(**state_dict_reduced)
        elif model_name == 'Moments':
            return PooledMoments(**state_dict_reduced)
        else:
            raise Exception("Don't know how to handle model_name %s!" % model_name)

    def summarize_representation(self, data=None, summary_func='mse', by_angle=False):
        r"""summarize representation by key and (optionally) quadrant

        This takes data and summarizes it within each key of the
        dictionary. The intended use case is to get the mean-squared
        error (by passing ``data=metamer.representation_error()``)
        within each orientation and scale. With ``by_angle=True``, also
        breaks it down by quadrant.

        Parameters
        ----------
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to convert. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation`` or the output, with the exact same
            structure
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
        if data is not None:
            if not isinstance(data, dict):
                data = self.output_to_representation(data)
        else:
            data = self.representation
            if data is None:
                raise Exception("No representation to summarize!")
        if not isinstance(data, dict):
            data = {'representation': data}
        summarized = {}
        if summary_func == 'mse':
            summary_func = lambda x: torch.pow(x, 2).mean()
        elif summary_func == 'l2':
            summary_func = lambda x: torch.norm(x, 2)
        for k, v in data.items():
            if by_angle:
                if isinstance(k, str):
                    mask_k = 0
                else:
                    mask_k = k[0]
                for i in range(4):
                    mask = self._spatial_masks[(mask_k, f'region_{i}')]
                    summarized[(k, f'region_{i}')] = summary_func(v[..., mask]).item()
            else:
                summarized[k] = summary_func(v).item()
        return summarized

    def _representation_for_plotting(self, batch_idx=0, data=None):
        r"""Get the representation in the form required for plotting

        PooledVentralStream objects' representation has a lot of structure:
        each consists of some number of different representation types,
        each averaged per window. And the windows themselves are
        structured: we have several different eccentricity bands, each
        of which contains the same number of angular windows. We want to
        use this structure when plotting the representation, as it makes
        it easier to see what's goin on.

        The representation is either a 3d tensor, with (batch, channel,
        windows) or a dictionary of tensors with that structure, with
        each key corresponding to a different type of representation. We
        transform those 3d tensors into 1d tensors for ease of plotting,
        picking one of the batches (we only ever have 1 channel)

        We allow an optional ``data`` argument. If set, we use this data
        instead of ``self.representation``.

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to get in shape. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        representation_copy : np.ndarray
            The expanded copy of the representation, which is either a
            1d tensor (if ``data``/``self.representation`` was a tensor)
            or a dict full of 1d tensors

        """
        if data is None:
            data = self.representation

        if not isinstance(data, dict):
            data = self.output_to_representation(data)
        rep_copy = {}
        for k, v in data.items():
            # if that batch_idx is included. otherwise, assume it's
            # been flattened already
            if len(v.shape) > 1:
                v = v[batch_idx]
            rep_copy[k] = po.to_numpy(v).flatten()
        return rep_copy

    def representation_to_output(self, data=None):
        r"""convert representation to output

        This takes data that looks like ``self.representation`` (i.e., a
        dictionary whose keys are ``"mean_luminance"`` and ``(scale,
        orientation)`` tuples) and flattens it to an output that looks
        like what we get from the ``forward()`` call.

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts representation to output.

        Parameters
        ----------
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to convert. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure

        """
        if data is None:
            data = self.representation
        return torch.cat(list(data.values()), dim=2)

    def output_to_representation(self, data):
        r"""convert output to representation

        This takes data that looks like the output from the
        ``forward()`` call (a big 3d tensor) and returns something that
        looks like ``self.representation`` (i.e., a dictionary whose
        keys are ``"mean_luminance"`` and ``(scale, orientation)``
        tuples).

        NOTE: for this to work, ``forward()`` must have been called
        before

        Plenoptic models have two 'modes' for their representation: the
        one returned by the ``forward()`` call ("output"), which must be
        a 3d or 4d tensor, and the one stored as ``self.representation``
        ("representation"), which can be structured however you want,
        and probably represents the structure of the data. For example,
        a dictionary of tensors, where keys naming the different types
        of statistics. We want functions to convert between the two of
        them. This converts output to representation.

        Parameters
        ----------
        data : torch.Tensor, np.ndarray, dict, optional
            The data to convert. Should look like the value returned by
            this model's forward call (i.e., a 3d tensor).

        """
        data_dict = {}
        idx = 0
        for k, v in self.representation.items():
            data_dict[k] = data[:, :, idx:idx+v.shape[-1]].reshape(v.shape)
            idx += v.shape[-1]
        data = data_dict
        return data

    @classmethod
    def _get_title(cls, title_list, idx, default_title):
        r"""helper function for dealing with the default way we handle title

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

    def update_plot(self, axes, batch_idx=0, data=None):
        r"""Update the information in our representation plot

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

        We can optionally accept a data argument, in which case it
        should look just like the representation of this model (or be
        able to transformed into that form, see
        ``PooledV1._representation_for_plotting`).

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Parameters
        ----------
        axes : list
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to show on the plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        stem_artists : list
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        data = self._representation_for_plotting(batch_idx, data)
        if not isinstance(data, dict):
            data = {'rep': data}
        for ax, d in zip(axes, data.values()):
            sc = update_stem(ax.containers[0], d)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists


class PooledRGC(PooledVentralStream):
    r"""Model RGCs by pooling pixel intensity

    This model averages together the pixel intensities in each of its
    pooling windows to generate a super simple
    representation. Currently, does not do anything to model the optics
    of the eye (no lens point-spread function), the photoreceptors (no
    cone lattice), or the center-surround nature of retinal ganglion
    cells' receptive fields.

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_t-{transition_region_width}.pt``.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows.
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.simul.non_linearities.generate_norm_stats``. If this is an
        empty dict, we don't normalize the model.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then ``std_dev``
        must be set.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    windows : torch.Tensor
        A list of 3d tensors containing the pooling windows in which the
        pixel intensities are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    image : torch.Tensor
        A 4d containing the image most recently analyzed.
    cone_responses : torch.Tensor
        A 4d tensor containing the cone responses to the most recent image
        analyzed. This is a copy of ``image``, but may be normalized.
    representation : torch.Tensor
        A tensor containing the averages of the pixel intensities within
        each pooling window for ``self.image``. This will be 3d: (batch,
        channel, windows).
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
    cached_paths : list
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.simul.non_linearities.generate_norm_stats``. If this is an
        empty dict, we don't normalize the model.
    to_normalize : list
        List of attributes that we want to normalize by whitening

    """

    def __init__(self, scaling, img_res, min_eccentricity=.5,
                 max_eccentricity=15, transition_region_width=.5,
                 normalize_dict={}, cache_dir=None, window_type='cosine',
                 std_dev=None):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         transition_region_width=transition_region_width,
                         cache_dir=cache_dir, window_type=window_type,
                         std_dev=std_dev, normalize_dict=normalize_dict)
        self.state_dict_reduced.update({'model_name': 'RGC'})
        self.image = None
        self.representation = None
        self.to_normalize += ['cone_responses']

    def forward(self, image):
        r"""Generate the RGC representation of an image

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d

        Returns
        -------
        representation : torch.Tensor
            A 3d tensor containing the averages of the pixel intensities
            within each pooling window for ``image``

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.image = image.detach().clone()
        self.cone_responses = image.clone()
        if self.normalize_dict:
            self = zscore_stats(self.normalize_dict, self)
        self.representation = {'mean_luminance': self.PoolingWindows(self.cone_responses)}
        return self.representation_to_output()

    def _plot_helper(self, n_cols=1, figsize=(10, 5), ax=None, title=None, batch_idx=0, data=None):
        r"""helper function for plotting that takes care of a lot of the standard stuff

        I'm pretty sure that n_cols should only ever be 1, but provided as an
        option, just in case.

        Parameters
        ----------
        n_cols : int
            The number of columns in the (sub-)figure we're creating (we
            always have a single row)
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'Mean pixel intensity in each window'
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).

        Returns
        -------
        ax : matplotlib.pyplot.axis
            The axis we've set up
        data : np.ndarray
            The output of self._representation_for_plotting(batch_idx, data)
        title : str
            The title to use

        """
        if not isinstance(title, list):
            title = [title] * n_cols
        if ax is None:
            fig, ax = plt.subplots(1, n_cols, figsize=figsize)
        else:
            warnings.warn("ax is not None, so we're ignoring figsize...")
        title = self._get_title(title, 0, 'mean pixel intensity')
        data = self._representation_for_plotting(batch_idx, data)
        return ax, data, title

    def plot_representation(self, figsize=(10, 5), ylim=None, ax=None, title=None, batch_idx=0,
                            data=None):
        r"""plot the representation of the RGC model

        Because our model just takes the average pixel intensities in each
        window, our representation plot is just a simple stem plot showing each
        of these average intensities (different positions on the x axis
        correspond to different windows). We have a small break in the data to
        show where we've moved out to the next eccentricity ring.

        This looks better when it's wider than it is tall (like the default
        figsize suggests)

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ylim : tuple, False, or None
            If a tuple, the y-limits to use for this plot. If None, we use
            the default, slightly adjusted so that the minimum is 0. If
            False, we do nothing.
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use ``self.representation`` . Else,
            should look like ``self.representation``, with the exact same
            structure (e.g., as returned by ``metamer.representation_error()``
            or another instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        ax, data, title = self._plot_helper(1, figsize, ax, title, batch_idx, data)
        clean_stem_plot(data['mean_luminance'], ax, title, ylim)
        axes = [ax]
        # fig won't always be defined, but this will return the figure belonging to our axis
        return ax.figure, axes

    def plot_representation_image(self, figsize=(5, 5), ax=None, title=None, batch_idx=0,
                                  data=None, vrange='indep1', zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows

        The RGC model consists of average pixel intensity per window, thus, our
        representation has a single value for each pooling window, so we take
        that value and multiply it by the pooling window, and then sum across
        all windows. Thus the value at a single pixel shows a weighted sum of
        the representation.

        By setting ``data``, you can use this to visualize any vector with the
        same length as the number of windows. For example, you can view metamer
        synthesis error by setting ``data=metamer.representation_error()``
        (then you'd probably want to set ``vrange='auto0'`` in order to change
        the colormap to a diverging one cenetered at 0).

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).
        vrange : str or tuple, optional
            The vrange value to pass to pyrtools.imshow
        zoom : float or int, optional
            If a float, must be castable to an int or an inverse power
            of 2. The amount to zoom the images in/out by.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        ax : matplotlib.pyplot.axis
            A list of axes that contain the plots we've created

        """
        ax, data, title = self._plot_helper(1, figsize, ax, title, batch_idx, data)
        ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
        # project expects a 3d tensor
        data = self.PoolingWindows.project(torch.tensor(data['mean_luminance']).unsqueeze(0).unsqueeze(0))
        po.imshow(data, vrange=vrange, ax=ax, title=title, zoom=zoom)
        return ax.figure, ax


class PooledV1(PooledVentralStream):
    r"""Model V1 by pooling spectral energy (using Steerable Pyramid)

    This just models V1 as containing complex cells and a representation
    of the mean luminance. For the complex cells, we take the outputs of
    the complex steerable pyramid and takes the complex modulus of them
    (that is, squares, sums, and takes the square root across the real
    and imaginary parts; this is a phase-invariant measure of the local
    magnitude). The mean luminance representation is the same as that
    computed by the PooledRGC model.

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    We will calculate the minimum eccentricity at which the area of the
    windows at half-max exceeds one pixel at each scale. For scales
    beyond the first however, we will not throw an Exception if this
    value is below ``min_eccentricity``. We instead print a warning to
    alert the user and use this value as ``min_eccentricity`` when
    creating the plots. In order to see what this value was, see
    ``self.calculated_min_eccentricity_degrees``

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_t-{transition_region_width}.pt``. We'll cache
    each scale separately, changing the img_res (and potentially
    min_eccentricity) values in that save path appropriately.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
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
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    normalize_dict : dict, optional
        Dict containing the statistics to normalize, as generated by
        ``po.optim.generate_norm_stats``. If this is an empty dict, we
        don't normalize the model. If it's non-empty, we expect it to
        have only two keys: "complex_cell_responses" and
        "cone_responses"
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then ``std_dev``
        must be set.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).
    moments : list, optional
        Subset of 2, 3, 4. Which moments (other than the mean) to include in
        this model's representation.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    num_scales : int, optional
        The number of scales (spatial frequency bands) in the steerable
        pyramid we use to build the V1 representation
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
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
    windows : torch.Tensor
        A list of 3d tensors containing the pooling windows in which the
        complex cell responses are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    image : torch.Tensor
        A 4d tensor containing the most recent image analyzed.
    cone_responses : torch.Tensor
        A 4d tensor containing the cone responses to the most recent image
        analyzed. This is a copy of ``image``, but may be normalized.
    pyr_coeffs : dict
        The dictionary containing the (complex-valued) coefficients of
        the steerable pyramid built on ``self.image``. Each of these is
        5d: ``(1, 1, *img_res, 2)``. The first two dimensions are for
        batch and channel, the last dimension contains the real and
        imaginary components of the complex number; channel is
        unnecessary for us but we might be able to get batch working.
    complex_cell_responses : dict
        Dictionary containing the complex cell responses, the squared
        and summed (i.e., the squared complex modulus) of
        ``self.pyr_coeffs``. Does not include the residual high- and
        low-pass bands. Each of these is now 4d: ``(1, 1, *img_res)``.
    mean_luminance : torch.Tensor
        A 1d tensor representing the mean luminance of the image, found
        by averaging the pixel values of the image using the windows at
        the lowest scale. This is identical to the PooledRGC
        representation of the image with the same ``scaling`` value.
    representation : torch.Tensor
        A dictionary containing the 'complex cell responses' (that is,
        the squared, summed, and square-rooted outputs of the complex
        steerable pyramid) and the mean luminance of the image in the
        pooling windows. Each of these is a 3d tensor: (batch, channel,
        windows)
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
    cached_paths : list
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.optim.generate_norm_stats``. If this is an empty dict, we
        don't normalize the model. If it's non-empty, we expect it to
        have only two keys: "complex_cell_responses" and
        "cone_responses"
    to_normalize : list
        List of attributes that we want to normalize by whitening (for
        PooledV1, that's just "complex_cell_responses")
    scales : list
        List of the scales in the model, from fine to coarse. Used for
        synthesizing in coarse-to-fine order

    """
    def __init__(self, scaling, img_res, num_scales=4, order=3, min_eccentricity=.5,
                 max_eccentricity=15, transition_region_width=.5, normalize_dict={},
                 cache_dir=None, window_type='cosine', std_dev=None, moments=[]):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity, num_scales,
                         transition_region_width=transition_region_width,
                         cache_dir=cache_dir, window_type=window_type, std_dev=std_dev,
                         normalize_dict=normalize_dict, moments=moments)
        self.state_dict_reduced.update({'order': order, 'model_name': 'V1',
                                        'num_scales': num_scales})
        self.num_scales = num_scales
        self.order = order
        # before the change in how complex tensors were handled, the default
        # was tight_frame=True. set that so it's comparable.
        try:
            self.complex_steerable_pyramid = po.simul.Steerable_Pyramid_Freq(img_res, self.num_scales,
                                                                             self.order, is_complex=True,
                                                                             tight_frame=True)
        except AttributeError:
            # at a certain point (May 17, 2023), plenoptic removed the underscores here
            self.complex_steerable_pyramid = po.simul.SteerablePyramidFreq(img_res, self.num_scales,
                                                                           self.order, is_complex=True,
                                                                           tight_frame=True)
        self.scales = ['mean_luminance'] + list(range(num_scales))[::-1]
        self.image = None
        self.pyr_coeffs = None
        self.complex_cell_responses = None
        self.mean_complex_cell_responses = None
        self.mean_luminance = None
        self.representation = None
        self.to_normalize += ['cone_responses', 'complex_cell_responses']

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

        Returns:
            Module: self
        """
        self.complex_steerable_pyramid.to(*args, **kwargs)
        super(self.__class__, self).to(do_windows=do_windows, *args, **kwargs)
        return self

    def forward(self, image, scales=[]):
        r"""Generate the V1 representation of an image

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
        if not scales:
            scales = self.scales
        self.representation = {}
        self.complex_cell_responses = {}
        self.mean_complex_cell_responses = {}
        # this is a little weird here: the image that we detach and
        # clone here is just a copy that we keep around for later
        # examination.
        self.image = image.detach().clone()
        # we save this here so that it can be normalized. At this point,
        # it's not normalized, but it will be during the
        # zscore_stats(self.normalize_dict, self) call below. We also
        # zscore cone_responses there because we want the values that go
        # into the mean pixel intensity to be normalized but not the
        # values that go into the steerable pyramid.
        self.cone_responses = image.clone()
        self.pyr_coeffs = {}
        if any([i in self.complex_steerable_pyramid.scales for i in scales]):
            self.pyr_coeffs.update(self.complex_steerable_pyramid(self.cone_responses,
                                                                  scales))
        if self.pyr_coeffs:
            # to get the energy, we just square and take the absolute value
            # (since this is a complex tensor, this is equivalent to summing
            # across the real and imaginary components). the if statement
            # avoids the residuals
            self.complex_cell_responses = dict((k, torch.pow(v, 2).abs())
                                               for k, v in self.pyr_coeffs.items()
                                               if not isinstance(k, str))
        if self.normalize_dict:
            self = zscore_stats(self.normalize_dict, self)
        if self.complex_cell_responses:
            self.mean_complex_cell_responses = self.PoolingWindows(self.complex_cell_responses)
        self.representation.update(self.mean_complex_cell_responses)
        if 'mean_luminance' in scales:
            self.mean_luminance = self.PoolingWindows(self.cone_responses)
            self.representation['mean_luminance'] = self.mean_luminance
            # use the original image, not cone_responses, because that might be
            # normalized and we want the moments to be computed on the
            # un-normalized image (to avoid possible negative values)
            moments = self._calculate_moments(image,
                                              2 in self._moments,
                                              3 in self._moments,
                                              4 in self._moments)
            self.representation.update({f'image_moment_{k}': v for
                                        k, v in moments.items()})
        return self.representation_to_output()

    def _representation_for_plotting(self, batch_idx=0, data=None):
        r"""Get data into the form required for plotting

        PooledV1 objects' representation has a lot of
        structure: each consists of some number of different
        representation types, each averaged per window. And the windows
        themselves are structured: we have several different
        eccentricity bands, each of which contains the same number of
        angular windows. We want to use this structure when plotting the
        representation, as it makes it easier to see what's goin on.

        The representation is a dictionary of tensors with that
        structure, with each key corresponding to a different type of
        representation. We transform those 4d tensors into 1d tensors
        for ease of plotting, picking one of the batches (we only ever
        have 1 channel) and collapsing the different windows onto one
        dimension, putting a NaN between each eccentricity band.

        We allow an optional ``data`` argument. If set, we use this data
        instead of ``self.representation``. In addition to being
        structured like the ``self.representation`` dictionary, it can
        also be an array or tensor, like this object's ``forward``
        method returns (and thus is stored by the various synthesis
        objects). This function then transforms that array into the
        dictionary we expect (taking advantage of the fact that, from
        that dictionary, we know the number of representation types and
        the number of each type of window) and passes it on to the
        parent class's ``_representation_for_plotting`` to finish it up.

        We expect this to be plotted using
        ``plenoptic.tools.display.clean_stem_plot``, and return a tuple
        ``xvals`` for use with that function to replace the base line
        (by default, ``plt.stem`` doesn't insert a break in the baseline
        if there's a NaN in the data, but we want that for ease of
        visualization)

        Parameters
        ----------
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should be a dictionary of 4d
            tensors like ``self.representation``, or a 3d tensor, like
            the value returned by ``self.forward()`` (e.g., as returned
            by ``metamer.representation_error()``).

        Returns
        -------
        representation_copy : np.ndarray
            The expanded copy of the representation, which is a dict
            full of 1d tensors, with np.nan inserted between each
            eccentricity band
        xvals : tuple
            A 2-tuple of lists, containing the start (``xvals[0]``) and
            stop (``xvals[1]``) x values for plotting. For use with
            plt.hlines, like so: ``plt.hlines(len(xvals[0])*[0],
            xvals[0], xvals[1])``

        """
        if data is not None and not isinstance(data, dict):
            data = self.output_to_representation(data)
        return super()._representation_for_plotting(batch_idx, data)

    def _plot_helper(self, n_rows, n_cols, figsize=(25, 15), ax=None, title=None, batch_idx=0,
                     data=None):
        r"""helper function for plotting that takes care of a lot of the standard stuff

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
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing our subplots
        gs : matplotlib.gridspec.GridSpec
            The GridSpec object to use for creating subplots. You should
            use it with ``fig`` to add subplots by indexing into it,
            like so: ``fig.add_subplot(gs[0, 1])``
        data : np.ndarray
            The output of self._representation_for_plotting(batch_idx, data)
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
        data = self._representation_for_plotting(batch_idx, data)
        if isinstance(title, str):
            # then this is a single str, so we'll make it the same on
            # every subplot
            title = (n_rows * n_cols) * [title]
        return fig, gs, data, title

    def plot_representation(self, figsize=(25, 15), ylim=None, ax=None, title=None, batch_idx=0,
                            data=None):
        r"""plot the representation of the V1 model

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
            as ``data`` (or, if ``data`` is None, of
            ``self.representation``), and each value will be put above
            the subplots. If a str, the same title will be put above
            each subplot. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If it includes a '|' (pipe), then we'll
            append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        n_cols = self.num_scales + 1
        n_rows = max(self.order + 1, len(self._moments) + 1)
        col_multiplier = 2
        col_offset = 1
        fig, gs, data, title_list = self._plot_helper(2*n_rows, 2*n_cols, figsize,
                                                      ax, title, batch_idx, data)
        # 0-indexed moment N
        moment_n_dict = {'mean_luminance': 0, 'second': 1,
                         'third': 2, 'fourth': 3}
        axes = []
        for i, (k, v) in enumerate(data.items()):
            if isinstance(k, tuple):
                t = self._get_title(title_list, i, "scale %s, band %s" % k)
                ax = fig.add_subplot(gs[int(2*k[1]):int(2*(k[1]+1)), int(col_multiplier*k[0]):
                                        int(col_multiplier*(k[0]+col_offset))])
                ax = clean_stem_plot(v, ax, t, ylim)
                axes.append(ax)
            else:
                t = self._get_title(title_list, i, k)
                moment_n = moment_n_dict[k.replace('image_moment_', '')]
                # last column, and moment_n determines the row
                ax = fig.add_subplot(gs[2*moment_n:2*(moment_n+1), 2*(n_cols-1):])
                ax = clean_stem_plot(v, ax, t, ylim)
                axes.append(ax)
        return fig, axes

    def plot_representation_image(self, figsize=(22, 11), ax=None, title=None, batch_idx=0,
                                  data=None, vrange='auto1', zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows

        Our representation is composed of pooled energy at several
        different scales and orientations, plus the pooled mean pixel
        intensity. In order to visualize these as images, we take each
        statistic, multiply it by the pooling windows, then sum across
        windows, as in
        ``PooledRGC.plot_representation_image``. We also sum
        across orientations at the same scale, so that we end up with an
        image for each scale (each will be a different size), plus one
        for mean pixel intensity, which we attempt to zoom so that they
        are all shown at approximately the same size.

        Similar to ``self.plot_representation``, you can set ``data`` to
        visualize something else. As in that function, ``data`` must
        have the same structure as ``self.representation`` (i.e., a
        dictionary of 3d tensors/arrays) or as the value returned by
        ``self.forward()`` (i.e., a large 3d tensor/array). For example,
        you can view metamer synthesis error by setting
        ``data=metamer.representation_error()`` (then you'd probably
        want to set ``vrange='auto0'`` in order to change the colormap
        to a diverging one cenetered at 0).

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create (ignored if ``ax`` is not
            None)
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on (in
            which case we ignore ``figsize``). If None, we create our
            own figure to hold it
        title : str, list, or None, optional
            Titles to use. If a list of strings, must be the same length
            as ``data`` (or, if ``data`` is None, of
            ``self.representation``), and each value will be put above
            the subplots. If a str, the same title will be put above
            each subplot. If None, we use the default choice, which
            specifies the scale and orientation of each plot (and the
            mean intensity). If it includes a '|' (pipe), then we'll
            append the default to the other side of the pipe.
        batch_idx : int, optional Which
            index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).
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
        n_cols = max(self.num_scales, len(self._moments) + 1)
        n_rows = 2
        fig, gs, data, title_list = self._plot_helper(n_rows, n_cols, figsize, ax,
                                                      title, batch_idx, data)
        titles = []
        axes = []
        imgs = []
        zooms = []
        # project expects a dictionary of 3d tensors
        data = self.PoolingWindows.project(dict((k, torch.Tensor(v).unsqueeze(0).unsqueeze(0))
                                                for k, v in data.items()))
        for i in self.scales:
            if isinstance(i, str):
                continue
            titles.append(self._get_title(title_list, i, "scale %s" % i))
            img = torch.zeros_like(data[(i, 0)])
            for j in range(self.order+1):
                img += data[(i, j)]
            ax = fig.add_subplot(gs[0, int(i)])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'], ['x', 'y'])
            imgs.append(img)
            axes.append(ax)
            if isinstance(i, int):
                zooms.append(zoom * round(data[(0, 0)].shape[-1] / img.shape[-1]))
            elif isinstance(i, float):
                zooms.append(zoom * round(data[(0.5, 0)].shape[-1] / img.shape[-1]))
        for i, k in enumerate(['mean_luminance'] + self._moments):
            k = {2: 'image_moment_second', 3: 'image_moment_third',
                 4: 'image_moment_fourth'}.get(k, k)
            ax = fig.add_subplot(gs[1, i])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'],
                               ['x', 'y'])
            axes.append(ax)
            titles.append(self._get_title(title_list, len(self.scales)+i-1, k))
            imgs.append(data[k])
            zooms.append(zoom)
        vrange, cmap = pt.tools.display.colormap_range(imgs, vrange)
        for ax, img, t, vr, z in zip(axes, imgs, titles, vrange, zooms):
            po.imshow(img, ax=ax, vrange=vr, cmap=cmap, title=t, zoom=z)
        return fig, axes


class PooledMoments(PooledVentralStream):
    r"""Pool pixel moments.

    This model computes the first N un-normalized moments of the pixel values
    in pooling windows across an image (where N is between 2 and 4; note this
    means we always include the mean). This is not a biological model, but a
    simple null model.

    Note that we will calculate the minimum eccentricity at which the
    area of the windows at half-max exceeds one pixel (based on
    ``scaling``, ``img_res`` and ``max_eccentricity``) and, if
    ``min_eccentricity`` is below that, will throw an Exception.

    We can optionally cache the windows tensor we create, if
    ``cache_dir`` is not None. In that case, we'll also check to see if
    appropriate cached windows exist before creating them and load them
    if they do. The path we'll use is
    ``{cache_dir}/scaling-{scaling}_size-{img_res}_e0-{min_eccentricity}_
    em-{max_eccentricity}_t-{transition_region_width}.pt``.

    NOTE: that we're assuming the input to this model contains values
    proportional to photon counts; thus, it should be a raw image or
    other linearized / "de-gamma-ed" image (all images meant to be
    displayed on a standard display will have been gamma-corrected,
    which involves raising their values to a power, typically 1/2.2).

    Parameters
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling
        windows. Other pooling windows parameters
        (``radial_to_circumferential_ratio``,
        ``transition_region_width``) cannot be set here. If that ends up
        being of interest, will change that.
    img_res : tuple
        The resolution of our image (should therefore contains
        integers). Will use this to generate appropriately sized pooling
        windows.
    min_eccentricity : float, optional
        The eccentricity at which the pooling windows start.
    max_eccentricity : float, optional
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods. 0.5 (the default) is the
        value used in the paper [1]_.
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.simul.non_linearities.generate_norm_stats``. If this is an
        empty dict, we don't normalize the model.
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    window_type : {'cosine', 'gaussian'}
        Whether to use the raised cosine function from [1]_ or a Gaussian that
        has approximately the same structure. If cosine,
        ``transition_region_width`` must be set; if gaussian, then ``std_dev``
        must be set.
    std_dev : float or None, optional
        The standard deviation of the Gaussian window. WARNING -- For
        now, we only support ``std_dev=1`` (in order to ensure that the
        windows tile correctly, intersect at the proper point, follow
        scaling, and have proper aspect ratio; not sure we can make that
        happen for other values).
    moments : list, optional
        Subset of 2, 3, 4. Which moments (other than the mean) to include in
        this model's representation.

    Attributes
    ----------
    scaling : float
        Scaling parameter that governs the size of the pooling windows.
    min_eccentricity : float
        The eccentricity at which the pooling windows start.
    max_eccentricity : float
        The eccentricity at which the pooling windows end.
    transition_region_width : `float`, optional
        The width of the transition region, parameter :math:`t` in
        equation 9 from the online methods.
    windows : torch.Tensor
        A list of 3d tensors containing the pooling windows in which the
        pixel intensities are averaged. Each entry in the list
        corresponds to a different scale and thus is a different size
        (though they should all have the same number of windows)
    image : torch.Tensor
        A 4d containing the image most recently analyzed.
    cone_responses : torch.Tensor
        A 4d tensor containing the cone responses to the most recent image
        analyzed. This is a copy of ``image``, but may be normalized.
    representation : torch.Tensor
        A tensor containing the averages of the pixel intensities within
        each pooling window for ``self.image``. This will be 3d: (batch,
        channel, windows).
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
    cached_paths : list
        List of strings, one per scale, that we either saved or loaded
        the cached windows tensors from
    normalize_dict : dict
        Dict containing the statistics to normalize, as generated by
        ``po.simul.non_linearities.generate_norm_stats``. If this is an
        empty dict, we don't normalize the model.
    to_normalize : list
        List of attributes that we want to normalize by whitening

    """

    def __init__(self, scaling, img_res, min_eccentricity=.5,
                 max_eccentricity=15, transition_region_width=.5,
                 normalize_dict={}, cache_dir=None, window_type='cosine',
                 std_dev=None, moments=[2, 3, 4]):
        super().__init__(scaling, img_res, min_eccentricity, max_eccentricity,
                         transition_region_width=transition_region_width,
                         cache_dir=cache_dir, window_type=window_type,
                         std_dev=std_dev, normalize_dict=normalize_dict,
                         moments=moments)
        self.state_dict_reduced.update({'model_name': 'Moments'})
        self.image = None
        self.representation = None
        self.to_normalize += ['cone_responses']

    def forward(self, image):
        r"""Generate the PooledMoments representation of an image.

        Parameters
        ----------
        image : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width). If it has fewer than 4 dimensions,
            we will unsqueeze it until its 4d

        Returns
        -------
        representation : torch.Tensor
            A 3d tensor containing the averages of the pixel intensities
            within each pooling window for ``image``, plus the specified
            un-normalized moments within those windows.

        """
        while image.ndimension() < 4:
            image = image.unsqueeze(0)
        self.image = image.detach().clone()
        self.cone_responses = image.clone()
        if self.normalize_dict:
            self = zscore_stats(self.normalize_dict, self)
        self.representation = {'mean_luminance': self.PoolingWindows(self.cone_responses)}
        moments = self._calculate_moments(self.cone_responses, 2 in self._moments,
                                          3 in self._moments, 4 in self._moments,
                                          self.representation['mean_luminance'])
        self.representation.update({f'image_moment_{k}': v for k, v in moments.items()})
        return self.representation_to_output()

    def _plot_helper(self, n_rows, n_cols, figsize=(10, 10), ax=None,
                     title=None, batch_idx=0, data=None):
        r"""helper function for plotting that takes care of a lot of the standard stuff

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
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure, or
            the structure returned by ``self.forward`` (e.g., as
            returned by ``metamer.representation_error()`` or another
            instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing our subplots
        gs : matplotlib.gridspec.GridSpec
            The GridSpec object to use for creating subplots. You should
            use it with ``fig`` to add subplots by indexing into it,
            like so: ``fig.add_subplot(gs[0, 1])``
        data : np.ndarray
            The output of self._representation_for_plotting(batch_idx, data)
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
        data = self._representation_for_plotting(batch_idx, data)
        if isinstance(title, str):
            # then this is a single str, so we'll make it the same on
            # every subplot
            title = (n_rows * n_cols) * [title]
        return fig, gs, data, title

    def plot_representation(self, figsize=(10, 10), ylim=None, ax=None,
                            title=None, batch_idx=0, data=None):
        r"""Plot the representation of the PooledMoments model.

        We plot each included moment as a stem plot on a separate axis. We have
        a small break in the data to show where we've moved out to the next
        eccentricity ring.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ylim : tuple, False, or None
            If a tuple, the y-limits to use for this plot. If None, we use
            the default, slightly adjusted so that the minimum is 0. If
            False, we do nothing.
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use ``self.representation`` . Else,
            should look like ``self.representation``, with the exact same
            structure (e.g., as returned by ``metamer.representation_error()``
            or another instance of this class).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        axes : list
            A list of axes that contain the plots we've created

        """
        nrows = 2
        ncols = len(self._moments)//2 + 1
        fig, gs, data, title_list = self._plot_helper(nrows, ncols,
                                                      figsize, ax, title,
                                                      batch_idx, data)
        axes = []
        for i, (k, v) in enumerate(data.items()):
            t = self._get_title(title_list, i, k)
            ax = fig.add_subplot(gs[i%2, i//2])
            ax = clean_stem_plot(v, ax, t, ylim)
            axes.append(ax)
        return fig, axes

    def plot_representation_image(self, figsize=(11, 11), ax=None, title=None, batch_idx=0,
                                  data=None, vrange='indep1', zoom=1):
        r"""Plot representation as an image, using the weights from PoolingWindows.

        The PooledMoments model consists of the first N moments per window. We
        thus create N plots, each of which has a single value per pooling
        window; we take that value and multiply it by the pooling window, and
        then sum across all windows. Thus the value at a single pixel shows a
        weighted sum of the representation.

        By setting ``data``, you can use this to visualize any vector with the
        same length as the number of windows. For example, you can view metamer
        synthesis error by setting ``data=metamer.representation_error()``
        (then you'd probably want to set ``vrange='auto0'`` in order to change
        the colormap to a diverging one cenetered at 0).

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create
        ax : matplotlib.pyplot.axis or None, optional
            If not None, the axis to plot this representation on. If
            None, we create our own 1 subplot figure to hold it
        title : str or None, optional
            The title to put above this axis. If you want no title, pass
            the empty string (``''``). If None, will use the default,
            'mean pixel intensity'. If it includes a '|' (pipe), then
            we'll append the default to the other side of the pipe.
        batch_idx : int, optional
            Which index to take from the batch dimension
        data : torch.Tensor, np.ndarray, dict or None, optional
            The data to plot. If None, we use
            ``self.representation``. Else, should look like
            ``self.representation``, with the exact same structure
            (e.g., as returned by ``metamer.representation_error()`` or
            another instance of this class).
        vrange : str or tuple, optional
            The vrange value to pass to pyrtools.imshow
        zoom : float or int, optional
            If a float, must be castable to an int or an inverse power
            of 2. The amount to zoom the images in/out by.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot
        ax : matplotlib.pyplot.axis
            A list of axes that contain the plots we've created

        """
        nrows = 2
        ncols = len(self._moments)//2 + 1
        fig, gs, data, title_list = self._plot_helper(nrows, ncols, figsize, ax, title,
                                                      batch_idx, data)
        # project expects a dictionary of 3d tensors
        data = self.PoolingWindows.project(dict((k, torch.Tensor(v).unsqueeze(0).unsqueeze(0))
                                                for k, v in data.items()))
        axes = []
        for i, (k, v) in enumerate(data.items()):
            title = self._get_title(title_list, i, k)
            ax = fig.add_subplot(gs[i % 2, i // 2])
            ax = clean_up_axes(ax, False, ['top', 'right', 'bottom', 'left'],
                               ['x', 'y'])
            po.imshow(v, vrange=vrange, ax=ax, title=title, zoom=zoom)
            axes.append(ax)
        return fig, axes
