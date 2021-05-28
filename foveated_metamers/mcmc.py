#!/usr/bin/env python3
"""Code for using MCMC to fit psychophysical data."""
# numpyro's MCMC is much faster than pyro's, especially for hierarchical
# models:
# https://forum.pyro.ai/t/mcmc-pyro-speed-compare-to-numpyro-on-google-colab/1621
import numpyro
import xarray
import pandas as pd
import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.scipy as jsc
import arviz as az


def calculate_discriminability(scaling, proportionality_factor, critical_scaling):
    r"""Calculate disriminability at given scaling, for specified parameters.

    This comes from the Online Methods section of [1]_, equation 17.

    Parameters
    ----------
    scaling : jnp.ndarray
        Scaling value(s) to calculate discriminability for.
    proportionality_factor : jnp.ndarray or float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : jnp.ndarray or float
        The "threshold" of the curve, the scaling value at which
           parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.

    Returns
    -------
    discrim : jnp.ndarray
        discriminability ($d^2$ in [1]_) at each scaling value.

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    vals = proportionality_factor * (1 - (critical_scaling**2 / scaling**2))
    # this has to be non-negative
    return vals.clip(0)


def proportion_correct_curve(scaling, proportionality_factor, critical_scaling):
    r"""Compute the proportion correct curve, as function of parameters.

    This comes from the Online Methods section of [1]_, equation 18.

    Parameters
    ----------
    scaling : jnp.ndarray
        Scaling value(s) to calculate discriminability for.
    proportionality_factor : jnp.ndarray or float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. This will vary more across subjects
        and isn't as directly relevant for this study.
    critical_scaling : jnp.ndarray or float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study.

    Returns
    -------
    proportion_correct : jnp.ndarray
        The proportion correct curve at each scaling value, as given by the
        parameter values

    References
    ----------
    .. [1] Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature
       Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889

    """
    norm = jsc.stats.norm
    discrim = calculate_discriminability(scaling, proportionality_factor,
                                         critical_scaling)
    # we use the fact that norm.cdf(-x) = 1 - norm.cdf(x) to speed up the
    # following, which is equivalent to: norm.cdf(d/sqrt(2)) * norm.cdf(d/2) +
    # norm.cdf(-d/sqrt(2)) * norm.cdf(-d/2)
    norm_cdf_sqrt_2 = norm.cdf(discrim / np.sqrt(2))
    norm_cdf_2 = norm.cdf(discrim / 2)
    return norm_cdf_sqrt_2 * norm_cdf_2 + (1-norm_cdf_sqrt_2) * (1-norm_cdf_2)


def partially_pooled_response_model(scaling, model='V1', observed_responses=None):
    r"""Partially pooled probabilistic model of responses, with lapse rate.

    This is "partially pooled" because of how we handle the image and subject
    level effects on the two parameters of interest.

    - Critical scaling ($s_0$) and proportionality factor / gain ($\alpha_0$)
      are both modeled on a natural log-scale, where they're the sum of
      trial_type, image_name (crossed with trial_type), and subject_name
      (crossed with trial_type) effects. Inspired by Wallis et al, 2019.

    - Lapse rate is modeled as a completely unpooled manner, independently for
      each (trial_type, subject) (shared across images)

    Model Parameters:

    - log_a0_global_mean, log_s0_global_mean: independent for each trial_type,
      this is the high-level mean for the gain and critical scaling.

    - a0_subject_sd, s0_subject_sd: independent for each trial type,
      this is the noise for the subject-level effects for the gain and critical
      scaling.

    - a0_image_sd, s0_image_sd: independent for each trial type,
      this is the noise for the image-level effects for the gain and critical
      scaling.

    - log_a0_image, log_s0_image: partially pooled within trial_type across
      images, this is the image-level effect.

    - log_a0_subject, log_s0_subject: partially pooled within trial_type across
      subjects, this is the subject-level effect.

    - pi_l: lapse rate, independent for each (trial_type, subject).

    Thus, for an individual subject and image:

    - a0 = exp(log_a0_global_mean + log_a0_subject + log_a0_image)

    - s0 = exp(log_s0_global_mean + log_s0_subject + log_s0_image)

    NOTE: no interaction between subject and image effects!
   
    Priors:

    - pi_l: Beta(2, 50)

    - log_a0_global_mean: Normal(1.6, 1), this gives expected value of 5 for
      the exponentiated version.

    - log_s0_global_mean: Normal(-1.38, 1) for V1 (expected value ~.25 of
      exponentiated version, following from Freeman and Simoncelli, 2011);
      Normal(-4, 1) for RGC (expected value ~.018 of exponentiated version,
      from Dacey, 1992)

    - a0_subject_sd, s0_subject_sd, a0_image_sd, s0_image_sd:
      HalfCauchy(.1)

    - log_a0_image: Normal(0, a0_image_sd)

    - log_s0_image: Normal(0, s0_image_sd)

    - log_a0_subject: Normal(0, a0_subject_sd)

    - log_s0_subject: Normal(0, s0_subject_sd)

    Parameters
    ----------
    scaling : jnp.ndarray
        scaling value(s) to calculate discriminability for. Must be 5d:
        (scaling, subject_name, image_name, trial_type, model), though
        currently we only work with a singleton model dimension.
    model : {'V1', 'RGC'}
        Whether we should use V1 or RGC prior for log_s0_global_mean.
    observed_responses : jnp.ndarray or None
        observed responses to condition our pulls on. If None, don't condition.

    Returns
    -------
    responses : numpyro.sample
        Samples of responses

    """
    # because this is a 2AFC task
    chance_correct = .5
    # first get rid of model dim, since we're not using it here.
    if scaling.shape[-1] > 1:
        raise Exception("Currently, response model can only handle 1 model"
                        " at a time!")
    scaling = scaling.squeeze(-1)
    if observed_responses is not None:
        observed_responses = observed_responses.squeeze(-1)
    # we have to do a weird hacky workaround if we only have one trial_type:
    # for some reason, we can't plate over it (we get a really strange error
    # when calling mcmc.run, which I think is a numpyro bug, though this
    # function itself can be called without a problem). So, instead, we squeeze
    # out trial_types (and add it back in later when we construct the
    # InferenceData object)
    dim_offset = 0
    if scaling.shape[-1] == 1:
        scaling = scaling.squeeze(-1)
        if observed_responses is not None:
            observed_responses = observed_responses.squeeze(-1)
        dim_offset = 1
        trial_type_plate = None
    else:
        trial_type_plate = numpyro.plate('trial_type', scaling.shape[-1], dim=-1)
    image_name_plate = numpyro.plate('image_name', scaling.shape[-(2-dim_offset)],
                                     dim=-(2-dim_offset))
    subject_name_plate = numpyro.plate('subject_name', scaling.shape[-(3-dim_offset)],
                                       dim=-(3-dim_offset))
    scaling_plate = numpyro.plate('scaling', scaling.shape[-(4-dim_offset)],
                                  dim=-(4-dim_offset))
    obs_nans = None
    if observed_responses is not None:
        # where's the missing data?
        obs_nans = jnp.isnan(observed_responses)
        trials_plate = numpyro.plate('trials', observed_responses.shape[0],
                                     dim=-(5-dim_offset))
    else:
        trials_plate = numpyro.plate('trials', 1, dim=-(5-dim_offset))

    def _sample(scaling, observed_responses, obs_nans):
        # expected value of 5 for exponentiated version, which looks reasonable
        a0_global_mean = numpyro.sample('log_a0_global_mean', dist.Normal(1.6, 1))
        # different priors for the two models
        if model == 'V1':
            # expected value of .25 for exponentiated version, from Freeman and
            # Simoncelli, 2011
            s0_global_mean = numpyro.sample('log_s0_global_mean', dist.Normal(-1.38, 1))
        elif model == 'RGC':
            # expected value of .018 for exponentiated version, from Dacey, 1992
            s0_global_mean = numpyro.sample('log_s0_global_mean', dist.Normal(-4, 1))
        # something vague and positive
        s0_subject_sd = numpyro.sample('s0_subject_sd', dist.HalfCauchy(.1))
        a0_subject_sd = numpyro.sample('a0_subject_sd', dist.HalfCauchy(.1))
        s0_image_sd = numpyro.sample('s0_image_sd', dist.HalfCauchy(.1))
        a0_image_sd = numpyro.sample('a0_image_sd', dist.HalfCauchy(.1))
        with image_name_plate:
            s0_image = numpyro.sample('log_s0_image', dist.Normal(0, s0_image_sd))
            a0_image = numpyro.sample('log_a0_image', dist.Normal(0, a0_image_sd))
        with subject_name_plate:
            s0_subject = numpyro.sample('log_s0_subject', dist.Normal(0, s0_subject_sd))
            a0_subject = numpyro.sample('log_a0_subject', dist.Normal(0, a0_subject_sd))
            lapse_rate = numpyro.sample('pi_l', dist.Beta(2, 50))
            with image_name_plate:
                # combine global, subject, and image effects
                a0 = a0_global_mean + a0_subject + a0_image
                s0 = s0_global_mean + s0_subject + s0_image
                # this is the value without the lapse rate
                prop_corr = proportion_correct_curve(scaling, jnp.exp(a0), jnp.exp(s0))
                # now with the lapse rate
                prob_corr = numpyro.deterministic('probability_correct',
                                                  ((1 - lapse_rate) * prop_corr +
                                                   lapse_rate * chance_correct))
                with trials_plate, scaling_plate:
                    if obs_nans is not None:
                        # expand this out, for broadcasting purposes
                        prob_corr_nan = jnp.expand_dims(prob_corr, 0).tile((observed_responses.shape[0],
                                                                            *([1]*prob_corr.ndim)))
                        # sample the responses...
                        imputed_responses = numpyro.sample(
                            'responses_imputed',
                            dist.Bernoulli(prob_corr_nan, validate_args=True).mask(False)
                        )
                        # ...and insert the imputed responses where there are
                        # NaNs in the observed data.
                        observed_responses = jnp.where(obs_nans,
                                                       imputed_responses,
                                                       observed_responses)
                    return numpyro.sample('responses', dist.Bernoulli(prob_corr,
                                                                      validate_args=True),
                                          obs=observed_responses)
    if trial_type_plate is not None:
        with trial_type_plate:
            return _sample(scaling, observed_responses, obs_nans)
    else:
        return _sample(scaling, observed_responses, obs_nans)


def unpooled_response_model(scaling, model='V1', observed_responses=None):
    r"""Probabilistic model of responses, with lapse rate.

    This is "unpooled" because we model the parameters of interest
    independently for each psychophysical curve (i.e., each image/subject)

    - Critical scaling ($s_0$) and proportionality factor / gain ($\alpha_0$)
      are both modeled on a natural log-scale, in a completely unpooled manner,
      independently for each (trial_type, subject, image).

    - Lapse rate is modeled as a completely unpooled manner, independently for
      each (trial_type, subject) (shared across images)

    Model Parameters:

    - a0: the proportionality factor / gain, independent for each (trial_type,
      subject, image)

    - s0: the critical scaling, independent for each (trial_type, subject,
      image)

    - pi_l: lapse rate, independent for each (trial_type, subject).

    Priors:

    - pi_l: Beta(2, 50)

    - a0: Normal(1.6, 1), this gives expected value of 5 for the exponentiated
      version.

    - s0: Normal(-1.38, 1) for V1 (expected value ~.25 of exponentiated
      version, following from Freeman and Simoncelli, 2011); Normal(-4, 1) for
      RGC (expected value ~.018 of exponentiated version, from Dacey, 1992)

    Parameters
    ----------
    scaling : jnp.ndarray
        scaling value(s) to calculate discriminability for. Must be 5d:
        (scaling, subject_name, image_name, trial_type, model), though
        currently we only work with a singleton model dimension.
    model : {'V1', 'RGC'}
        Whether we should use V1 or RGC prior for log_s0_global_mean.
    observed_responses : jnp.ndarray or None
        observed responses to condition our pulls on. If None, don't condition.

    Returns
    -------
    responses : numpyro.sample
        Samples of responses

    """
    # because this is a 2AFC task
    chance_correct = .5
    # first get rid of model dim, since we're not using it here.
    if scaling.shape[-1] > 1:
        raise Exception("Currently, response model can only handle 1 model"
                        " at a time!")
    scaling = scaling.squeeze(-1)
    if observed_responses is not None:
        observed_responses = observed_responses.squeeze(-1)
    # we have to do a weird hacky workaround if we only have one trial_type:
    # for some reason, we can't plate over it (we get a really strange error
    # when calling mcmc.run, which I think is a numpyro bug, though this
    # function itself can be called without a problem). So, instead, we squeeze
    # out trial_types (and add it back in later when we construct the
    # InferenceData object)
    dim_offset = 0
    if scaling.shape[-1] == 1:
        scaling = scaling.squeeze(-1)
        if observed_responses is not None:
            observed_responses = observed_responses.squeeze(-1)
        dim_offset = 1
        trial_type_plate = None
    else:
        trial_type_plate = numpyro.plate('trial_type', scaling.shape[-1], dim=-1)
    image_name_plate = numpyro.plate('image_name', scaling.shape[-(2-dim_offset)],
                                     dim=-(2-dim_offset))
    subject_name_plate = numpyro.plate('subject_name', scaling.shape[-(3-dim_offset)],
                                       dim=-(3-dim_offset))
    scaling_plate = numpyro.plate('scaling', scaling.shape[-(4-dim_offset)],
                                  dim=-(4-dim_offset))
    obs_nans = None
    if observed_responses is not None:
        # where's the missing data?
        obs_nans = jnp.isnan(observed_responses)
        trials_plate = numpyro.plate('trials', observed_responses.shape[0],
                                     dim=-(5-dim_offset))
    else:
        trials_plate = numpyro.plate('trials', 1, dim=-(5-dim_offset))

    # different priors for the two models
    if model == 'V1':
        # expected value of .25 for exponentiated version, from Freeman and
        # Simoncelli, 2011
        s0_mean = -1.38
    elif model == 'RGC':
        # expected value of .018 for exponentiated version, from Dacey, 1992
        s0_mean = -4

    def _sample(scaling, observed_responses, obs_nans):
        with subject_name_plate:
            lapse_rate = numpyro.sample('pi_l', dist.Beta(2, 50))
            with image_name_plate:
                # expected value of 5 for exponentiated version, which looks reasonable
                a0 = numpyro.sample('a0', dist.Normal(1.6, 1))
                # use the prior mean from above
                s0 = numpyro.sample('s0', dist.Normal(s0_mean, 1))
                # this is the value without the lapse rate
                prop_corr = proportion_correct_curve(scaling, jnp.exp(a0), jnp.exp(s0))
                # now with the lapse rate
                prob_corr = numpyro.deterministic('probability_correct',
                                                  ((1 - lapse_rate) * prop_corr +
                                                   lapse_rate * chance_correct))
                with trials_plate, scaling_plate:
                    if obs_nans is not None:
                        # expand this out, for broadcasting purposes
                        prob_corr_nan = jnp.expand_dims(prob_corr, 0).tile((observed_responses.shape[0],
                                                                            *([1]*prob_corr.ndim)))
                        # sample the responses...
                        imputed_responses = numpyro.sample(
                            'responses_imputed',
                            dist.Bernoulli(prob_corr_nan, validate_args=True).mask(False)
                        )
                        # ...and insert the imputed responses where there are
                        # NaNs in the observed data.
                        observed_responses = jnp.where(obs_nans,
                                                       imputed_responses,
                                                       observed_responses)
                    return numpyro.sample('responses', dist.Bernoulli(prob_corr,
                                                                      validate_args=True),
                                          obs=observed_responses)
    if trial_type_plate is not None:
        with trial_type_plate:
            return _sample(scaling, observed_responses, obs_nans)
    else:
        return _sample(scaling, observed_responses, obs_nans)


def assemble_dataset_from_expt_df(expt_df,
                                  dep_variables=['subject_name', 'image_name',
                                                 'trial_type', 'model']):
    """Create Dataset from expt_df.

    Creates the xarray Dataset necessary for using MCMC to fit psychophysical
    curve to the expt_df, as created by `analysis.create_experiment_df` and
    `analysis.add_response_info`.

    Parameters
    ----------
    expt_df : pd.DataFrame
        DataFrame containing the results of at least one session for at least
        one subject, as created by a combination of
        `analysis.create_experiment_df` and `analysis.add_response_info`, then
        concatenating them across runs (and maybe sessions / subjects).
    dep_variables : list, optional
        List of columns in expt_df to hold onto for investigation when fitting
        the curve (in addition to scaling).

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the response data, with coordinates properly
        labeled. The data variable 'observed_responses' holds the data (was
        called 'hit_or_miss_numeric' in expt_df).

    """
    expt_df = expt_df[dep_variables + ['scaling', 'hit_or_miss_numeric']]
    expt_df = expt_df.set_index(dep_variables + ['scaling'])
    for n in expt_df.index.unique():
        expt_df.loc[n, 'trials'] = np.arange(len(expt_df.loc[n]))
    dataset = expt_df.reset_index().set_index(['trials', 'scaling'] +
                                              dep_variables).to_xarray()
    dataset = dataset.rename({'hit_or_miss_numeric': 'observed_responses'})
    return dataset


def simulate_dataset(critical_scaling, proportionality_factor,
                     scaling=jnp.logspace(-1, -.3, num=8), num_trials=30,
                     num_subjects=1, num_images=1, trial_types=1,
                     proportionality_factor_noise=.2,
                     critical_scaling_noise=.3, seed=10):
    r"""Simulate a dataset to fit psychophysical curve to.

    Parameters
    ----------
    critical_scaling : float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study. Currently only handle single values
        for this (i.e., one curve)
    proportionality_factor : float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. Currently only handle single values
        for this (i.e., one curve)
    scaling : jnp.ndarray, optional
        The scaling values to test. Default corresponds roughly to V1 tested
        values.
    num_trials : int, optional
        The number of trials to have per scaling value.
    num_subjects : int, optional
        The number of subjects to simulate. Their parameter values will be
        sampled from a lognormal distribution with the appropriate means.
    num_images : int, optional
        The number of images to simulate. Like subjects, parameter values
        sampled from lognormal distributions with appropriate means (sampled
        from same distribution).
    trial_types : {1, 2, 3}, optional
        How many trial types to have. If 2, the second will have half the true
        parameter values of the first. If 3, the third trial type will have
        double the true parameter values fo the first.
    {proportionality_factor, critical_scaling}_noise : float, optional
        The noise on these distributions. Note that since they're sampled in
        log-space, these should be relatively small (the sampled paramters can
        get large quickly!) and you should examine the true values returned in
        the returned Dataset to make sure they look reasonable.
    seed : int, optional
        The seed to use.

    Returns
    -------
    simul : xarray.Dataset
        simulated dataset containing the response data, with coordinates
        properly labeled.

    """
    np.random.seed(seed)
    # grab two random integers to use for jax seeds -- jax seeds seem to be
    # "less random" than numpy's: if you use the same seed to sample from two
    # normal distributions (with different loc and same scale), the samples
    # will be the same amount above / below the mean
    s0_seed, a0_seed, obs_seed = np.random.randint(np.iinfo(np.uint32).max, size=3)
    # get this the right number of dimensions so we can broadcast correctly
    scaling = jnp.expand_dims(scaling, (-1, -2, -3, -4))
    a0 = np.log([proportionality_factor])
    s0 = np.log([critical_scaling])
    if trial_types > 1:
        a0 = np.concatenate([a0, np.log([proportionality_factor/2])])
        s0 = np.concatenate([s0, np.log([critical_scaling/2])])
    if trial_types > 2:
        a0 = np.concatenate([a0, np.log([proportionality_factor*2])])
        s0 = np.concatenate([s0, np.log([critical_scaling*2])])
    if trial_types > 3:
        raise Exception(f"trial_types must be one of {1, 2, 3}, but got {trial_types}!")
    # convert to jax.numpy and add a dimension for model
    a0 = jnp.expand_dims(jnp.asarray(a0), -1)
    s0 = jnp.expand_dims(jnp.asarray(s0), -1)
    # set up distributions
    a0 = dist.Normal(a0, proportionality_factor_noise)
    s0 = dist.Normal(s0, critical_scaling_noise)
    # sample appropriately
    a0 = jnp.exp(a0.sample(PRNGKey(a0_seed), (num_subjects, num_images)))
    s0 = jnp.exp(s0.sample(PRNGKey(s0_seed), (num_subjects, num_images)))
    prop_corr = proportion_correct_curve(scaling, a0, s0)
    obs = dist.Bernoulli(prop_corr).sample(PRNGKey(obs_seed), (num_trials,))
    dims = ('trials', 'scaling', 'subject_name', 'image_name', 'trial_type',
            'model')
    coords = {'scaling': scaling.squeeze(), 'trials': np.arange(num_trials),
              'subject_name': [f'sub-{s:02d}' for s in range(num_subjects)],
              'image_name': [f'image_{i:02d}' for i in range(num_images)],
              'trial_type': [f'trial_type_{i:02d}' for i in range(trial_types)],
              'model': [['simulated_V1', 'simulated_RGC'][i] for i in np.arange(1)]}
    return xarray.Dataset({'observed_responses': (dims, obs),
                           'true_proportionality_factor': (dims[2:], a0),
                           'true_critical_scaling': (dims[2:], s0)},
                          coords)


def _assign_inf_dims(samples_dict, dataset, dummy_dim=None):
    """Figure out the mapping between vars and coords.

    It's annoying to line up variables and coordinates. this does the best it
    can: if a variable has a coordinate, it will be in the same order as in the
    dataset and have the same number of values. So if dataset has coords
    (trials, scaling, subject_name, image_name, trial_type) with shape (30, 8,
    3, 2, 3), then variables could have coords (trials, scaling, image_name,
    subject_name) but not (trials, scaling, subject_name, image_name). In this
    example, if the variable was shape (30, 8, 3), we would not know if that
    corresponded to (trials, scaling, subject_name) or (trials, scaling,
    image_name) -- would assume the latter.

    """
    dims = {}
    if dummy_dim is not None and not hasattr(dummy_dim, '__iter__'):
        dummy_dim = [dummy_dim]
    if sum([dataset.dims['scaling'] == v for v in dataset.dims.values()]) > 1:
        # then we have something that's the same size as scaling and need to do
        # the complicated check below
        scaling_check = True
    else:
        scaling_check = False
    for k, v in samples_dict.items():
        var_shape = v.shape
        dims[k] = []
        i = 1
        for d in dataset.observed_responses.dims:
            if i >= len(var_shape):
                break
            if len(dataset.coords[d]) == var_shape[i]:
                # if scaling is the same shape as one of the other dims, can
                # sometimes mis-assign. but we know that if there's only one
                # thing that has the same shape as scaling, then it's not
                # scaling (parameters won't have scaling, responses or
                # probability correct will have scaling and the other coords)
                if d == 'scaling' and scaling_check and sum([len(dataset.coords[d]) == s for s in var_shape]) == 1:
                    continue
                dims[k] += [d]
                i += 1
            elif dummy_dim is not None:
                for dummy in dummy_dim:
                    if (i in [dummy, len(var_shape)+dummy] and var_shape[dummy] == 1):
                        dims[k] += ['dummy']
                        i += 1
    return dims


def _arrange_vars(dataset):
    """Get and reshape scaling and observed responses from dataset."""
    if dataset.observed_responses.dims[:2] != ('trials', 'scaling'):
        raise Exception("First two dimensions of observed responses must "
                        "be trials and scaling!")
    observed_responses = jnp.array(dataset.observed_responses.values,
                                   dtype=jnp.float32)
    scaling = jnp.array(dataset.scaling.values, dtype=jnp.float32)
    # get scaling into the appropriate shape -- scaling on the first dimension,
    # and then repeated to match the shape of observed_responses after that
    scaling = jnp.expand_dims(scaling, tuple(-(i+1) for i in
                                             range(observed_responses.ndim-2)))
    scaling = scaling.tile((1, *observed_responses.shape[2:]))
    return scaling, observed_responses


def run_inference(dataset, mcmc_model_type='partially-pooled', step_size=.1,
                  num_draws=1000, num_chains=1, num_warmup=500, seed=0,
                  target_accept_prob=.8, max_tree_depth=10, **nuts_kwargs):
    """Run MCMC inference for our response_model, conditioned on data.

    Uses NUTS sampler.

    For some reason, numpyro has trouble running in parallel, even with the
    suggested `numpyro.set_host_device_count(n)` -- I think the solution is to
    set the proper environmental variable yourself: `export
    XLA_FLAGS=--xla_force_host_platform_device_count=n`. Check
    `jax.lib.xla_bridge.device_count()` to see if it worked.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing observed_responses data variable and at least the
        coordinates trials and scaling (must be first two).
    step_size : float, optional
        Size of a single step.
    num_draws : int, optional
        Number of draws (samples in numpyro's terminology) in each chain. The
        higher the better.
    num_chains : int, optional
        The number of independent MCMC chains to run. The higher the better. If
        greater than 1, will use multiprocessing.
    num_warmup : int, optional
        The number of "warmup" steps to include in each chain. These are
        discarded.
    seed : int, optional
        RNG seed.
    nuts_kwargs :
        Passed to NUTS at initialization

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        The MCMC object that has run inference. Pass to assemble_inf_data.

    """
    if len(dataset.model) > 1:
        raise Exception("For now, can only handle one model at a time!")
    model = dataset.model.values[0].split('_')[0]
    if model == 'simulated':
        # then it's simulate_{actual_model_name}
        model = dataset.model.values[0].split('_')[1]
    if mcmc_model_type == 'partially-pooled':
        response_model = partially_pooled_response_model
    elif mcmc_model_type == 'unpooled':
        response_model = unpooled_response_model
    else:
        raise Exception(f"Don't know how to handle mcmc_model_type {mcmc_model_type}!")
    scaling, observed_responses = _arrange_vars(dataset)
    mcmc_kernel = numpyro.infer.NUTS(response_model,
                                     step_size=step_size,
                                     init_strategy=numpyro.infer.init_to_sample,
                                     target_accept_prob=target_accept_prob,
                                     max_tree_depth=max_tree_depth,
                                     **nuts_kwargs)
    # for now, progress bar doesn't show for multiple chains:
    # https://github.com/pyro-ppl/numpyro/issues/309
    mcmc = numpyro.infer.MCMC(mcmc_kernel, num_samples=num_draws,
                              num_chains=num_chains,
                              num_warmup=num_warmup, progress_bar=True)
    mcmc.run(PRNGKey(seed), scaling, model, observed_responses)
    return mcmc


def assemble_inf_data(mcmc, dataset, mcmc_model_type='partially-pooled',
                      seed=1):
    """Convert mcmc into properly-formatted inference data object.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        The MCMC object returned by `run_inference`
    dataset: xarray.Dataset
        Dataset containing observed_responses data variable and at least the
        coordinates trials and scaling (must be first two).
    seed : int, optional
        RNG seed.

    Returns
    -------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) containing the posterior,
        posterior_predictive, prior, prior_predictive, and observed_data.
-
    """
    if len(dataset.subject_name) == 1 or len(dataset.image_name) == 1:
        raise Exception("This will fail if subject_name or image_name only "
                        "has one value! We can handle only 1 trial_type, "
                        "but others must have multiple")
    if mcmc_model_type == 'partially-pooled':
        response_model = partially_pooled_response_model
    elif mcmc_model_type == 'unpooled':
        response_model = unpooled_response_model
    else:
        raise Exception(f"Don't know how to handle mcmc_model_type {mcmc_model_type}!")
    scaling, obs = _arrange_vars(dataset)
    model = dataset.model.values[0].split('_')[0]
    if model == 'simulated':
        # then it's simulate_{actual_model_name}
        model = dataset.model.values[0].split('_')[1]
    # different dummy dimensions when there was a single trial_type (which gets
    # squeezed out in the response model) vs. more than one.
    if scaling.shape[-2] == 1:
        dummy_dims = [[-1, 1, 2], [1, 2], -1]
    else:
        dummy_dims = [[-2, 1], 1, -2]
    n_total_samples = list(mcmc.get_samples().values())[0].shape[0]
    prior = numpyro.infer.Predictive(response_model,
                                     num_samples=n_total_samples)
    posterior_pred = numpyro.infer.Predictive(response_model,
                                              posterior_samples=mcmc.get_samples())
    # need to create each of these separately because they have different
    # coords
    prior = prior(PRNGKey(seed), scaling, model)
    # the subject-level variables have a dummy dimension at the same place as
    # the image_name dimension, in order to allow broadcasting. we allow it
    # here, and then drop it later
    prior_dims = _assign_inf_dims(prior, dataset, dummy_dim=dummy_dims[0])
    prior = az.from_numpyro(prior=prior, coords=dataset.coords, dims=prior_dims)
    posterior_pred = posterior_pred(PRNGKey(seed+1), scaling, model)
    post_dims = _assign_inf_dims(posterior_pred, dataset,
                                 dummy_dim=dummy_dims[1])
    posterior_pred = az.from_numpyro(posterior_predictive=posterior_pred,
                                     coords=dataset.coords, dims=post_dims)
    # the observed data will have a trials dim first
    post_dims['responses'][0] = 'trials'
    # in this case, there was only one trial_type, and the shape of the
    # responses from the posterior predictive has an extra dummy dimension for
    # some reason. but I think it doesn't happen when there's missing (and thus
    # imputed) data for some reason?
    if obs.shape[-2] == 1 and post_dims['responses'].count('dummy') > 1:
        post_dims['responses'].pop(1)
    # the subject-level variables these have a dummy dimension at the same
    # place as the image_name dimension, in order to allow broadcasting. we
    # allow it here, and then drop it later
    variable_dims = _assign_inf_dims(mcmc.get_samples(), dataset,
                                     dummy_dim=dummy_dims[2])
    variable_dims.update(post_dims)
    # if there was missing data, it will need to the imputation in order to
    # compute log-likelihood, so we need the seed handler
    with numpyro.handlers.seed(rng_seed=seed+2):
        inf_data = (az.from_numpyro(mcmc, coords=dataset.coords, dims=variable_dims) +
                    prior + posterior_pred)
    # because of how we create it, prior predictive gets mushed up with prior
    # -- we move that around here.
    inf_data.add_groups({'prior_predictive':
                         inf_data.prior[['responses', 'probability_correct']]})
    inf_data.prior = inf_data.prior.drop_vars(['responses', 'probability_correct',
                                               'scaling'])
    # drop the dummy dimension -- it gets broadcasted out but filled with NaNs.
    # also the probability_correct shouldn't be in the posterior
    inf_data.posterior = inf_data.posterior.dropna('dummy').squeeze('dummy', True).drop(['probability_correct', 'scaling'])
    inf_data.prior = inf_data.prior.dropna('dummy').squeeze('dummy', True)
    # for the predictives, there's no NaNs in the dummy dimensions, they're
    # just extra single dimensions, so we can squeeze them right out
    inf_data.posterior_predictive = inf_data.posterior_predictive.squeeze('dummy', True)
    inf_data.prior_predictive = inf_data.prior_predictive.squeeze('dummy', True)
    # then there was missing data, so we imputed the responses
    if np.isnan(dataset.observed_responses).any():
        inf_data.observed_data = inf_data.observed_data.rename({'responses': 'imputed_responses'})
        # if everything else had model squeezed out of it, want to squeeze
        # it out here too.
        try:
            responses = dataset.observed_responses.squeeze('model')
        except ValueError:
            responses = dataset.observed_responses
        # if everything else had trial_type squeezed out of it, want to squeeze
        # it out here too.
        try:
            inf_data.observed_data['responses'] = responses.squeeze('trial_type')
        except ValueError:
            inf_data.observed_data['responses'] = responses
    # this gets weirdly shaped, so just remove it
    del inf_data.log_likelihood
    if obs.shape[-2] == 1:
        # then there was only one trial type and it was squeezed out during
        # fitting -- add it in now.
        inf_data.posterior = inf_data.posterior.expand_dims('trial_type', -1).assign_coords({'trial_type': dataset.trial_type})
        inf_data.posterior_predictive = inf_data.posterior_predictive.expand_dims('trial_type', -1).assign_coords({'trial_type': dataset.trial_type})
        inf_data.prior = inf_data.prior.expand_dims('trial_type', -1).assign_coords({'trial_type': dataset.trial_type})
        inf_data.prior_predictive = inf_data.prior_predictive.expand_dims('trial_type', -1).assign_coords({'trial_type': dataset.trial_type})
        inf_data.observed_data = inf_data.observed_data.expand_dims('trial_type', -1).assign_coords({'trial_type': dataset.trial_type})
    if obs.shape[-1] == 1:
        # then there was only one model and it was squeezed out during fitting
        # -- add it in now.
        inf_data.posterior = inf_data.posterior.expand_dims('model', -1).assign_coords({'model': dataset.model})
        inf_data.posterior_predictive = inf_data.posterior_predictive.expand_dims('model', -1).assign_coords({'model': dataset.model})
        inf_data.prior = inf_data.prior.expand_dims('model', -1).assign_coords({'model': dataset.model})
        inf_data.prior_predictive = inf_data.prior_predictive.expand_dims('model', -1).assign_coords({'model': dataset.model})
        inf_data.observed_data = inf_data.observed_data.expand_dims('model', -1).assign_coords({'model': dataset.model})
    inf_data.add_groups({'metadata': {'mcmc_model_type': mcmc_model_type}})
    if mcmc_model_type == 'unpooled':
        # in this case, we want to drop all the values where we had no
        # information -- since the model is unpooled, the parameters we
        # inferred for those missing data points is just the prior. this is
        # confusing, and so we just mask them out.
        nan_mask = inf_data.observed_data.responses.mean(('trials', 'scaling')).notnull()
        inf_data.posterior = inf_data.posterior.where(nan_mask)
        inf_data.posterior_predictive = inf_data.posterior_predictive.where(nan_mask)
    return inf_data


def _compute_hdi(tmp, hdi):
    """Compute the HDI of a variable.

    hdi must lie in [0, 1]
    """
    hdi_xr = az.hdi(tmp, hdi)
    hdi_xr['hdi'] = 100 * np.array([.5-hdi/2, .5+hdi/2])
    tmp = tmp.assign_coords({'hdi': 50})
    tmp = tmp.median(['chain', 'draw']).expand_dims('hdi')
    if isinstance(tmp, xarray.DataArray):
        hdi_xr = hdi_xr[tmp.name]
    return xarray.concat([hdi_xr, tmp], 'hdi')


def inf_data_to_df(inf_data, kind='predictive', query_str=None, hdi=False):
    """Convert inf_data to a dataframe, for plotting.

    We exponentiate the log_s0, log_a0, log_s0_global_mean, and
    log_a0_global_mean variables (but not the global_sd variables).

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `assemble_inf_data`.
    kind : {'predictive', 'parameters', 'psychophysical curve parameters'}, optional
        Whether to create df containing predictive info (responses and
        probability_correct), model parameter info, or psychophysical curve
        parameters (where we've combined across the different effects and
        exponentiated them in order to get the critical scaling and
        proportionality factor values for each separate image, subject, and
        trial type).
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'").
    hdi : bool or float, optional
        Whether to compute the HDI (highest density interval) on the parameters
        or return the full distributions. If True, we compute the 95% HDI, if a
        float, must lie in (0, 1] and give the percentage HDI (we also include
        the median). The HDI is one way of constructing a summary credible
        interval, the other common way is to use the equal-tailed interval
        (ETI), where a 95% ETI has 2.5% of the distribution on either side of
        its limits (so it goes from the 2.5th to 97.5th percentile). The 95%
        HDI, on the other hand, contains the central 95% with the highest
        probability density; with symmetric distirbutions, this will be the
        same as ETI. See
        https://www.sciencedirect.com/topics/mathematics/highest-density-interval
        for some more discussion, excerpted from [2]_

    To quote

    Returns
    -------
    df : pd.DataFrame
        The DataFrame described above

    References
    ----------
    .. [2] Kruschke, J. K. (2015). Doing Bayesian Data Analysis. : Elsevier.

    """
    if hdi is True:
        hdi = .95
    if kind == 'predictive':
        dists = ['observed_data', 'posterior_predictive', 'prior_predictive']
        df = []
        for d in dists:
            tmp = inf_data[d]
            # doesn't make sense to compute the HDI for observed data.
            if hdi and d != 'observed_data':
                tmp = _compute_hdi(tmp, hdi)
            tmp = tmp.to_dataframe().reset_index()
            tmp['distribution'] = d
            df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
    elif kind == 'parameters':
        dists = ['prior', 'posterior']
        df = []
        for d in dists:
            tmp = inf_data[d]
            if hdi:
                tmp = _compute_hdi(tmp, hdi)
            tmp = tmp.to_dataframe()
            for c in tmp.columns:
                if c.startswith('log') and 'sd' not in c:
                    tmp[c.replace('log_', '')] = tmp[c].map(np.exp)
            tmp = tmp.reset_index().melt(tmp.index.names)
            tmp['distribution'] = d
            df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
        vars = [v for v in df.variable.unique() if 'log' not in v
                or 'sd' in v]
        df = df.query("variable in @vars")
    elif kind == 'psychophysical curve parameters':
        dists = ['prior', 'posterior']
        params = ['a0', 's0']
        df = []
        for d in dists:
            for p in params:
                try:
                    tmp = np.exp(inf_data[d][f'log_{p}_global_mean'] + inf_data[d][f'log_{p}_image'] +
                                 inf_data[d][f'log_{p}_subject'])
                except KeyError:
                    # then this is the unpooled version, and so we can directly
                    # grab the parameter
                    tmp = np.exp(inf_data[d][p])
                if hdi:
                    tmp = _compute_hdi(tmp, hdi)
                tmp = tmp.to_dataframe('value').reset_index()
                tmp['distribution'] = d
                tmp['parameter'] = p
                df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
    elif kind == 'parameter grouplevel means':
        dists = ['prior', 'posterior']
        params = ['a0', 's0']
        mean_level = ['subject_name', 'image_name']
        df = []
        for d in dists:
            for p in params:
                for m, other_m in zip(mean_level+[('subject_name', 'image_name')],
                                      mean_level[::-1]+['all']):
                    try:
                        tmp = np.exp(inf_data[d][f'log_{p}_global_mean'] + inf_data[d][f'log_{p}_image'] +
                                     inf_data[d][f'log_{p}_subject'])
                    except KeyError:
                        # then this is the unpooled version, and so we can directly
                        # grab the parameter
                        tmp = np.exp(inf_data[d][p])
                    tmp = tmp.mean(m)
                    if hdi:
                        tmp = _compute_hdi(tmp, hdi)
                    tmp = tmp.to_dataframe('value').reset_index()
                    tmp['distribution'] = d
                    tmp['parameter'] = p
                    tmp['level'] = other_m
                    if other_m == 'all':
                        dep_var = 'all subjects, all images'
                    else:
                        dep_var = tmp[other_m]
                        tmp = tmp.drop(columns=[other_m])
                    tmp['dependent_var'] = dep_var
                    df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
        df.dependent_var = df.dependent_var.map(lambda x: x.split('_')[0])
    if query_str is not None:
        df = df.query(query_str)
    if 'image_name' in df.columns:
        # clean up the plots by removing this redundant text
        df.image_name = df.image_name.map(lambda x: x.replace('_range-.05,.95_size-2048,2600', ''))
    metadata = inf_data.metadata.to_dataframe().reset_index(drop=True)
    if len(metadata) > 1:
        raise Exception("Don't know how to handle metadata with multiple values!")
    metadata = {k: v[0] for k, v in metadata.to_dict().items()}
    df = df.assign(**metadata)
    return df
