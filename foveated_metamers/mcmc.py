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
    return vals.clip(a_min=0)


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


def response_model(scaling, model='V1'):
    r"""Probabilistic model of responses, with lapse rate.

    - Critical scaling ($s_0$) and proportionality factor / gain ($\alpha_0$)
      are both modeled on a natural log-scale, where they're the sum of
      trial_type, image_name (crossed with trial_type), and subject_name
      (crossed with trial_type) effects. Inspired by Wallis et al, 2019.

    - Lapse rate is modeled as a completely unpooled manner, independently for
      each (trial_type, subject) (shared across images)

    Model Parameters:

    - log_a0_global_mean, log_s0_global_mean: independent for each trial_type,
      this is the high-level mean for the gain and critical scaling.

    - log_a0_subject_sd, log_s0_subject_sd: independent for each trial type,
      this is the noise for the subject-level effects for the gain and critical
      scaling.

    - log_a0_image_sd, log_s0_image_sd: independent for each trial type,
      this is the noise for the image-level effects for the gain and critical
      scaling.

    - log_a0_image, log_s0_image: partially pooled within trial_type across
      images, this is the image-level effect.

    - log_a0_subject, log_s0_subject: partially pooled within trial_type across
      subjects, this is the subject-level effect.

    - pi_l: lapse rate, independent for each (trial_type, subject).

    NOTE: no interaction between subject and image effects!

    Priors:

    - pi_l: Beta(2, 50)

    - log_a0_global_mean: Normal(1.6, 1), this gives expected value of 5 for
      the exponentiated version.

    - log_s0_global_mean: Normal(-1.38, 1) for V1 (expected value ~.25 of
      exponentiated version, following from Freeman and Simoncelli, 2011);
      Normal(-4, 1) for RGC (expected value ~.018 of exponentiated version,
      from Dacey, 1992)

    - log_a0_subject_sd, log_s0_subject_sd, log_a0_image_sd, log_s0_image_sd:
      HalfCauchy(.1)

    - log_a0_image: Normal(0, a0_image_sd)

    - log_s0_image: Normal(0, s0_image_sd)

    - log_a0_subject: Normal(0, a0_subject_sd)

    - log_s0_subject: Normal(0, s0_subject_sd)

    Parameters
    ----------
    scaling : jnp.ndarray
        1d, scaling value(s) to calculate discriminability for.
    model : {'V1', 'RGC'}
        Whether we should use V1 or RGC prior for critical_scaling.

    Returns
    -------
    responses : numpyro.sample
        Samples of responses

    """
    # because this is a 2AFC task
    chance_correct = .5
    trial_type_plate = numpyro.plate('trial_type', scaling.shape[-1], dim=-1)
    image_name_plate = numpyro.plate('image_name', scaling.shape[-2], dim=-2)
    subject_name_plate = numpyro.plate('subject_name', scaling.shape[-3], dim=-3)
    with trial_type_plate:
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
        s0_subject_sd = numpyro.sample('log_s0_subject_sd', dist.HalfCauchy(.1))
        a0_subject_sd = numpyro.sample('log_a0_subject_sd', dist.HalfCauchy(.1))
        s0_image_sd = numpyro.sample('log_s0_image_sd', dist.HalfCauchy(.1))
        a0_image_sd = numpyro.sample('log_a0_image_sd', dist.HalfCauchy(.1))
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
                return numpyro.sample('responses', dist.Bernoulli(prob_corr,
                                                                  validate_args=True))


def assemble_dataset_from_expt_df(expt_df,
                                  dep_variables=['subject_name', 'image_name',
                                                 'trial_type']):
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
                     critical_scaling_noise=.3, seed=0):
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
    trial_types : {1, 2}, optional
        How many trial types to have. If 2, one will have half the true
        parameter values of the other.
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
    scaling = jnp.expand_dims(scaling, (-1, -2, -3))
    a0 = np.log([proportionality_factor])
    s0 = np.log([critical_scaling])
    if trial_types == 2:
        a0 = np.concatenate([a0, np.log([proportionality_factor/2])])
        s0 = np.concatenate([s0, np.log([critical_scaling/2])])
    # convert to jax.numpy
    a0 = jnp.asarray(a0)
    s0 = jnp.asarray(s0)
    # set up distributions
    a0 = dist.Normal(a0, proportionality_factor_noise)
    s0 = dist.Normal(s0, critical_scaling_noise)
    # sample appropriately
    a0 = jnp.exp(a0.sample(PRNGKey(a0_seed), (num_subjects, num_images)))
    s0 = jnp.exp(s0.sample(PRNGKey(s0_seed), (num_subjects, num_images)))
    prop_corr = proportion_correct_curve(scaling, a0, s0)
    obs = dist.Bernoulli(prop_corr).sample(PRNGKey(obs_seed), (num_trials,))
    dims = ('trials', 'scaling', 'subject_name', 'image_name', 'trial_type')
    coords = {'scaling': scaling.squeeze(), 'trials': np.arange(num_trials),
              'subject_name': [f'sub-{s:02d}' for s in range(num_subjects)],
              'image_name': [f'image_{i:02d}' for i in range(num_images)],
              'trial_type': [f'trial_type_{i:02d}' for i in range(trial_types)]}
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
    for k, v in samples_dict.items():
        var_shape = v.shape
        dims[k] = []
        i = 1
        for d in dataset.observed_responses.dims:
            if i >= len(var_shape):
                break
            if len(dataset.coords[d]) == var_shape[i]:
                dims[k] += [d]
                i += 1
            elif (dummy_dim is not None and
                  i in [dummy_dim, len(var_shape)+dummy_dim] and
                  var_shape[dummy_dim] == 1):
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


def run_inference(dataset, model='V1', step_size=.1, num_draws=1000,
                  num_chains=1, num_warmup=500, seed=0, **nuts_kwargs):
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
    model : {'V1', 'RGC'}
        Whether we should use V1 or RGC prior for critical_scaling.
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
    scaling, observed_responses = _arrange_vars(dataset)
    conditioned_responses = numpyro.handlers.condition(response_model,
                                                       data={'responses': observed_responses})
    mcmc_kernel = numpyro.infer.NUTS(conditioned_responses, step_size=step_size,
                                     # jit_compile=True, ignore_jit_warnings=True,
                                     **nuts_kwargs)
    # for now, progress bar doesn't show for multiple chains:
    # https://github.com/pyro-ppl/numpyro/issues/309
    mcmc = numpyro.infer.MCMC(mcmc_kernel, num_samples=num_draws,
                              num_chains=num_chains,
                              num_warmup=num_warmup, progress_bar=True)
    mcmc.run(PRNGKey(seed), scaling, model)
    return mcmc


def assemble_inf_data(mcmc, dataset, seed=0):
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
    seed = PRNGKey(seed)
    scaling, _ = _arrange_vars(dataset)
    n_total_samples = list(mcmc.get_samples().values())[0].shape[0]
    prior = numpyro.infer.Predictive(response_model, num_samples=n_total_samples)
    posterior_pred = numpyro.infer.Predictive(response_model,
                                              posterior_samples=mcmc.get_samples())
    # need to create each of these separately because they have different
    # coords
    prior = prior(seed, scaling, 'V1')
    # the subject-level variables have a dummy dimension at the same place as
    # the image_name dimension, in order to allow broadcasting. we allow it
    # here, and then drop it later
    prior_dims = _assign_inf_dims(prior, dataset, dummy_dim=-2)
    prior = az.from_numpyro(prior=prior, coords=dataset.coords, dims=prior_dims)
    posterior_pred = posterior_pred(seed, scaling, 'V1')
    post_dims = _assign_inf_dims(posterior_pred, dataset)
    posterior_pred = az.from_numpyro(posterior_predictive=posterior_pred,
                                     coords=dataset.coords, dims=post_dims)
    # the observed data will have a trials dim first
    post_dims['responses'].insert(0, 'trials')
    # the subject-level variables these have a dummy dimension at the same
    # place as the image_name dimension, in order to allow broadcasting. we
    # allow it here, and then drop it later
    variable_dims = _assign_inf_dims(mcmc.get_samples(), dataset, dummy_dim=-2)
    variable_dims.update(post_dims)
    inf_data = (az.from_numpyro(mcmc, coords=dataset.coords, dims=variable_dims) +
                prior + posterior_pred)
    # because of how we create it, prior predictive gets mushed up with prior
    # -- we move that around here.
    inf_data.add_groups({'prior_predictive':
                         inf_data.prior[['responses', 'probability_correct']]})
    inf_data.prior = inf_data.prior.drop_vars(['responses', 'probability_correct',
                                               'scaling'])
    # drop the dummy dimension -- it gets broadcasted out but filled with NaNs.
    inf_data.posterior = inf_data.posterior.dropna('dummy').squeeze('dummy').drop('dummy')
    inf_data.prior = inf_data.prior.dropna('dummy').squeeze('dummy').drop('dummy')
    return inf_data


def inf_data_to_df(inf_data, kind='predictive', jitter_scaling=False, query_str=None):
    """Convert inf_data to a dataframe, for plotting.

    We exponentiate the log_s0, log_a0, log_s0_global_mean, and
    log_a0_global_mean variables (but not the global_sd variables).

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `assemble_inf_data`.
    kind : {'predictive', 'parameters'}, optional
        Whether to create df containing predictive info (responses and
        probability_correct) or model parameter info
    jitter_scaling : bool or float, optional
        If not False, we jitter scaling values (so they don't get plotted on
        top of each other). If True, we jitter by 5e-3, else, the amount to
        jitter by. Will need to rework this for log axis.
    query_str : str or None, optional
        If not None, the string to query dataframe with to limit the plotted
        data (e.g., "distribution == 'posterior'").

    Returns
    -------
    df : pd.DataFrame
        The DataFrame described above

    """
    if kind == 'predictive':
        dists = ['observed_data', 'posterior_predictive', 'prior_predictive']
        df = []
        for d in dists:
            tmp = inf_data[d].to_dataframe().reset_index()
            tmp['distribution'] = d
            df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
    elif kind == 'parameters':
        dists = ['prior', 'posterior']
        df = []
        for d in dists:
            tmp = inf_data[d].to_dataframe()
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
    if jitter_scaling:
        if jitter_scaling is True:
            jitter_scaling = 5e-3
        df = df.set_index('distribution')
        for i, v in enumerate(df.index.unique()):
            df.loc[v, 'scaling'] += i*jitter_scaling
        df = df.reset_index()
    if query_str is not None:
        df = df.query(query_str)
    if 'image_name' in df.columns:
        # clean up the plots by removing this redundant text
        df.image_name = df.image_name.map(lambda x: x.replace('_range-.05,.95_size-2048,2600', ''))
    return df
