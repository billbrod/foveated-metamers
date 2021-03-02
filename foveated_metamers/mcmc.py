#!/usr/bin/env python3
"""Code for using MCMC to fit psychophysical data."""
import pyro
import xarray
import numpy as np
import pyro.distributions as dist
import torch
import arviz as az
from . import curve_fit


def response_model(scaling, model='V1'):
    """Probabilistic model of responses, with lapse rate.

    Priors:
    - lapse_rate: Beta(2, 50)
    - critical_scaling: Beta(5,25) for V1, Beta(2, 70) for RGC
    - proportionality_factor: Exponential(.2)

    Parameters
    ----------
    scaling : torch.Tensor
        1d, scaling value(s) to calculate discriminability for.
    model : {'V1', 'RGC'}
        Whether we should use V1 or RGC prior for critical_scaling.

    Returns
    -------
    responses : pyro.sample
        Samples of responses

    """
    lapse_rate = pyro.sample('pi_l', dist.Beta(2, 50))
    # different priors for the two models
    if model == 'V1':
        critical_scaling = pyro.sample('s_0', dist.Beta(5, 25))
    elif model == 'RGC':
        critical_scaling = pyro.sample('s_0', dist.Beta(2, 70))
    # expected value of 5. don't really have a good reason to do anything for this, but it can't be negative
    proportionality_factor = pyro.sample('a_0', dist.Exponential(.2))
    # because this is a 2AFC task
    chance_correct = .5
    # this is the value without the lapse rate
    prop_corr = curve_fit.proportion_correct_curve(scaling, proportionality_factor, critical_scaling)
    # now with the lapse rate
    prob_corr = pyro.deterministic('probability_correct',
                                   ((1 - lapse_rate) * prop_corr +
                                    lapse_rate * chance_correct))
    with pyro.plate('subject_name', 3):
        return pyro.sample('responses', dist.Bernoulli(prob_corr, validate_args=True))


def assemble_dataset_from_expt_df(expt_df,
                                  dep_variables=['image_name', 'trial_type',
                                                 'subject_name']):
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
                     scaling=torch.logspace(-1, -.3, steps=8),
                     num_trials=30):
    r"""Simulate a dataset to fit psychophysical curve to.

    Parameters
    ----------
    proportionality_factor : float
        The "gain" of the curve, determines how quickly it rises, parameter
        $\alpha_0$ in [1]_, equation 17. Currently only handle single values
        for this (i.e., one curve)
    critical_scaling : float
        The "threshold" of the curve, the scaling value at which
        discriminability falls to 0 and thus performance falls to chance,
        parameter $s_0$ in [1]_, equation 17. This should be more consistent,
        and is the focus of this study. Currently only handle single values
        for this (i.e., one curve)
    scaling : torch.tensor, optional
        The scaling values to test. Default corresponds roughly to V1 tested
        values.
    num_trials : int, optional
        The number of trials to have per scaling value.

    Returns
    -------
    simul : xarray.Dataset

        simulated dataset containing the response data, with coordinates
        properly labeled.

    """
    prop_corr = curve_fit.proportion_correct_curve(scaling,
                                                   proportionality_factor,
                                                   critical_scaling)
    obs = torch.distributions.Bernoulli(prop_corr).sample((num_trials,))
    dims = ('trials', 'scaling', 'image_name',
            'trial_type', 'subject_name')
    while obs.ndim < len(dims):
        # add an extra dimension for each of these
        obs = obs.unsqueeze(-1)
    coords = {'scaling': scaling.numpy(), 'trials': np.arange(num_trials),
              'subject_name': [1], 'image_name': ['simulated'],
              'trial_type': ['simulated']}
    return xarray.Dataset({'observed_responses': (dims, obs.numpy())}, coords)


def _assign_inf_dims(samples_dict, dataset,
                     vars=['responses', 'probability_correct']):
    """Figure out the mapping between vars and coords.

    It's annoying to line up variables and coordinates. this does the best it
    can: if a variable has a coordinate, it will be in the same order as in the
    dataset and have the same number of values. So if dataset has coords
    (trials, scaling, image_name, trial_type, subject_name) with shape (30, 8,
    3, 2, 3), then variables could have coords (trials, scaling, image_name,
    subject_name) but not (trials, scaling, subject_name, image_name). In this
    example, if the variable was shape (30, 8, 3), we would not know if that
    corresponded to (trials, scaling, subject_name) or (trials, scaling,
    image_name) -- would assume the latter.

    """
    dims = {}
    for v in vars:
        var_shape = samples_dict[v].shape
        dims[v] = []
        i = 1
        for d in dataset.observed_responses.dims:
            if len(dataset.coords[d]) == var_shape[i]:
                dims[v] += [d]
                i += 1
            if i >= len(var_shape):
                break
    return dims


def run_inference(dataset, model='V1', step_size=.1,
                  num_draws=1000, num_chains=1, warmup_steps=500,
                  **nuts_kwargs):
    """Run MCMC inference for our response_model, conditioned on data.

    Uses NUTS sampler.

    This currently doesn't work on GPU, see
    https://github.com/arviz-devs/arviz/issues/1529. However GPU also appears
    to be about 2x slower than CPU (at least on my laptop), which is weird.

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
        Number of draws (samples in pyro's terminology) in each chain. The
        higher the better.
    num_chains : int, optional
        The number of independent MCMC chains to run. The higher the better. If
        greater than 1, will use multiprocessing.
    warmup_steps : int, optional
        The number of "warmup" steps to include in each chain. These are
        discarded.
    nuts_kwargs :
        Passed to NUTS at initialization

    Returns
    -------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) containing the posterior,
        posterior_predictive, prior, prior_predictive, and observed_data.

    """
    if dataset.observed_responses.dims[:2] != ('trials', 'scaling'):
        raise Exception("First two dimensions of observed responses must "
                        "be trials and scaling!")
    observed_responses = torch.tensor(dataset.observed_responses.values,
                                      dtype=torch.float32)
    scaling = torch.tensor(dataset.scaling.values, dtype=torch.float32)
    # get scaling into the appropriate shape -- scaling on the first dimension,
    # and then repeated to match the shape of observed_responses after that
    scaling = scaling.expand(*([1]*(observed_responses.ndim-2)), len(scaling))
    scaling = scaling.transpose(-1, 0).repeat(1, *observed_responses.shape[2:])
    conditioned_responses = pyro.condition(response_model,
                                           data={'responses': observed_responses})
    mcmc_kernel = pyro.infer.NUTS(conditioned_responses, step_size=step_size,
                                  **nuts_kwargs)
    mcmc = pyro.infer.MCMC(mcmc_kernel, num_samples=num_draws,
                           num_chains=num_chains, mp_context='spawn',
                           warmup_steps=warmup_steps)
    mcmc.run(scaling, model)
    prior = pyro.infer.Predictive(response_model,
                                  num_samples=num_draws*num_chains)
    posterior_pred = pyro.infer.Predictive(response_model,
                                           posterior_samples=mcmc.get_samples())
    # need to create each of these separately because they have different coords
    prior = prior(scaling, 'V1')
    # for some reason, prior squeezes everything out when creating inf data
    prior_dims = _assign_inf_dims({k: v.squeeze() for k, v in prior.items()},
                                  dataset)
    prior = az.from_pyro(prior=prior, coords=dataset.coords, dims=prior_dims)
    posterior_pred = posterior_pred(scaling, 'V1')
    # for some reason, this has a weird dummy dimension. this removes that
    # (assuming it's only shape 1; else I should probably be aware of it)
    posterior_pred['probability_correct'] = posterior_pred['probability_correct'].squeeze(1)
    post_dims = _assign_inf_dims(posterior_pred, dataset)
    posterior_pred = az.from_pyro(posterior_predictive=posterior_pred,
                                  coords=dataset.coords, dims=post_dims)
    # the observed data will have a trials dim first
    post_dims['responses'].insert(0, 'trials')
    inf_data = (az.from_pyro(mcmc, coords=dataset.coords, dims=post_dims) +
                prior + posterior_pred)
    # because of how we create it, prior predictive gets mushed up with prior
    # -- we move that around here.
    inf_data.add_groups({'prior_predictive':
                         inf_data.prior[['responses', 'probability_correct']]})
    inf_data.prior = inf_data.prior.drop_vars(['responses', 'probability_correct',
                                               'scaling'])
    # drop probability_correct from observed data because it doesn't make sense
    # -- the probability correct in our response_model has lapse_chance (one of
    # our parameters) built in and so doesn't make sense to talk about
    # observing it. to put it another way: the responses are our only
    # observations and we need to infer the probability correct
    inf_data.observed_data = inf_data.observed_data.drop_vars('probability_correct')
    return inf_data


def inf_data_to_df(inf_data, kind='predictive', jitter_scaling=False):
    """Convert inf_data to a dataframe, for plotting.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `run_inference`.
    kind : {'predictive', 'parameters'}, optional
        Whether to create df containing predictive info (responses and
        probability_correct) or model parameter info
    jitter_scaling : bool or float, optional
        If not False, we jitter scaling values (so they don't get plotted on
        top of each other). If True, we jitter by 5e-3, else, the amount to
        jitter by. Will need to rework this for log axis.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame described above

    """
    if kind == 'predictive':
        df = inf_data.observed_data.to_dataframe().reset_index()
        df['distribution'] = 'observed_data'
        tmp = inf_data.posterior_predictive.to_dataframe().reset_index()
        tmp['distribution'] = 'posterior_predictive'
        df = df.append(tmp).reset_index(drop=True)
        tmp = inf_data.prior_predictive.to_dataframe().reset_index()
        tmp['distribution'] = 'prior_predictive'
        df = df.append(tmp).reset_index(drop=True)
    elif kind == 'parameters':
        df = inf_data.prior.to_dataframe().reset_index().melt(['chain', 'draw'])
        df['distribution'] = 'prior'
        tmp = inf_data.posterior.to_dataframe().reset_index().melt(['chain', 'draw'])
        tmp['distribution'] = 'posterior'
        df = df.append(tmp).reset_index(drop=True)
    if jitter_scaling:
        if jitter_scaling is True:
            jitter_scaling = 5e-3
        df = df.set_index('distribution')
        for i, v in enumerate(df.index.unique()):
            df.loc[v, 'scaling'] += i*jitter_scaling
        df = df.reset_index()
    return df
