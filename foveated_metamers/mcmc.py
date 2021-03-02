#!/usr/bin/env python3
import pyro
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
    # Beta(5, 25) for V1, Beta(2, 70) for RGC
    critical_scaling = pyro.sample('s_0', dist.Beta(5, 25))
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
    return pyro.sample('responses', dist.Bernoulli(prob_corr, validate_args=True))


def _add_coords(inf_data, scaling, trials):
    """Add coordinates to inf_data

    Modifies inf_data in place to add scaling and trials coordinates to
    probability_correct and responses data.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        InferenceData object with posterior_predictive, prior_predictive, and
        observed_data Datasets.
    scaling, trials : torch.Tensor
        The tensors containing data to add as scaling and trial coordinates.

    """
    def _add_coords_helper(dataset):
        # this hacky approach is modified from from
        # https://stackoverflow.com/a/49916326/4659293
        dims = ['probability_correct_dim_0', 'responses_dim_0', 'responses_dim_1']
        dims = [d for d in dims if d in dataset]
        remap_dims = {}
        for d in dims:
            if dataset[d].shape == scaling.shape:
                remap_dims[d] = 'scaling'
            else:
                remap_dims[d] = 'trials'
        new_coords = {name: val.numpy() for name, val in zip(['scaling', 'trials'],
                                                             [scaling, trials])}
        dataset = dataset.reset_index(dims, drop=True)
        dataset = dataset.assign_coords({k: new_coords[k] for k in set(remap_dims.values())})
        for col in ['probability_correct', 'responses']:
            restricted_remap_dims = {k: remap_dims[k] for k in remap_dims if col in k}
            dataset[col] = dataset[col].rename(restricted_remap_dims)
        return dataset
    # need to use the inf_data.col syntax (instead fo inf_data['col']), so
    # can't loop through this and need to do each individually
    inf_data.posterior_predictive = _add_coords_helper(inf_data.posterior_predictive)
    inf_data.prior_predictive = _add_coords_helper(inf_data.prior_predictive)
    inf_data.observed_data = _add_coords_helper(inf_data.observed_data)
    # this modifies it in place, so don't need to return anything


def run_inference(scaling, observed_responses, model='V1', step_size=.1, num_draws=1000,
                  num_chains=1, warmup_steps=500, **nuts_kwargs):
    """Run MCMC inference for our response_model, conditioned on data.

    Uses NUTS sampler.

    Parameters
    ----------
    scaling : torch.Tensor
        1d, scaling value(s) to calculate discriminability for.
    observed_responses : torch.Tensor
        Tensor of shape (num_trials, len(scaling)). The observed responses,
        which we'll use to condition on our model.
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
    inf_data = az.from_pyro(mcmc,
                            prior=prior(scaling, model),
                            posterior_predictive=posterior_pred(scaling, model))
    # this changes inf_data in place, so don't need to grab it
    _add_coords(inf_data, scaling, torch.arange(observed_responses.shape[0],))
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
