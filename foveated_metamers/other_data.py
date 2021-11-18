#!/usr/bin/env python3

"""Code that makes use of other people's data.
"""
import arviz as az
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from .mcmc import _compute_hdi


def assemble_dacey_dataset(df):
    """Assemble xarray dataset from [1]_ data.

    Parameters
    ----------
    df : pd.DataFrame
         The Dacey1992_RGC dataframe containing dendritic field diameter, in
         arcmin, of parasol and midget RGCs.

    Returns
    -------
    dataset : xarray.Dataset
         That data, rearranged to a dataset.

    References
    ----------
    .. [1] Dacey, D. M., & Petersen, M. R. (1992). Dendritic field size and
       morphology of midget and parasol ganglion cells of the human retina.
       Proceedings of the National Academy of Sciences, 89(20), 9666–9670.
       http://dx.doi.org/10.1073/pnas.89.20.9666

    """
    df['dendritic_field_diameter_deg'] = df['dendritic_field_diameter_min'] / 60
    df = df.set_index('cell_type')
    for n in df.index.unique():
        df.loc[n, 'cells'] = np.arange(len(df.loc[n]))
    return df.reset_index().set_index(['cells', 'cell_type']).to_xarray()


def hinged_line(ecc, slope, hinge_ecc, intercept=0):
    """Hinged line with an optional intercept.

    Parameters
    ----------
    ecc : jnp.ndarray
        Eccentricity of RGC cells.
    slope : jnp.ndarray
        The slope giving the relationship between eccentricity and diameter.
    hinge_ecc : jnp.ndarray
        Eccentricity at which the line hinges.
    intercept : jnp.ndarray, optional
        The intercept for the line.

    Returns
    -------
    diameter_mean : jnp.ndarray
        The mean diameter at each location.

    """
    diam = intercept + ecc * slope
    return diam.clip(intercept + hinge_ecc * slope)


def model_physiological_scaling(eccentricity, observed_diameter=None):
    """Probabilistic model of dendritic field size, as function of eccentricity.

    Fits ``hinged_line`` to eccentricity in order to get the mean diameter at
    each location, then uses a normal distribution with the given mean (and
    standard deviation equal to a scaling parameter times the mean) to predict
    the observed diameters. We use this scaled normal because of the
    observation that the spread of observed diameters increases with
    eccentricity.

    The slope of this line is our physiological scaling value.

    We fit everything independently across cell types.

    Parameters
    ----------
    eccentricity : jnp.ndarray
        Eccentricity of RGC cells, of shape (cells, cell_types)
    observed_diameter : jnp.ndarray or None, optional
        Observed diameters to condition our results on. If None, don't
        condition.

    Returns
    -------
    diameters : jnp.ndarray
        Samples of RGC diameters.

    """
    cell_type_plate = numpyro.plate('cell_type', eccentricity.shape[-1],
                                    dim=-1)
    obs_nans = jnp.isnan(eccentricity)
    with cell_type_plate:
        # exponentiated to .018
        diam_slope = numpyro.sample('log_diameter_slope', dist.Normal(-4, 1))
        diam_int = numpyro.sample('diameter_intercept', dist.HalfCauchy(.1))
        diam_hinge = numpyro.sample('log_diameter_hinge_ecc', dist.Normal(0, 1))
        diam_scale = numpyro.sample('diameter_scale', dist.HalfCauchy(.1))
        # fill in missing eccentricity observations with random values in the
        # right range.
        if obs_nans is not None:
            eccentricity_fillin = numpyro.sample('dummy_ecc',
                                                 dist.Uniform(jnp.nanmin(eccentricity),
                                                              jnp.nanmax(eccentricity)).mask(False),
                                                 sample_shape=eccentricity.shape)
            eccentricity = jnp.where(obs_nans, eccentricity_fillin, eccentricity)
        diameter_mean = numpyro.deterministic('diameter_mean',
                                              hinged_line(eccentricity,
                                                          jnp.exp(diam_slope),
                                                          jnp.exp(diam_hinge),
                                                          diam_int))
        if observed_diameter is not None:
            # use Bayesian imputation to fill out any missing diameter observations
            imputed_diam = numpyro.sample('rgc_diameter_imputed',
                                          dist.Normal(diameter_mean, diam_scale*diameter_mean,
                                                      validate_args=True).mask(False))
            observed_diameter = jnp.where(obs_nans, imputed_diam, observed_diameter)
        return numpyro.sample('rgc_diameter', dist.Normal(diameter_mean,
                                                          diam_scale*diameter_mean, validate_args=True),
                              obs=observed_diameter)


def run_phys_scaling_inference(dataset, step_size=.1, num_draws=1000,
                               num_chains=1, num_warmup=500, seed=0,
                               target_accept_prob=.8, max_tree_depth=10,
                               **nuts_kwargs):
    """Run MCMC inference for physiological scaling, conditioned on data.

    Uses NUTS sampler.

    For some reason, numpyro has trouble running in parallel, even with the
    suggested `numpyro.set_host_device_count(n)` -- I think the solution is to
    set the proper environmental variable yourself: `export
    XLA_FLAGS=--xla_force_host_platform_device_count=n`. Check
    `jax.lib.xla_bridge.device_count()` to see if it worked.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing eccentricity and dendritic_field_diameter_deg data
        variables.
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
    target_accept_prob : float, optional
        Target acceptance probability for NUTS.
    max_tree_depth : int, optional
        Max depth of the tree for NUTS.
    nuts_kwargs :
        Passed to NUTS at initialization

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        The MCMC object that has run inference. Pass to assemble_inf_data.

    """
    ecc = jnp.array(dataset.eccentricity_deg.values, dtype=jnp.float32)
    diams = jnp.array(dataset.dendritic_field_diameter_deg.values, dtype=jnp.float32)
    mcmc_kernel = numpyro.infer.NUTS(model_physiological_scaling,
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
    mcmc.run(PRNGKey(seed), ecc, diams)
    return mcmc


def assemble_inf_data(mcmc, dataset, seed=1):
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

    """
    ecc = jnp.array(dataset.eccentricity_deg.values, dtype=jnp.float32)
    post_pred = numpyro.infer.Predictive(model_physiological_scaling,
                                         posterior_samples=mcmc.get_samples())
    post_pred = post_pred(PRNGKey(seed), ecc)
    post_pred = az.from_numpyro(posterior_predictive=post_pred,
                                coords=dataset.coords,
                                dims={'rgc_diameter': ['cells', 'cell_type'],
                                      'diameter_mean': ['cells', 'cell_type']})

    n_total_samples = list(mcmc.get_samples().values())[0].shape[0]
    prior = numpyro.infer.Predictive(model_physiological_scaling,
                                     num_samples=n_total_samples)
    prior = prior(PRNGKey(seed+1), ecc)
    prior = az.from_numpyro(prior=prior,
                            coords=dataset.coords,
                            dims={'rgc_diameter': ['cells', 'cell_type'],
                                  'log_diameter_slope': ['cell_type'],
                                  'diameter_scale': ['cell_type'],
                                  'log_diameter_hinge_ecc': ['cell_type'],
                                  'diameter_intercept': ['cell_type'],
                                  'dummy_ecc': ['cells', 'cell_type'],
                                  'diameter_mean': ['cells', 'cell_type']})

    inf_data = az.from_numpyro(mcmc,
                               coords=dataset.coords,
                               dims={'log_diameter_slope': ['cell_type'],
                                     'diameter_scale': ['cell_type'],
                                     'log_diameter_hinge_ecc': ['cell_type'],
                                     'diameter_intercept': ['cell_type'],
                                     'dummy_ecc': ['cells', 'cell_type'],
                                     'diameter_mean': ['cells', 'cell_type'],
                                     'rgc_diameter_imputed': ['cells', 'cell_type'],
                                     'rgc_diameter': ['cells', 'cell_type']})

    inf_data = inf_data + post_pred + prior
    inf_data.add_groups({'prior_predictive':
                         inf_data.prior[['rgc_diameter',]]})
    inf_data.prior = inf_data.prior.drop_vars(['cells', 'rgc_diameter',])
    inf_data.posterior_predictive['eccentricity'] = (['cells', 'cell_type'], ecc)
    inf_data.posterior_predictive = inf_data.posterior_predictive.set_coords('eccentricity')
    inf_data.prior_predictive['eccentricity'] = (['cells', 'cell_type'], ecc)
    inf_data.prior_predictive = inf_data.prior_predictive.set_coords('eccentricity')
    inf_data.observed_data['eccentricity'] = (['cells', 'cell_type'], ecc)
    # then there was missing data, so we imputed the responses
    if np.isnan(dataset.dendritic_field_diameter_deg).any():
        inf_data.observed_data = inf_data.observed_data.rename({'rgc_diameter': 'imputed_rgc_diameter'})
    return inf_data


def inf_data_to_df(inf_data, kind='predictive', query_str=None, hdi=False):
    """Convert inf_data to a dataframe, for plotting.

    We exponentiate the log_diameter_slope and log_diameter_hinge_ecc
    variables.

    Parameters
    ----------
    inf_data : arviz.InferenceData
        arviz InferenceData object (xarray-like) created by `assemble_inf_data`.
    kind : {'predictive', 'parameters'}, optional
        Whether to create df containing predictive info (responses and
        probability_correct) or model parameter info.
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
            tmp = inf_data[d].drop_dims('cells')
            if hdi:
                tmp = _compute_hdi(tmp, hdi)
            tmp = tmp.to_dataframe()
            for c in tmp.columns:
                if c.startswith('log'):
                    tmp[c.replace('log_', '')] = tmp[c].map(np.exp)
            tmp = tmp.reset_index().melt(tmp.index.names)
            tmp['distribution'] = d
            df.append(tmp)
        df = pd.concat(df).reset_index(drop=True)
        vars = [v for v in df.variable.unique() if 'log' not in v]
        df = df.query("variable in @vars")
    if query_str is not None:
        df = df.query(query_str)
    return df
