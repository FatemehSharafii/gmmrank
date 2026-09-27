import numpy as np


def logprior_sigma(sigma, s=1.0):
    """
    Compute the log-prior for sigma using a half-normal distribution.

    Parameters
    ----------
    sigma : float
        Standard deviation parameter.
    s : float, optional
        Scale parameter of the half-normal prior. Default is 1.0.

    Returns
    -------
    float
        Log-prior density.
    """
    if sigma <= 0:
        return -np.inf

    return (
        0.5 * np.log(2 / (np.pi * s**2))
        - (sigma**2) / (2 * s**2)
    )


def log_likelihood_sigma(observed_data, model_data, sigma):
    """
    Compute the log-likelihood assuming normally distributed residuals.

    Parameters
    ----------
    observed_data : array-like
        Observed intensity measure values.
    model_data : array-like
        GMM-predicted intensity measure values.
    sigma : float
        Residual standard deviation.

    Returns
    -------
    float
        Log-likelihood.
    """
    if sigma <= 0:
        return -np.inf

    observed_data = np.asarray(observed_data)
    model_data = np.asarray(model_data)

    residual = observed_data - model_data
    n = len(observed_data)

    return (
        -n / 2 * np.log(sigma**2)
        - np.sum(residual**2) / (2 * sigma**2)
    )


def estimate_sigma_mcmc(
    observed_data,
    model_data,
    n_samples=5000,
    n_chains=4,
    burnin=0.20,
    proposal_sd=0.02,
    sigma_inits=None,
    prior_scale=1.0,
    random_seed=12345
):
    """
    Estimate the residual standard deviation using
    four-chain Metropolis-Hastings MCMC.

    Parameters
    ----------
    observed_data : array-like
        Observed intensity measure values.
    model_data : array-like
        GMM-predicted intensity measure values.
    n_samples : int, optional
        Number of MCMC iterations per chain. Default is 5000.
    n_chains : int, optional
        Number of independent MCMC chains. Default is 4.
    burnin : float, optional
        Fraction of samples discarded as burn-in. Default is 0.20.
    proposal_sd : float, optional
        Standard deviation of the random-walk proposal
        on the log(sigma) scale. Default is 0.02.
    sigma_inits : array-like, optional
        Initial sigma values for the chains.
        Default is [0.5, 0.7, 1.0, 1.3].
    prior_scale : float, optional
        Scale of the half-normal prior. Default is 1.0.
    random_seed : int, optional
        Random seed for reproducibility. Default is 12345.

    Returns
    -------
    posterior_sigma : float
        Posterior mean of sigma after burn-in.
    samples : ndarray
        MCMC samples with shape
        (n_chains, n_samples).
    acceptance_rates : ndarray
        Acceptance rate for each chain.
    """
    observed_data = np.asarray(observed_data)
    model_data = np.asarray(model_data)

    if sigma_inits is None:
        sigma_inits = [0.5, 0.7, 1.0, 1.3]

    if len(sigma_inits) != n_chains:
        raise ValueError(
            "The number of initial sigma values must equal n_chains."
        )

    burnin_samples = int(burnin * n_samples)

    rng = np.random.default_rng(random_seed)

    samples = np.zeros((n_chains, n_samples))
    acceptance_counts = np.zeros(n_chains, dtype=int)

    for chain in range(n_chains):

        sigma_current = sigma_inits[chain]
        log_sigma_current = np.log(sigma_current)

        logpost_current = (
            logprior_sigma(
                sigma_current,
                s=prior_scale
            )
            + log_likelihood_sigma(
                observed_data,
                model_data,
                sigma_current
            )
            + log_sigma_current
        )

        for i in range(n_samples):

            log_sigma_proposal = (
                log_sigma_current
                + rng.normal(0, proposal_sd)
            )

            sigma_proposal = np.exp(log_sigma_proposal)

            logpost_proposal = (
                logprior_sigma(
                    sigma_proposal,
                    s=prior_scale
                )
                + log_likelihood_sigma(
                    observed_data,
                    model_data,
                    sigma_proposal
                )
                + log_sigma_proposal
            )

            log_alpha = logpost_proposal - logpost_current

            if (
                log_alpha >= 0
                or np.log(rng.random()) < log_alpha
            ):
                sigma_current = sigma_proposal
                log_sigma_current = log_sigma_proposal
                logpost_current = logpost_proposal

                acceptance_counts[chain] += 1

            samples[chain, i] = sigma_current

    posterior_samples = samples[:, burnin_samples:]

    posterior_sigma = np.mean(posterior_samples)

    acceptance_rates = acceptance_counts / n_samples

    return (
        posterior_sigma,
        samples,
        acceptance_rates
    )


def compute_dic(
    observed_data,
    model_data,
    n_samples=5000,
    n_chains=4,
    burnin=0.20,
    proposal_sd=0.02,
    sigma_inits=None,
    prior_scale=1.0,
    random_seed=12345
):
    """
    Compute the Deviance Information Criterion (DIC)
    for a GMM.

    The residual standard deviation is estimated using
    four-chain Metropolis-Hastings MCMC. DIC is calculated as

        DIC = 2 * D_bar - D_hat

    where D_bar is the posterior mean deviance and D_hat
    is the deviance evaluated at the posterior mean of sigma.

    Parameters
    ----------
    observed_data : array-like
        Observed intensity measure values.
    model_data : array-like
        GMM-predicted intensity measure values.
    n_samples : int, optional
        Number of MCMC iterations per chain. Default is 5000.
    n_chains : int, optional
        Number of independent MCMC chains. Default is 4.
    burnin : float, optional
        Fraction of samples discarded as burn-in. Default is 0.20.
    proposal_sd : float, optional
        Standard deviation of the random-walk proposal
        on the log(sigma) scale. Default is 0.02.
    sigma_inits : array-like, optional
        Initial sigma values for the chains.
        Default is [0.5, 0.7, 1.0, 1.3].
    prior_scale : float, optional
        Scale of the half-normal prior. Default is 1.0.
    random_seed : int, optional
        Random seed for reproducibility. Default is 12345.

    Returns
    -------
    float
        DIC value.
    """
    observed_data = np.asarray(observed_data)
    model_data = np.asarray(model_data)

    n = len(observed_data)

    residual = observed_data - model_data
    rss = np.sum(residual**2)

    (
        posterior_sigma,
        samples,
        acceptance_rates
    ) = estimate_sigma_mcmc(
        observed_data,
        model_data,
        n_samples=n_samples,
        n_chains=n_chains,
        burnin=burnin,
        proposal_sd=proposal_sd,
        sigma_inits=sigma_inits,
        prior_scale=prior_scale,
        random_seed=random_seed
    )

    burnin_samples = int(burnin * n_samples)

    sigma_samples = samples[:, burnin_samples:].ravel()

    # Deviance for each posterior sample
    D_samples = (
        n * np.log(2 * np.pi)
        + 2 * n * np.log(sigma_samples)
        + rss / (sigma_samples**2)
    )

    # Posterior mean deviance
    D_bar = np.mean(D_samples)

    # Deviance at posterior mean sigma
    D_hat = (
        n * np.log(2 * np.pi)
        + 2 * n * np.log(posterior_sigma)
        + rss / (posterior_sigma**2)
    )

    # Deviance Information Criterion
    dic = 2 * D_bar - D_hat

    return dic