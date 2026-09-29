"""Training objectives for GeoDynFormer."""


def elbo(model, xu, anneal_param):
    """Calculate the evidence lower bound (ELBO) training objective.

    Parameters
    ----------
    model : torch.nn.Module
        GeoDynFormer model used to evaluate the objective.
    xu : torch.Tensor
        Model input containing the response and stimulus sequences.
    anneal_param : torch.Tensor
        Single-element annealing parameter used in the ELBO calculation.
        The value is expected to lie between 0 and 1.

    Returns
    -------
    avg_negative_log_likelihood : torch.Tensor
        Negative log-likelihood of the model responses, averaged over the
        batch.
    loss : torch.Tensor
        Negative ELBO, averaged over the batch.
    """
    x, likelihood, w, _, w_means, w_vars = model(xu)

    log_likelihood = likelihood.log_prob(x).sum(0).sum(-1)
    log_posterior = (
        model.qw_x(w_means, w_vars)
        .log_prob(w)
        .sum(0)
        .sum(-1)
    )
    log_prior = (
        model.pw(*model.pw_params)
        .log_prob(w)
        .sum(0)
        .sum(-1)
    )

    loss = -(
            anneal_param * log_likelihood
            + anneal_param * log_prior
            - log_posterior
    ).mean(0)

    avg_negative_log_likelihood = -log_likelihood.mean(0)

    return avg_negative_log_likelihood, loss
