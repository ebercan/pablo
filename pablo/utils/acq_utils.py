"""Acquisition helpers shared by the optimization loops.

:func:`get_joint_log_ei` scores a whole candidate batch with one joint
qLogEI value; :func:`best_observed_f` returns the incumbent in the original
outcome space, so ``best_f`` matches the scale of the posterior.
"""

import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler


def get_joint_log_ei(model, best_f, candidates, num_samples=1024):
    r"""Compute the joint qLogExpectedImprovement value for a batch of candidates.

    Constructs a ``qLogExpectedImprovement`` acquisition function and evaluates
    it on the full candidate batch to obtain a single joint acquisition value.

    Args:
        model: A fitted BoTorch model.
        best_f: Best observed function value (scalar).
        candidates: A ``q x d`` or ``1 x q x d`` tensor of candidate points.
        num_samples: Number of MC samples for the Sobol QMC sampler.

    Returns:
        The joint qLogEI value as a Python float.
    """
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

    q_log_ei = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)

    with torch.no_grad():
        val = q_log_ei(candidates)
    return val.item()


def best_observed_f(model: Model) -> float:
    r"""Largest observed target of ``model`` in outcome space.

    ``model.train_targets`` lives in the transformed space when the model has
    an ``outcome_transform`` (e.g. ``Standardize``), while acquisition
    functions compare against posteriors in the original space. This helper
    undoes the transform so ``best_f`` is on the same scale as the posterior.

    Args:
        model: A BoTorch model, possibly a fantasy model from
            ``condition_on_observations``.

    Returns:
        The maximum observed target in the original outcome space.
    """
    targets = model.train_targets
    transform = getattr(model, "outcome_transform", None)
    if transform is not None:
        targets, _ = transform.untransform(targets.unsqueeze(-1))
    return targets.max().item()
