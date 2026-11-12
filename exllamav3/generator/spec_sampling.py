"""
Speculative sampling with sampled drafts (Leviathan et al. 2023, Chen et al. 2023) for the MTP drafter.

Greedy drafting proposes the drafter's top token and keeps it when the target's own sample happens to equal it, so a
drafted position survives with probability p(argmax q): at temperature 0.6 an even split between two tokens loses
half the drafts even when the drafter agrees with the target exactly. Here the drafter samples x ~ q from the same
truncated distribution the target samples from (optionally at a scaled temperature), and the target accepts x with
probability min(1, p(x) / q(x)); on rejection it samples from the residual max(p - q, 0), renormalised, and the round
ends. The accepted sequence is distributed exactly as the target's sampler would have produced it token by token, and
a drafted position survives with probability sum_x min(p(x), q(x)) = 1 - TV(p, q).

p and q are the distributions of the job's ComboSampler as nominal parameters (temperature, min-p, top-k, top-p,
temp_last), rebuilt here over the top-k support with the same step order and boundary rules as the sampler steps
(sampler/custom.py). Only penalty-free samplers with 1 <= top_k <= MAX_K qualify (ComboSampler.spec_nominal); every
other job keeps greedy drafting.
"""
from __future__ import annotations

import torch

MAX_K = 256   # support kept per drafted position (the top-k set); larger top_k falls back to greedy drafting


def truncated_dist(logits: torch.Tensor, nominal: dict, temperature: float | None = None):
    """
    (ids, probs), each (rows, K): the sampler's distribution over its top-k support, normalised. logits: (rows, V).
    temperature overrides the nominal one (drafter scaling); <= 0 gives a one-hot on the argmax.
    """
    T = nominal["temperature"] if temperature is None else temperature
    K = nominal["top_k"]
    x = logits.float()
    if T <= 0.0:
        ids = torch.argmax(x, dim = -1, keepdim = True)
        return ids, torch.ones_like(ids, dtype = torch.float)
    temp_last = nominal["temp_last"]
    # SS_Temperature, then SS_MinP on the normalised full distribution (threshold relative to the top token, so the
    # normaliser cancels and the test is done on the top-k values), then SS_TopK
    vals, ids = torch.topk(x if temp_last else x / T, K, dim = -1)
    probs = torch.softmax(vals, dim = -1) if nominal["min_p"] == 0.0 else torch.exp(vals - vals[:, :1])
    if nominal["min_p"] > 0.0:
        probs = probs * (probs >= nominal["min_p"])   # relative to the top token, which is probs[:, 0] == 1
    # SS_TopP after renormalising the surviving tokens: keep each token whose preceding mass is short of P
    probs = probs / probs.sum(dim = -1, keepdim = True)
    if nominal["top_p"] < 1.0:
        keep = (probs.cumsum(dim = -1) - probs) < nominal["top_p"]
        keep[:, 0] = True
        probs = probs * keep
    if temp_last and T != 1.0:
        probs = probs.pow(1.0 / T)
    probs = probs / probs.sum(dim = -1, keepdim = True)
    return ids, probs


def sample_draft(logits: torch.Tensor, nominal: dict, temperature: float | None, gen: torch.Generator):
    """Drafted token per row from q, plus q's support: (token (rows, 1), ids (rows, K), probs (rows, K))."""
    ids, probs = truncated_dist(logits, nominal, temperature)
    pick = torch.multinomial(probs, 1, generator = gen)
    return torch.gather(ids, 1, pick), ids, probs


def verify(target_logits: torch.Tensor, draft: torch.Tensor, q_ids: list, q_probs: list, nominal: dict,
           gen: torch.Generator) -> torch.Tensor:
    """
    Tokens to feed the generator's acceptance walk for one job: (W,) long on the logits' device, W = drafted + 1.
    target_logits: (W, V); draft: (W - 1,) drafted ids; q_ids / q_probs: per drafted position (K,) tensors.
    Position i holds draft[i] when accepted, else a residual sample (never equal to draft[i]); the walk stops at the
    first mismatch, so positions after it are don't-cares. The last position is a plain sample from p (the bonus
    token when every draft is accepted).
    """
    W, V = target_logits.shape
    p_ids, p_probs = truncated_dist(target_logits, nominal)
    dev = target_logits.device
    p = torch.zeros((W, V), dtype = torch.float, device = dev).scatter_(1, p_ids, p_probs)
    out = torch.multinomial(p_probs, 1, generator = gen).view(-1)
    out = torch.gather(p_ids, 1, out.view(-1, 1)).view(-1)
    if W == 1:
        return out
    n = W - 1
    q = torch.zeros((n, V), dtype = torch.float, device = dev)
    q.scatter_(1, torch.stack(q_ids).view(n, -1), torch.stack(q_probs).view(n, -1))
    d = draft.view(n, 1).to(dev)
    pd, qd = torch.gather(p[:n], 1, d).view(-1), torch.gather(q, 1, d).view(-1)
    u = torch.rand(n, generator = gen, device = dev)
    accept = u * qd < pd                      # u < p/q without dividing (qd > 0: d was sampled from q)
    resid = (p[:n] - q).clamp_(min = 0.0)
    mass = resid.sum(dim = -1, keepdim = True)
    # p == q leaves no residual, but then the draft is always accepted; guard the multinomial with p itself
    resid = torch.where(mass > 0, resid, p[:n])
    fix = torch.multinomial(resid, 1, generator = gen).view(-1)
    out[:n] = torch.where(accept, d.view(-1), fix)
    return out
