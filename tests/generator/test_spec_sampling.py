"""
Speculative sampling with sampled drafts (exllamav3/generator/spec_sampling.py), CPU only:
- truncated_dist reproduces the ComboSampler step stack (temperature, min-p, top-k, top-p, temp_last) exactly, run
  step by step up to the final sampling step
- verify() is lossless: the emitted first token is distributed as the target's p whatever q is, and a draft survives
  with probability sum min(p, q)
"""
import os
os.environ["EXL3_FUSED_SAMPLER"] = "0"   # the step-by-step stack, so its distribution can be read before sampling

import pytest

pytestmark = pytest.mark.nogpu
import torch

from exllamav3.generator.sampler.custom import SamplingState, SS, SS_Sample
from exllamav3.generator.sampler.presets import ComboSampler
from exllamav3.generator import spec_sampling as sp

V = 512


def stack_dist(sampler, logits):
    """Dense distribution of the sampler's steps up to (not including) the final sample step."""
    st = SamplingState(rand_u32 = 0, dim = V, bsz = 1, in_logits = logits.view(1, V))
    for step in sampler.steps:
        if isinstance(step, SS_Sample):
            break
        step.run(st)
    if st.state in (SS.LOGITS, SS.LOGITS_S):
        probs = torch.softmax(st.logits, dim = -1)
    else:
        probs = st.probs / st.probs.sum(dim = -1, keepdim = True)
    dense = torch.zeros(1, V)
    if st.state in (SS.LOGITS_S, SS.PROBS_S, SS.PROBS_N_S):
        dense.scatter_(1, st.indices, probs)
    else:
        dense = probs
    return dense.view(-1)


SETTINGS = [dict(temperature = 0.6, top_p = 0.95, top_k = 20),
            dict(temperature = 1.0, top_p = 0.9, top_k = 40, min_p = 0.05),
            dict(temperature = 0.7, top_k = 8, min_p = 0.1, temp_last = True),
            dict(temperature = 1.3, top_k = 64)]


@pytest.mark.parametrize("kw", SETTINGS)
def test_truncated_dist_matches_sampler_steps(kw):
    torch.manual_seed(0)
    s = ComboSampler(**kw)
    assert s.spec_nominal is not None
    for _ in range(20):
        logits = torch.randn(V) * 3
        ref = stack_dist(s, logits.clone())
        ids, probs = sp.truncated_dist(logits.view(1, V), s.spec_nominal)
        got = torch.zeros(V).scatter_(0, ids.view(-1), probs.view(-1))
        assert torch.allclose(got, ref, atol = 1e-5), (kw, (got - ref).abs().max())


def test_ineligible_samplers_have_no_nominal():
    assert ComboSampler(temperature = 0.6, top_k = 20, rep_p = 1.1).spec_nominal is None
    assert ComboSampler(temperature = 0.6, top_k = 0).spec_nominal is None   # unbounded support
    assert ComboSampler(temperature = 0.0, top_k = 20).spec_nominal is None   # greedy


@pytest.mark.parametrize("draft_temp", [0.0, 0.5, 1.0, 1.5])
def test_verify_is_lossless_and_accepts_overlap(draft_temp):
    torch.manual_seed(1)
    nom = ComboSampler(temperature = 0.6, top_p = 0.95, top_k = 20).spec_nominal
    gen = torch.Generator().manual_seed(2)
    tgt = torch.randn(2, V) * 2                      # position 0 verifies the draft, position 1 is the bonus
    drf = tgt[0:1] + torch.randn(1, V) * 1.0          # a drafter that roughly agrees
    p_ids, p_probs = sp.truncated_dist(tgt[0:1], nom)
    p = torch.zeros(V).scatter_(0, p_ids.view(-1), p_probs.view(-1))
    q_ids, q_probs = sp.truncated_dist(drf, nom, nom["temperature"] * draft_temp)
    q = torch.zeros(V).scatter_(0, q_ids.view(-1), q_probs.view(-1))
    N = 40000
    counts = torch.zeros(V)
    acc = 0
    for _ in range(N):
        tok, qi, qp = sp.sample_draft(drf, nom, nom["temperature"] * draft_temp, gen)
        out = sp.verify(tgt, tok.view(-1), [qi.view(-1)], [qp.view(-1)], nom, gen)
        counts[out[0]] += 1
        acc += int(out[0] == tok.view(-1)[0])
    emp = counts / N
    # lossless: total variation from p within sampling noise
    assert 0.5 * (emp - p).abs().sum() < 0.03, 0.5 * (emp - p).abs().sum()
    assert abs(acc / N - torch.minimum(p, q).sum().item()) < 0.015, (acc / N, torch.minimum(p, q).sum().item())
