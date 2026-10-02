import torch

def normalise(a):
    return torch.softmax(a.double(), dim=0)


def ess(a):
    a = a.double()
    return float(torch.exp(2 * torch.logsumexp(a, 0) - torch.logsumexp(2 * a, 0)))


def systematic_resample(w, generator=None, num=None):
    """w [N] normalised -> ancestor indices [num] int64 (num defaults to N). One uniform draw.

    `num` must be the number of draws wanted: taking `num` strata out of N and truncating
    would bias the result toward low indices.
    """
    n = w.shape[0]
    num = n if num is None else num
    u = torch.rand((), generator=generator, dtype=torch.float64, device=w.device)
    positions = (torch.arange(num, dtype=torch.float64, device=w.device) + u) / num
    cdf = w.double().cumsum(0).contiguous()
    return torch.searchsorted(cdf, positions.contiguous()).clamp_(max=n - 1)


def residual_resample(w, generator=None):
    """Residual resampling: floor(N*w) deterministic copies, remainder drawn systematically.

    With exactly uniform weights this returns arange(N) and draws no randomness, which is what
    makes a lambda=0 run bit-identical to the unsteered sampler even with resampling enabled.
    """
    n = w.shape[0]
    w = w.double()
    counts = torch.floor(n * w)
    n_det = int(counts.sum().item())
    idx = torch.repeat_interleave(torch.arange(n, device=w.device), counts.long())
    n_res = n - n_det
    if n_res > 0:
        resid = n * w - counts
        resid = resid / resid.sum()
        idx = torch.cat([idx, systematic_resample(resid, generator, num=n_res)])
    return idx[:n].long()
