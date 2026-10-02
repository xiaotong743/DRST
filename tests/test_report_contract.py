"""Every diag key `run_variants.report` reads must be one the vanilla sampler produces.

`report` runs at the END of a GPU job, after every particle has been stepped. A key it reads that
only one branch emits does not fail fast -- it fails after hours of sampling, having saved
nothing. A branch-specific key read with `r["..."]` fails after hours of sampling, having saved
nothing.

So this asserts the contract directly, reading the required keys out of report's own source so a
newly added `r["..."]` is covered without anyone remembering to update a list here.
"""
import ast
import inspect

import torch

import common
from steer_a import ConfigA, SteererA
from steering.rewards import Pointwise
from steering.steerer import Config, Steerer


class _Flat:
    """Minimal endpoint-contract model: no GPU, no checkpoint, no reward structure."""
    ts = torch.linspace(0, 1, 5)

    def prior(self, n, generator=None):
        return torch.zeros(n, 3, 3)

    def endpoint(self, x, t):
        return x, torch.zeros_like(x)

    def score(self, x, v, t):
        return torch.zeros_like(x)

    def eps(self, t):
        return 0.5

    def project(self, x):
        return x


def _keys_report_requires():
    """The literal keys report() subscripts on a diag row, from its source."""
    src = inspect.getsource(common.run_variants)
    fn = next(n for n in ast.walk(ast.parse(src.lstrip()))
              if isinstance(n, ast.FunctionDef) and n.name == "report")
    required = set()
    for node in ast.walk(fn):
        # r["key"] -- a hard requirement. r.get("key", default) is tolerated by design.
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "r" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            required.add(node.slice.value)
    assert required, "found no r[...] reads; the extractor is broken, not the code"
    return required


def _diag(cls, cfg):
    return cls(_Flat(), Pointwise(lambda th: th.reshape(th.shape[0], -1).sum(1)), cfg, 4).run()


def test_report_only_requires_diag_keys_the_published_sampler_emits():
    required = _keys_report_requires()
    emitted = set(_diag(Steerer, Config(ts=_Flat.ts, lam=0.0, global_rng=True))["diag"][0])
    assert required <= emitted, (
        f"run_variants.report() subscripts {sorted(required - emitted)}, which Steerer never "
        f"puts in a diag row -- it would KeyError after the whole run had finished. Use "
        f"r.get(key, default) for anything only one branch emits.")


def test_report_survives_a_branch_a_run_end_to_end():
    """The path the unsteered control takes: SteererA at lam=0, straight through report()."""
    d = _diag(SteererA, ConfigA(ts=_Flat.ts, lam=0.0, temperature=0.45, global_rng=True))["diag"]
    for key in _keys_report_requires():
        assert key in d[0], key
    # and the branch-specific one degrades to the value the stored A campaign records carry
