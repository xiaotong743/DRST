"""Every global a experiments module's functions reach for must actually exist in that module.

Python binds function bodies at call time, so a missing import is invisible to `import`, to
`py_compile` and to any test that does not execute the specific line. This has now shipped twice:
`common._native_pool` and `common.mem_probe` called `steer(...)` and `Config(...)` with neither
name imported -- breaking the `calibrate` stage of both drivers -- and `utils.pareto` used `json`,
`os` and `out_path` after being moved out of steer_a.py without them. Both would have raised on a
GPU node, an hour into a job.

LOAD_GLOBAL is the exact signal: an attribute access (`mdl.ts`) compiles to LOAD_ATTR and a
function-local import (`import shutil` inside the body) to a local store, so neither is flagged.
"""
import builtins
import dis
import importlib

import pytest

MODULES = ["common", "utils", "steer_a", "run", "verify", "toy", "analytic"]


def _loaded_globals(code, seen=None):
    """Every name reached by LOAD_GLOBAL in `code` and in every function nested inside it."""
    seen = seen if seen is not None else set()
    for ins in dis.get_instructions(code):
        if ins.opname == "LOAD_GLOBAL" and isinstance(ins.argval, str):
            seen.add(ins.argval)
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _loaded_globals(const, seen)
    return seen


@pytest.mark.parametrize("name", MODULES)
def test_every_global_an_experiment_module_reaches_for_is_defined(name):
    mod = importlib.import_module(name)
    missing = sorted(n for n in _loaded_globals(mod.__loader__.get_code(name))
                     if not hasattr(mod, n) and not hasattr(builtins, n))
    assert not missing, (
        f"{name}.py calls these names but nothing defines or imports them: {missing}. "
        f"They will raise NameError the first time that line runs.")


def test_the_check_catches_a_missing_import():
    """A guard against the guard: if LOAD_GLOBAL stopped being the right signal this would pass
    vacuously, and the whole file would be theatre."""
    ns = {}
    exec("def f():\n    return json.load(open('x'))\n", ns)
    assert "json" in _loaded_globals(ns["f"].__code__)
    exec("def g():\n    import json\n    return json.load(open('x'))\n", ns)
    assert "json" not in _loaded_globals(ns["g"].__code__), "a local import must not be flagged"
    exec("def h(mdl):\n    return mdl.json\n", ns)
    assert "json" not in _loaded_globals(ns["h"].__code__), "an attribute must not be flagged"
