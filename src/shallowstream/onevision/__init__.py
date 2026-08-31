"""Current ShallowStream LLaVA-OneVision runtimes with lazy exports."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "ShallowStreamLLaVAOneVisionV3": ".v3",
    "LLaVAOneVisionRuntime": ".runtime",
}
_LAZY_MODULES = {
    "retrieval_instrumentation",
    "runtime",
    "v3",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_MODULES:
        return import_module(f".{name}", __name__)
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()).union(_LAZY_EXPORTS).union(_LAZY_MODULES))
