"""Model package with lazy access to optional estimator implementations."""

__all__ = ["HistGradientBoostingModel", "DEFAULT_FEATURE_NAMES"]


def __getattr__(name):
    if name in __all__:
        from .train_hist_gradient_boosting import (
            DEFAULT_FEATURE_NAMES,
            HistGradientBoostingModel,
        )
        return {
            "HistGradientBoostingModel": HistGradientBoostingModel,
            "DEFAULT_FEATURE_NAMES": DEFAULT_FEATURE_NAMES,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
