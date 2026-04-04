"""
Time Series Models
"""

# Lazy imports
def __getattr__(name):
    if name == "ARIMAModel":
        from .arima import ARIMAModel
        return ARIMAModel
    elif name == "ProphetModel":
        from .prophet_model import ProphetModel
        return ProphetModel
    elif name == "LSTMModel":
        from .lstm import LSTMModel
        return LSTMModel
    elif name == "TransformerForecaster":
        from .transformer import TransformerForecaster
        return TransformerForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ARIMAModel", "ProphetModel", "LSTMModel", "TransformerForecaster"]
