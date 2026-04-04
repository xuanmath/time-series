"""
Time Series Analysis Package
"""

# Lazy imports to avoid importing torch when not needed
__version__ = "0.1.0"

def __getattr__(name):
    if name == "ARIMAModel":
        from .models.arima import ARIMAModel
        return ARIMAModel
    elif name == "ProphetModel":
        from .models.prophet_model import ProphetModel
        return ProphetModel
    elif name == "LSTMModel":
        from .models.lstm import LSTMModel
        return LSTMModel
    elif name == "TransformerForecaster":
        from .models.transformer import TransformerForecaster
        return TransformerForecaster
    elif name == "FeatureEngineer":
        from .features.engineer import FeatureEngineer
        return FeatureEngineer
    elif name == "load_data":
        from .utils.data import load_data
        return load_data
    elif name == "evaluate_predictions":
        from .utils.metrics import evaluate_predictions
        return evaluate_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
