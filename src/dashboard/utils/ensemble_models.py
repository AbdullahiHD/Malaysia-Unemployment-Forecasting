"""
Ensemble models for youth unemployment forecasting
This module provides compatibility classes for loading the pickled wrapper models
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys

# Try to import PyEMD if available
try:
    from PyEMD import CEEMDAN
    PYEMD_AVAILABLE = True
    print("✅ PyEMD package imported successfully")
except ImportError:
    PYEMD_AVAILABLE = False
    print("⚠️ PyEMD package not available - using simplified fallback")


class CEEMDANEnsemble:
    """
    CEEMDAN Ensemble model compatible with pickled wrappers
    This is a simplified version of the original class to enable loading
    the pickled wrapper models
    """
    
    def __init__(self, n_estimators=10, random_state=42, ensemble_estimator='rf'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ensemble_estimator = ensemble_estimator
        self.ceemdan = CEEMDAN(trials=20) if PYEMD_AVAILABLE else None
        self.fitted = False
        self.imfs = None
        self.models = []
        self.residual_model = None
        self.n_imfs = 0
        self.X_history = None
        self.input_size = None
        
        # Set up the ensemble estimator
        if ensemble_estimator == 'rf':
            self.base_estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state
            )
        else:
            self.base_estimator = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0, random_state=random_state)
            )

    def fit(self, X, y=None):
        """Fit the model - simplified implementation for compatibility"""
        self.fitted = True
        self.input_size = len(X)
        self.X_history = X.copy()
        return self
        
    def predict(self, X):
        """
        Predict using the ensemble model
        If PyEMD is not available, provide a simplified prediction
        """
        # Ensure we've been fitted
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Helper: simple trend-based fallback
        def _trend_forecast(series, horizon):
            if len(series) < 2:
                return np.array([series[-1]])
            last_vals = series[-5:] if len(series) >= 5 else series
            trend = np.mean(np.diff(last_vals))
            base = series[-1]
            return np.array([base + (i + 1) * trend for i in range(horizon)])

        # Decide forecast horizon (max 14 like original wrappers)
        horizon = min(14, max(1, len(X) // 3))

        # If no sub-models are available, fall back immediately
        if not getattr(self, "models", None):
            return _trend_forecast(X, horizon)

        # 1) decompose (or use stored IMFs)
        imfs = self.ceemdan(X) if PYEMD_AVAILABLE else self.imfs

        # Missing or mismatched IMFs → fallback
        if imfs is None or len(imfs) != len(self.models):
            return _trend_forecast(X, horizon)

        # 2) forecast each IMF with its own sub-model
        subfs = []
        for idx, m in enumerate(self.models):
            series = imfs[idx]
            # --- Case 1: Keras model + scaler (original notebooks) ---
            if (
                hasattr(m, 'predict') and
                hasattr(self, 'scalers') and
                len(self.scalers) > idx
            ):
                scaler = self.scalers[idx]
                win_raw = series[-self.n_lag:]
                scaled_hist = scaler.transform(win_raw.reshape(-1, 1)).flatten().tolist()
                preds_scaled = []
                hist = scaled_hist.copy()
                for _ in range(horizon):
                    x_in = np.array(hist[-self.n_lag:]).reshape((1, self.n_lag, 1))
                    yhat = m.predict(x_in, verbose=0)[0][0]
                    preds_scaled.append(yhat)
                    hist.append(yhat)
                preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

            # --- Case 2: Custom LSTMForecaster wrapper with sequence_length ---
            elif hasattr(m, 'sequence_length'):
                win = series[-m.sequence_length:].reshape(-1,1)
                scaled = m.scaler.transform(win).flatten()
                preds = m.predict(scaled, steps=horizon)
            else:
                win = series[-horizon:].reshape(-1,1)
                preds = m.predict(win)
            subfs.append(np.asarray(preds).reshape(-1))

        # 3) forecast residual if exists
        if self.residual_model:
            resid = X - np.sum(imfs, axis=0)
            if hasattr(self.residual_model, 'sequence_length'):
                win = resid[-self.residual_model.sequence_length:].reshape(-1,1)
                scaled = self.residual_model.scaler.transform(win).flatten()
                r = self.residual_model.predict(scaled, steps=horizon)
            else:
                win = resid[-horizon:].reshape(-1,1)
                r = self.residual_model.predict(win)
            subfs.append(np.asarray(r).reshape(-1))

        # 4) sum up and return a clean 1-D array
        combined = np.sum(np.stack(subfs, axis=0), axis=0)
        return combined.reshape(-1)
    
    def __getstate__(self):
         """Custom state for pickling: include sub-models."""
         state = {
             'n_estimators': self.n_estimators,
             'random_state': self.random_state,
             'ensemble_estimator': self.ensemble_estimator,
             'fitted': self.fitted,
             'n_imfs': self.n_imfs,
             'input_size': self.input_size,
             'X_history': self.X_history,
             # Persist the actual trained components
             'imfs': self.imfs,
             'models': self.models,
             'residual_model': self.residual_model,
             # Extra attributes used by original Colab wrappers
             'scalers': self.scalers,
             'n_lag': self.n_lag,
             'horizon': self.horizon
         }
         return state
        
    def __setstate__(self, state):
        """Custom state for unpickling"""
        self.__init__(
            n_estimators=state.get('n_estimators', 10),
            random_state=state.get('random_state', 42),
            ensemble_estimator=state.get('ensemble_estimator', 'rf')
        )
        self.fitted       = state.get('fitted', False)
        self.n_imfs       = state.get('n_imfs', 0)
        self.input_size   = state.get('input_size', None)
        self.X_history    = state.get('X_history', None)
        # Restore trained components
        self.imfs           = state.get('imfs', None)
        self.models         = state.get('models', [])
        self.residual_model = state.get('residual_model', None)
        # Extra attributes used by original Colab wrappers
        self.scalers        = state.get('scalers', [])
        self.n_lag          = state.get('n_lag', getattr(self, 'n_lag', 12))
        self.horizon        = state.get('horizon', getattr(self, 'horizon', 14))
        # Retain any other keys present in state for future use
        for k, v in state.items():
            if not hasattr(self, k):
                setattr(self, k, v)

# Make sure the class is available in the global namespace for pickle loading
sys.modules['__main__'].CEEMDANEnsemble = CEEMDANEnsemble 