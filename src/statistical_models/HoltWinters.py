
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class HoltWintersForecast:
    def __init__(self, horizon = None):
        self.horizon = horizon

    def predict(self, X):
        forecasts = np.array([ExponentialSmoothing(x.ravel(), trend='add', seasonal='add', seasonal_periods=self.horizon).fit().forecast(self.horizon) for x in X])

        return np.array(forecasts)
    