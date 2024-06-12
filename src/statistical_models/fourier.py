import numpy as np
from numpy import fft


class FourierForecast:
    def __init__(self, horizon=None):
        self.horizon = horizon

    def fourierExtrapolation(self, x):
        n = x.size
        # Use a smaller number of harmonics if possible
        n_harm = min(2000, n // 2)
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # Find linear trend in x
        x_notrend = x - p[0] * t  # Detrended x
        x_freqdom = fft.fft(x_notrend)  # Detrended x in frequency domain
        f = fft.fftfreq(n)  # Frequencies

        # Sort indexes by frequency, lower -> higher
        indexes = np.argsort(np.abs(f))

        # Precompute the time array for forecasting
        t_extended = np.arange(0, n + self.horizon)
        restored_sig = np.zeros(t_extended.size)

        for i in indexes[:1 + n_harm * 2]:
            ampli = np.abs(x_freqdom[i]) / n  # Amplitude
            phase = np.angle(x_freqdom[i])  # Phase
            restored_sig += ampli * \
                np.cos(2 * np.pi * f[i] * t_extended + phase)

        forecast = restored_sig + p[0] * t_extended
        return forecast[-self.horizon:]

    def predict(self, X):
        forecasts = np.array([self.fourierExtrapolation(x.ravel()) for x in X])
        return forecasts
