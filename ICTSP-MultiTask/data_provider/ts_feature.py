import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller
from scipy.signal import find_peaks
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf, pacf
from tsfresh.feature_extraction import feature_calculators as fc
import pywt
import random
from statsmodels.tsa.seasonal import STL

from scipy.signal import lfilter

def clip_output():
    min_value = np.random.randint(-25, -15)
    max_value = np.random.randint(15, 25)
    def decorator(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            return np.clip(output, min_value, max_value)
        return wrapper
    return decorator

# 1. Calculate mean
def mean(series):
    return np.mean(series, axis=1)

# 2. Calculate median
def median(series):
    return np.median(series, axis=1)

# 3. Calculate maximum value
def max_value(series):
    return np.max(series, axis=1)

# 4. Calculate minimum value
def min_value(series):
    return np.min(series, axis=1)

# 5. Calculate range
def range_value(series):
    return np.ptp(series, axis=1)

# 6. Calculate interquartile range
def iqr(series):
    return np.percentile(series, 75, axis=1) - np.percentile(series, 25, axis=1)

# 7. Calculate standard deviation
def std(series):
    return np.std(series, axis=1)

# 8. Calculate variance
def var(series):
    return np.var(series, axis=1)

# 9. Calculate skewness
def skewness(series):
    return skew(series, axis=1)

# 10. Calculate kurtosis
def kurt(series):
    return kurtosis(series, axis=1)

# 11. Calculate mean absolute deviation
def mad(series):
    return np.mean(np.abs(series - np.mean(series, axis=1, keepdims=True)), axis=1)

# 12. Calculate mean absolute percentage error
def mape(series):
    diffs = np.abs(np.diff(series, axis=1))
    return np.mean(diffs / (series[:, :-1] + 1e-5), axis=1) * 100

# 13. Calculate symmetric mean absolute percentage error
def smape(series):
    diffs = np.abs(np.diff(series, axis=1))
    return 2.0 * np.mean(diffs / (series[:, :-1] + series[:, 1:] + 1e-5), axis=1) * 100

# 14. Calculate maximum of rolling mean
def rolling_mean_max(series, window):
    return np.max(pd.DataFrame(series).T.rolling(window=window).mean().dropna().T.values, axis=1)

# 15. Calculate minimum of rolling mean
def rolling_mean_min(series, window):
    return np.min(pd.DataFrame(series).T.rolling(window=window).mean().dropna().T.values, axis=1)

# 16. Calculate maximum of rolling standard deviation
def rolling_std_max(series, window):
    return np.max(pd.DataFrame(series).T.rolling(window=window).std().dropna().T.values, axis=1)

# 17. Calculate minimum of rolling standard deviation
def rolling_std_min(series, window):
    return np.min(pd.DataFrame(series).T.rolling(window=window).std().dropna().T.values, axis=1)

# 18. Calculate p-value of ADF (Augmented Dickey-Fuller) unit root test
def adf_test_p_value(series):
    return np.array([adfuller(s)[1] for s in series])

# 19. Calculate maximum of autocorrelation function (ACF)
def max_acf(series, nlags):
    return np.array([np.max(acf(s, nlags=nlags)) for s in series])

# 20. Calculate maximum of partial autocorrelation function (PACF)
def max_pacf(series, nlags):
    return np.array([np.max(pacf(s, nlags=nlags)) for s in series])

# 21. Calculate number of periodic peaks
def num_peaks(series, distance, prominence):
    return np.array([len(find_peaks(s, distance=distance, prominence=prominence)[0]) for s in series])

# 22. Calculate signal energy
def signal_energy(series):
    return np.sum(series ** 2, axis=1) / (series.shape[1] ** 2)

# 23. Calculate Shannon entropy of the signal
def shannon_entropy(series):
    return np.array([entropy(pd.Series(s).value_counts(normalize=True)) for s in series]) / (series.shape[1])

# 24. Calculate signal percentiles
def percentile(series, q):
    return np.percentile(series, q, axis=1)

# 25. Calculate sum of absolute values of autocorrelation function
def autocorrelation_sum(series, lag):
    return np.array([np.sum(np.abs(np.correlate(s, s, mode='full')[len(s) - 1:-lag])) for s in series])

# 26. Calculate mean absolute change of the signal
def mean_abs_change(series):
    return np.mean(np.abs(np.diff(series, axis=1)), axis=1)

# 27. Calculate mean squared change of the signal
def mean_squared_change(series):
    return np.mean(np.diff(series, axis=1) ** 2, axis=1) ** (1/2)

# 28. Calculate sum of absolute values of the signal
def abs_energy(series):
    return np.array([fc.abs_energy(s) for s in series]) / (series.shape[1] ** 2)

# 29. Calculate energy of the signal after wavelet transform
def wavelet_energy(series, wavelet):
    return np.array([np.sum(np.array(pywt.wavedec(s, wavelet)[0]) ** 2) for s in series])

# 30. Calculate energy of the signal after discrete wavelet transform
def wavelet_denoised_energy(series, wavelet):
    return np.array([np.sum(np.array(pywt.waverec(pywt.wavedec(s, wavelet)[:-1] + [None] * len(pywt.wavedec(s, wavelet)[-1]), wavelet)) ** 2) for s in series])

# 31. Calculate time reversibility of the signal
def time_reversibility(series):
    return np.array([fc.time_reversal_asymmetry_statistic(s, lag=1) for s in series])

# 32. Calculate autoregression coefficients of the signal
def ar_coeffs(series, n_coeffs):
    return np.array([fc.autoregression_coefficients(s, n_coeffs) for s in series])

# 33. Calculate long and short term memory of the signal
def long_short_term_memory(series, memory):
    below_mean = np.array([fc.longest_strike_below_mean(s) for s in series])
    above_mean = np.array([fc.longest_strike_above_mean(s) for s in series])
    return below_mean / (series.shape[1]), above_mean / (series.shape[1])

# 34. Calculate the proportion of time the signal is above the mean
def time_above_mean(series):
    return np.array([fc.mean_abs_change(s) / len(s) for s in series])

@clip_output()
def extract_time_series_features(series, indices=None, randomized=True, get_indices=False):
    nlags = 10
    wavelet = 'db4'
    distance = 10
    prominence = 1
    n_coeffs = 5
    memory = 1
    
    lstm_0, lstm_1 = long_short_term_memory(series, memory)
    features = {
        'mean': mean(series),
        'median': median(series),
        'max': max_value(series),
        'min': min_value(series),
        'range': range_value(series),
        'iqr': iqr(series),
        'std': std(series),
        'var': var(series),
        'skewness': skewness(series),
        'kurtosis': kurt(series),
        'mad': mad(series),
        'mape': mape(series),
        'smape': smape(series),
        'rolling_mean_max_3': rolling_mean_max(series, 3),
        'rolling_mean_min_3': rolling_mean_min(series, 3),
        'rolling_std_max_3': rolling_std_max(series, 3),
        'rolling_std_min_3': rolling_std_min(series, 3),
        #'adf_test_p_value': adf_test_p_value(series),
        #'max_acf': max_acf(series, nlags),
        #'max_pacf': max_pacf(series, nlags),
        'num_peaks': num_peaks(series, distance, prominence),
        'signal_energy': signal_energy(series),
        'shannon_entropy': shannon_entropy(series),
        'percentile_25': percentile(series, 25),
        'percentile_75': percentile(series, 75),
        #'autocorrelation_sum': autocorrelation_sum(series, nlags),
        'mean_abs_change': mean_abs_change(series),
        'mean_squared_change': mean_squared_change(series),
        'abs_energy': abs_energy(series),
#         'wavelet_energy': wavelet_energy(series, wavelet),
#         'wavelet_denoised_energy': wavelet_denoised_energy(series, wavelet),
#         'time_reversibility': time_reversibility(series),
#         **ar_coeffs(series, n_coeffs),
        'longest_strike_below_mean': lstm_0,
        'longest_strike_above_mean': lstm_1,
        'time_above_mean': time_above_mean(series)
    }
    features = list(features.values())
    if randomized:
        if indices is None:
            indices = range(len(features))
            indices = random.sample(indices, int(np.ceil(len(features)*np.random.rand())))
            if get_indices:
                return indices
        features = [features[i] for i in indices]
    return np.stack(features).T

def differencing(series, order):
    diff_series = np.diff(series, n=order, axis=1)
    return diff_series

def moving_average(series, window):
    ma_series = np.zeros((series.shape[0], series.shape[1] - window + 1))
    for i in range(series.shape[0]):
        ma_series[i, :] = np.convolve(series[i, :], np.ones(window)/window, mode='valid')
    return ma_series

def exponential_smoothing(series, alpha):
    b = [alpha]
    a = [1, alpha - 1]
    smoothed_series = np.zeros(series.shape)
    for i in range(series.shape[0]):
        smoothed_series[i, :] = lfilter(b, a, series[i, :])
    return smoothed_series

def stl_decomposition(series, period, seasonal=7):
    decomposed = []
    for i in range(series.shape[0]):
        ts_series = pd.Series(series[i, :])
        stl = STL(ts_series, period=period, seasonal=seasonal)
        result = stl.fit()
        decomposed.append(np.stack([result.trend.values, result.seasonal.values, result.resid.values]))
    return np.concatenate(decomposed, axis=0)

@clip_output()
def time_series_transformation(data, method="differencing"):
    C, L = data.shape
    if method == "differencing":
        order = random.randint(1, min(3, L))  # Random order for differencing
        return differencing(data, order)
    elif method == "movingavg":
        window = random.randint(2, min(40, L))  # Random window size for moving average
        return moving_average(data, window)
    elif method == "expsmoothing":
        alpha = random.uniform(0.02, 0.98)  # Random alpha for exponential smoothing
        return exponential_smoothing(data, alpha)
    elif method == "decomposition":
        period = random.randint(2, min(12, L))  # Random period for STL decomposition
        seasonal = random.choice([3, 5, 7, 9, 11])  # Random seasonal parameter for STL decomposition
        return stl_decomposition(data, period, seasonal)