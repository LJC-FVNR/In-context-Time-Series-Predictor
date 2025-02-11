import numpy as np
import pandas as pd
from scipy.integrate import odeint
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm

from multiprocessing import Pool, Manager, Process
import os
from tqdm import tqdm

def standardize(series):
    return (series - np.mean(series)) / (np.std(series) + 1e-8)

def random_scale(series):
    scale_factor = np.random.uniform(0.2, 2.2)
    offset = np.random.uniform(-1.5, 1.5)
    return (series + offset) * scale_factor

def generate_random_walk(length, noise_level=1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    series = np.cumsum(np.random.randn(length) * noise_level).astype(np.float64)
    return random_scale(standardize(series))

def generate_sine_wave(length, freq=None, amplitude=None, phase_shift=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if freq is None:
        freq = np.random.uniform(0.001, 500)
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5)
    if phase_shift is None:
        phase_shift = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 2 * np.pi, length)
    sine_wave = amplitude * np.sin(freq * t + phase_shift)
    noise = noise_level * np.random.randn(length)
    series = sine_wave + noise
    return random_scale(standardize(series))

def generate_cosine_wave(length, freq=None, amplitude=None, phase_shift=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if freq is None:
        freq = np.random.uniform(0.001, 200)
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5)
    if phase_shift is None:
        phase_shift = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 2 * np.pi, length)
    cosine_wave = amplitude * np.cos(freq * t + phase_shift)
    noise = noise_level * np.random.randn(length)
    series = cosine_wave + noise
    return random_scale(standardize(series))

def generate_square_wave(length, freq=None, amplitude=None, phase_shift=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if freq is None:
        freq = np.random.uniform(0.01, 100)
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5)
    if phase_shift is None:
        phase_shift = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 2 * np.pi, length)
    square_wave = amplitude * np.sign(np.sin(freq * t + phase_shift))
    noise = noise_level * np.random.randn(length)
    series = square_wave + noise
    return random_scale(standardize(series))

def generate_sawtooth_wave(length, freq=None, amplitude=None, phase_shift=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if freq is None:
        freq = np.random.uniform(0.01, 100)  # Adjust frequency range if needed
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5)
    if phase_shift is None:
        phase_shift = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 2 * np.pi * freq, length)
    sawtooth_wave = amplitude * (2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5)))
    noise = noise_level * np.random.randn(length)
    series = sawtooth_wave + noise
    return random_scale(standardize(series))

def generate_polynomial_trend(length, degree=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if degree is None:
        degree = np.random.randint(1, 10)
    x = np.linspace(-8, 8, length)
    coeffs = np.random.randn(degree + 1)
    polynomial_trend = np.polyval(coeffs, x)
    noise = noise_level * np.random.randn(length)
    series = polynomial_trend + noise
    return random_scale(standardize(series))

def generate_exponential_trend(length, base=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if base is None:
        base = np.random.uniform(1.000001, 1.0001)
    x = np.arange(length)
    exponential_trend = base ** x
    noise = noise_level * np.random.randn(length)
    series = exponential_trend + noise
    return random_scale(standardize(series))

def generate_count_series(length, lambda_=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if lambda_ is None:
        lambda_ = np.random.uniform(1, 10)
    series = np.random.poisson(lambda_, length).astype(np.float64)
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_pulse_series(length, period=None, amplitude=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if period is None:
        period = np.random.randint(2, 100)
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5)
    pulse_series = np.zeros(length)
    pulse_indices = np.arange(0, length, period)
    pulse_series[pulse_indices] = amplitude
    noise = noise_level * np.random.randn(length)
    series = pulse_series + noise
    return random_scale(series) / 4

# New ODE/PDE systems
def lorenz_system(state, t, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_series(length, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    t = np.linspace(0, np.random.randint(0, 40), length)
    state0 = np.random.rand(3)
    sigma = np.random.uniform(4, 16)
    beta = np.random.uniform(1, 3)
    rho = np.random.uniform(2, 35)
    states = odeint(lorenz_system, state0, t, args=(sigma, beta, rho))
    series = states[:, 0]  # Use x-component
    noise = noise_level * np.random.randn(length)
    series = series + noise
    return random_scale(standardize(series))

def van_der_pol_oscillator(state, t, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def generate_van_der_pol_series(length, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    t = np.linspace(0, np.random.randint(0, 40), length)
    state0 = np.random.rand(2)
    mu = np.random.uniform(1, 5)
    states = odeint(van_der_pol_oscillator, state0, t, args=(mu,))
    series = states[:, 0]  # Use x-component
    noise = noise_level * np.random.randn(length)
    series = series + noise
    return random_scale(standardize(series))

def rossler_system(state, t, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

def generate_rossler_series(length, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    t = np.linspace(0, np.random.randint(0, 40), length)
    state0 = np.random.rand(3)
    a = np.random.uniform(0.1, 0.5)
    b = np.random.uniform(0.1, 0.5)
    c = np.random.uniform(4, 8)
    states = odeint(rossler_system, state0, t, args=(a, b, c))
    series = states[:, 0]  # Use x-component
    noise = noise_level * np.random.randn(length)
    series = series + noise
    return random_scale(standardize(series))

def sir_system(state, t, beta, gamma):
    S, I, R = state
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def generate_sir_series(length, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    t = np.linspace(0, np.random.randint(160, 480), length)
    state0 = np.random.rand(3)
    beta = np.random.uniform(0.01, 0.1)
    gamma = np.random.uniform(0.005, 0.015)
    states = odeint(sir_system, state0, t, args=(beta, gamma))
    series = states[:, 1]  # Use I-component (Infected)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    return random_repeat(random_scale(standardize(series)), np.random.randint(1, 5))

def fitzhugh_nagumo_system(state, t, a, b, c):
    V, W = state
    dVdt = c * (V - V**3 / 3 + W)
    dWdt = -1 / c * (V - a + b * W)
    return [dVdt, dWdt]

def generate_fitzhugh_nagumo_series(length, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    t = np.linspace(0, np.random.randint(0, 80), length)
    state0 = np.random.rand(2)
    a = np.random.uniform(0.1, 0.9)
    b = np.random.uniform(0.8, 1.0)
    c = np.random.uniform(1, 8)
    states = odeint(fitzhugh_nagumo_system, state0, t, args=(a, b, c))
    series = states[:, 0]  # Use V-component
    noise = noise_level * np.random.randn(length)
    series = series + noise
    return random_scale(standardize(series))

def generate_ar_process(length, ar_params=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if ar_params is None:
        ar_params = np.random.uniform(-0.5, 0.5, size=np.random.randint(1, 8))
    ar = np.r_[1, -ar_params]
    ma = np.r_[1]
    ar_process = ArmaProcess(ar, ma)
    series = ar_process.generate_sample(nsample=length)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_ma_process(length, ma_params=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if ma_params is None:
        ma_params = np.random.uniform(-0.5, 0.5, size=np.random.randint(1, 8))
    ar = np.r_[1]
    ma = np.r_[1, ma_params]
    ma_process = ArmaProcess(ar, ma)
    series = ma_process.generate_sample(nsample=length)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_arma_process(length, ar_params=None, ma_params=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if ar_params is None:
        ar_params = np.random.uniform(-0.5, 0.5, size=np.random.randint(1, 8))
    if ma_params is None:
        ma_params = np.random.uniform(-0.5, 0.5, size=np.random.randint(1, 8))
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arma_process = ArmaProcess(ar, ma)
    series = arma_process.generate_sample(nsample=length)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_arima_process(length, order=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if order is None:
        order = (np.random.randint(1, 5), np.random.randint(0, 2), np.random.randint(1, 5))
    ar_params = np.random.uniform(-0.5, 0.5, size=order[0]) if order[0] > 0 else []
    ma_params = np.random.uniform(-0.5, 0.5, size=order[2]) if order[2] > 0 else []
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arima_process = ArmaProcess(ar, ma)
    series = arima_process.generate_sample(nsample=length)
    series = np.cumsum(series) if order[1] > 0 else series
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_sarima_process(length, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    model = sm.tsa.SARIMAX(np.zeros(length), order=order, seasonal_order=seasonal_order)
    fit_result = model.fit(disp=False, maxiter=100, method='lbfgs')
    series = model.simulate(nsimulations=length, params=fit_result.params)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_garch_process(length, omega=None, alpha=None, beta=None, noise_level=0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if omega is None:
        omega = np.random.uniform(0.1, 0.5)
    if alpha is None:
        alpha = np.random.uniform(0.1, 0.3)
    if beta is None:
        beta = np.random.uniform(0.1, 0.6)
    
    series = np.zeros(length)
    var = np.zeros(length)
    var[0] = omega / (1 - alpha - beta)
    
    for t in range(1, length):
        var[t] = omega + alpha * (series[t-1] ** 2) + beta * var[t-1]
        series[t] = np.random.normal(0, np.sqrt(var[t]))
    
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_markov_process(length, num_states=None, noise_level=0, state_duration=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    if state_duration is None:
        state_duration = np.random.randint(1, 200)
    if num_states is None:
        num_states = np.random.randint(2, 6)
    trans_matrix = np.random.dirichlet(np.ones(num_states), size=num_states)
    states = np.arange(num_states)
    current_state = np.random.choice(states)
    series = []
    while len(series) < length:
        state_len = np.random.randint(1, state_duration + 1)
        series.extend([current_state] * state_len)
        if len(series) >= length:
            break
        current_state = np.random.choice(states, p=trans_matrix[current_state])
    series = np.array(series[:length]).astype(float)
    noise = noise_level * np.random.randn(length)
    series = series + noise
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_hawkes_process(length, mu=0.01, alpha=0.1, beta=0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    events = []
    t = 0
    while t < length:
        u = np.random.uniform(0, 1)
        t += -np.log(u) / mu
        if t >= length:
            break
        events.append(t)
        mu += alpha * np.exp(-beta * t)
    series = np.zeros(length)
    series[np.array(events).astype(int)] = 1
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_poisson_process(length, lambda_=1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    series = np.random.poisson(lambda_, length).astype(float)
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_geometric_brownian_motion(length, mu=0.1, sigma=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = 1 / length
    series = np.zeros(length)
    series[0] = 1
    for t in range(1, length):
        series[t] = series[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal(0, np.sqrt(dt)))
    return random_scale(standardize(series))

def generate_ornstein_uhlenbeck_process(length, theta=0.15, mu=0, sigma=0.3, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = 1 / length
    series = np.zeros(length)
    series[0] = np.random.normal()
    for t in range(1, length):
        series[t] = series[t-1] + theta * (mu - series[t-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
    series = random_crop_interpolate(series, max_factor=10)
    return random_scale(standardize(series))

def generate_fractional_brownian_motion(length, hurst=0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    increments = np.random.normal(0, 1, size=length)
    series = np.zeros(length)
    for t in range(1, length):
        series[t] = np.sum(increments[:t] * (t - np.arange(t))**(hurst - 0.5))
    return random_scale(standardize(series))

def random_shift(series, shift_max):
    shift = np.random.randint(0, shift_max)
    return np.roll(series, shift)

def random_stretch(series, stretch_factor):
    length = len(series)
    indices = np.linspace(0, length - 1, length)
    stretched_indices = np.linspace(0, length - 1, int(length * stretch_factor))
    stretched_series = np.interp(indices, stretched_indices, np.interp(stretched_indices, indices, series))
    return random_scale(standardize(stretched_series))

def random_nonlinear_transform(series):
    choice = np.random.choice(['square', 'sqrt', 'log', 'exp'])
    series = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-8)  # Normalize to [0, 1]
    if choice == 'square':
        return np.square(series)
    elif choice == 'sqrt':
        return np.sqrt(series)
    elif choice == 'log':
        return np.log1p(series)
    elif choice == 'exp':
        return np.expm1(series / 10)  # Scale down to prevent explosion
    return random_scale(series)

def random_clipping(series):
    v_mean = np.mean(series)
    series = series - v_mean
    v_min, v_max = np.min(series), np.max(series)
    series = np.clip(series, v_min*np.random.rand()*2, v_max*np.random.rand()*2)
    series = series + v_mean
    return random_scale(standardize(series))

def average_downsample(arr, new_length):
    n = len(arr)
    factor = n // new_length
    downsampled = np.array([np.mean(arr[i*factor:(i+1)*factor]) for i in range(new_length)])
    return downsampled

def mirror_repeat(array, num_repeats):
    total_length = len(array) * num_repeats
    result = np.empty(total_length, dtype=array.dtype)
    for i in range(num_repeats):
        start_idx = i * len(array)
        if i % 2 == 0:
            result[start_idx:start_idx + len(array)] = array
        else:
            result[start_idx:start_idx + len(array)] = array[::-1]
    return result

def random_repeat(series, num_repeats):
    if np.random.rand() > 0.5:
        series = series[::-1]
    if np.random.rand() > 0.5:
        return average_downsample(np.tile(series, num_repeats), len(series))[:len(series)]
    else:
        return average_downsample(mirror_repeat(series, num_repeats), len(series))[:len(series)]

def random_crop_interpolate(series, max_factor=100):
    if np.random.rand() > 0.5:
        series = series[::-1]
    length = len(series)
    factor = np.random.uniform(1, max_factor)
    cropped_length = int(length / factor)

    # Randomly crop to the reduced length
    start = np.random.randint(0, length - cropped_length)
    cropped_series = series[start:start + cropped_length]

    # Interpolate to original length
    interpolated_series = np.interp(
        np.linspace(0, cropped_length - 1, length),
        np.arange(cropped_length),
        cropped_series
    )

    return interpolated_series

def generate_combined_series(length, num_series=10, noise_level=0.1, random_seed=None, init_padding=5000):
    length = length + init_padding
    if random_seed is not None:
        np.random.seed(random_seed)

    dates = pd.date_range(start="2001-01-01", periods=length, freq='min').strftime('%Y-%m-%d')
    data = {'date': dates}
    
    # Generate different types of series
    series_list = []
    p = np.array([0.15, 0.75, 0.5, 0.5, 
                  0.5, 1.0, 0.75, 0.5, 
                  0.01, 0.01, 0.05, 
                  0.05, 0.05, 0.05, 0.05, 
                  0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 
                  0.05, 0.05, 0.05, 
                  0.05, 0.15])
    p = p / p.sum()
    choices = []
    for _ in range(num_series):
        generators = [
            'random_walk', 'sine_wave', 'cosine_wave', 'square_wave',
            'sawtooth_wave', 'linear_trend', 'polynomial_trend', 'exponential_trend',
            'count_series', 'pulse_series', 'lorenz_series',
            'van_der_pol_series', 'rossler_series', 'sir_series', 'fitzhugh_nagumo_series',
            'ar_process', 'ma_process', 'arma_process', 'arima_process', 'garch_process', 'markov_process',
            'hawkes_process', 'geometric_brownian_motion', 'ornstein_uhlenbeck_process', 
            'fractional_brownian_motion', 'poisson_process'
        ]
        choice = np.random.choice(generators, p=p)
        choices.append(choice)
        if choice == 'random_walk':
            series = generate_random_walk(length, noise_level * np.abs(np.random.randn()), random_seed)
        elif choice == 'sine_wave':
            series = generate_sine_wave(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'cosine_wave':
            series = generate_cosine_wave(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'square_wave':
            series = generate_square_wave(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'sawtooth_wave':
            series = generate_sawtooth_wave(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'linear_trend':
            series = generate_polynomial_trend(length, degree=1, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'polynomial_trend':
            series = generate_polynomial_trend(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'exponential_trend':
            series = generate_exponential_trend(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'count_series':
            series = generate_count_series(length, random_seed=random_seed)
        elif choice == 'pulse_series':
            series = generate_pulse_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'lorenz_series':
            series = generate_lorenz_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'van_der_pol_series':
            series = generate_van_der_pol_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'rossler_series':
            series = generate_rossler_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'sir_series':
            series = generate_sir_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'fitzhugh_nagumo_series':
            series = generate_fitzhugh_nagumo_series(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'ar_process':
            series = generate_ar_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'ma_process':
            series = generate_ma_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'arma_process':
            series = generate_arma_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'arima_process':
            series = generate_arima_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'sarima_process':
            series = generate_sarima_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'garch_process':
            series = generate_garch_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'markov_process':
            series = generate_markov_process(length, noise_level=noise_level * np.abs(np.random.randn()), random_seed=random_seed)
        elif choice == 'hawkes_process':
            series = generate_hawkes_process(length, random_seed=random_seed)
        elif choice == 'poisson_process':
            series = generate_poisson_process(length, random_seed=random_seed)
        elif choice == 'geometric_brownian_motion':
            series = generate_geometric_brownian_motion(length, random_seed=random_seed)
        elif choice == 'ornstein_uhlenbeck_process':
            series = generate_ornstein_uhlenbeck_process(length, random_seed=random_seed)
        elif choice == 'fractional_brownian_motion':
            series = generate_fractional_brownian_motion(length, random_seed=random_seed)
        try:
            pre_trans = np.random.rand()
            if pre_trans > 2/3:
                series = random_repeat(series, num_repeats=np.random.randint(1, 3))
            elif pre_trans > 1/3:
                series = random_crop_interpolate(series, max_factor=3)
            series_list.append(series)
        except:
            print(f'Warning: choice {choice} failed')
            series_list.append(np.zeros(length))
        series = np.clip(series, -25, 25)
        
    
    # Combine series and add interactions between series
    for i in range(num_series):
        combined_series = np.random.uniform(-2, 2)*series_list[i]
        if i > 0:
            for _ in range(np.random.randint(0, min(np.random.randint(1, 5), num_series))):  # Combine with 1 to 3 other series
                interaction_factor = np.random.uniform(-2, 2)
                interaction_series = series_list[np.random.randint(0, i)]
                transform_choice = np.random.choice(['notrans', 'shift', 'shift', 
                                                     'nonlinear', 'repeat', 'multiply', 'clip', 'interpolate'])
                if transform_choice == 'notrans':
                    pass
                elif transform_choice == 'shift':
                    interaction_series = random_shift(interaction_series, shift_max=init_padding)
                elif transform_choice == 'nonlinear':
                    interaction_series = standardize(random_nonlinear_transform(interaction_series))
                elif transform_choice == 'repeat':
                    interaction_series = random_repeat(interaction_series, num_repeats=np.random.randint(2, 4))
                elif transform_choice == 'multiply':
                    interaction_series = standardize(standardize(interaction_series) * combined_series) #* np.random.uniform(-2.1, 2.1)
                elif transform_choice == 'clip':
                    interaction_series = random_clipping(interaction_series)
                elif transform_choice == 'interpolate':
                    interaction_series = random_crop_interpolate(interaction_series, max_factor=4)
                combined_series += interaction_factor * interaction_series
        combined_series += noise_level * np.random.randn(length)
        data[f's{i+1}_{choices[i]}'] = standardize(random_scale(combined_series))  # Standardize and scale the combined series
        if np.random.rand() > 0.5:
            data[f's{i+1}_{choices[i]}'] = data[f's{i+1}_{choices[i]}'] + np.clip(np.random.randn(), -1.5, 1.5)
        data[f's{i+1}_{choices[i]}'] = np.clip(data[f's{i+1}_{choices[i]}'], -np.random.randint(5, 15), np.random.randint(5, 15))

    df = pd.DataFrame(data).iloc[init_padding:].fillna(0).round(3)
    
    return df



def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
              
def generate_and_save_dataset(args):
    i, = args
    length = np.random.randint(2500, 35000)
    num_series = np.random.randint(2, 300)
    noise_level = 0.001 * np.abs(np.random.randn())
    df = generate_combined_series(length, num_series, noise_level, None)
    os.makedirs("../dataset/ICL_pretrain_1", exist_ok=True)
    df.to_csv(f"../dataset/ICL_pretrain_1/ICL_{i}.csv", index=False)

if __name__ == '__main__':
    n_ds = 15000
    with Pool() as pool:
        pool.map(generate_and_save_dataset, [(i,) for i in range(n_ds)])

    # future_pool = list(set(list(range(1, 17)) + list(np.linspace(1, 1000, 51).astype(int)) + [24, 36, 48, 60, 96, 192, 336, 720]))
    # configs = ",".join([f"[1.0,{ft}]ICL_pretrain/ICL_{i}.csv" for ft in future_pool for i in range(n_ds)]) + ',[1.0,96]ETTh2.csv'
    # write_string_to_file('pretrain_config.txt', configs)