import sympy as sp
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import time
from tqdm import tqdm
import os
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import signal
from datetime import datetime

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

def random_function(x, t):
    f1, f2 = random.uniform(1, 10), random.uniform(1, 10)
    functions = [
        lambda x, t: x**2,
        lambda x, t: sp.sin(f1 * x), 
        lambda x, t: sp.cos(f1 * x), 
        lambda x, t: sp.sin(f1 * t), 
        lambda x, t: sp.cos(f1 * t), 
        lambda x, t: sp.exp(sp.cos(f1 * t)), 
        lambda x, t: sp.sin(f1 * t) * sp.cos(f2 * x),
        lambda x, t: sp.cos(f1 * t) * sp.sin(f2 * x),
        lambda x, t: sp.sin(f1 * x) * sp.cos(f2 * t),
        lambda x, t: sp.sin(f1 * x) * sp.sin(f2 * t),
        lambda x, t: sp.sin(f1 * x + f2 * t),
        lambda x, t: sp.cos(f1 * x + f2 * t),
        lambda x, t: sp.exp(sp.cos(f1 * t + f2 * x)),
    ]
    func = random.choice(functions)
    return func(x, t)

def generate_random_odes(num_variables):
    """ Generate random ODE systems based on symbolic logic to ensure equation validity """
    t = sp.symbols('t')
    variables = sp.symbols(f'x0:{num_variables}')
    functions = [sp.Function(f'x{i}')(t) for i in range(num_variables)]
    equations = []
    
    for i in range(num_variables):
        equation = 0
        num_terms = random.randint(1, 4)  # Increase the number of terms
        for _ in range(num_terms):
            coeff = random.uniform(-0.2, 0.2)  # Strictly limit the range of coefficients
            random_var = random.choice(variables)
            math_func = random_function(random_var, t)
            operation = random.choice([lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a - b])
            equation = operation(equation, coeff * math_func)
        
        # Add cross-coupling terms, periodic terms, and external forcing terms to ensure the system has complex dynamic behavior
        cross_coupling = random.uniform(-0.2, 0.2) * sum(random.uniform(-0.1, 0.1) * v for v in variables if v != variables[i])
        frequency_a = random.uniform(1, 150)
        frequency_b = random.uniform(1, 15)
        external_forcing = random.uniform(-1, 1) * (sp.sin(frequency_a * t) + sp.cos(frequency_b * t))
        damping = random.uniform(0.01, 0.1) * variables[i]
        equation += cross_coupling + external_forcing - damping
        
        derivative = sp.Derivative(functions[i], t)
        ode = sp.Eq(derivative, equation)
        equations.append(ode)
    return equations

def solve_odes(odes, num_variables, t_max=100, steps=5000):
    t = sp.symbols('t')
    variables = sp.symbols(f'x0:{num_variables}')
    rhs = [sp.lambdify((t,) + tuple(variables), ode.rhs, 'numpy') for ode in odes]

    def system(y, t):
        return [rhs_i(t, *y) for rhs_i in rhs]

    t_values = np.linspace(0, t_max, steps)
    y0 = np.random.uniform(-1, 1, size=num_variables)  # Random initial conditions
    solution = odeint(system, y0, t_values, rtol=1e-6, atol=1e-8)
    
    output = np.nan_to_num(np.clip(solution, -np.random.randint(7, 15), np.random.randint(7, 15)))

    return output
    
def visualize_odes_plot(ys, t_max=None, steps=None):
    t_values = np.linspace(0, t_max, steps) if t_max is not None and steps is not None else np.arange(ys.shape[0])
    # Plot the results
    plt.figure(figsize=(12, 8))
    for i in range(ys.shape[-1]):
        plt.plot(t_values, ys[:, i], label=f'x{i}') # Limit the output range
    plt.title('ODE System Solution')
    plt.xlabel('Time t')
    plt.ylabel('Variables')
    plt.legend()
    plt.show()
    
def random_partition_with_limits(X, L, U):
    if L > U:
        raise ValueError("L should not be greater than U")
    if X < L:
        return [X]
    parts = []
    while X > 0:
        if X <= U:
            part = X
        else:
            part = random.randint(L, min(U, X - L))
        parts.append(part)
        X -= part
    return parts

def random_ode_generator():
    # Ensure the number of equations and the number of variables are consistent
    partition_min = 5
    partition_max = 30
    t_max = np.random.randint(15, 400)
    n_steps = np.random.randint(1024, 8192*2)
    num_vars = np.random.randint(3, 100)

    n_ode_groups = random_partition_with_limits(num_vars, partition_min, partition_max)
    print('t={}, step={}, n_vars={}, group={}'.format(t_max, n_steps, num_vars, n_ode_groups), flush=True)
    
    generated_ode_groups = []

    start_time = time.time()
    for n in n_ode_groups:
        random_odes = generate_random_odes(n)
        # for ode in random_odes:
        #     print(ode)
        res = solve_odes(random_odes, n, t_max=t_max, steps=n_steps)
        generated_ode_groups.append(res)

    generated_ode_groups = np.concatenate(generated_ode_groups, axis=1)
    
    end_time = time.time()
    execution_time = end_time - start_time
    # analysis.append([t_max, n_steps, num_vars, n_ode_groups, execution_time])
    return generated_ode_groups

def transform_to_ts_df(x):
    length = x.shape[0]
    dates = pd.date_range(start="2001-01-01", periods=length, freq='min')
    data = {'date': dates}
    df = pd.DataFrame(x)
    cols = df.columns.tolist()
    df['date'] = dates
    df = df[['date'] + cols]
    return df.fillna(0).round(4)

def generate_and_write_ode_csv(to_path):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + generate_random_string(6)
    csv_name = os.path.join(f"data_{timestamp}.csv")
    generated_ode_data_df = transform_to_ts_df(random_ode_generator())
    generated_ode_data_df.to_csv(f"{to_path}/{csv_name}", index=False)

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    def handler(signum, frame):
        raise TimeoutError()
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = None
    finally:
        signal.alarm(0)
    
    return result
    
dataset_path = "../dataset"
to_path = f"{dataset_path}/ODE_pretrain_1"
os.makedirs(to_path, exist_ok=True)

# Use ProcessPoolExecutor to parallelize the generation process
num_tasks = 10000
n_cpus = cpu_count()
with ProcessPoolExecutor(n_cpus-4) as executor:
    futures = [executor.submit(run_with_timeout, generate_and_write_ode_csv, (to_path,), {}, 300) for _ in range(num_tasks)]
    for future in tqdm(as_completed(futures), total=num_tasks):
        try:
            future.result()
        except Exception as e:
            print(f"Task failed with exception: {e}")
