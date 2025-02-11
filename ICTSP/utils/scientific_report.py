import numpy as np
from IPython.display import display, Markdown, Latex, HTML
import matplotlib.pyplot as plt
import torch

def mts_visualize(pred, true, split_step=720, title='Long-term Time Series Forecasting', dpi=72, col_names=None):
    groups = range(true.shape[-1])
    C = true.shape[-1]
    i = 1
    # plot each column
    f = plt.figure(figsize=(10, 2.1*len(groups)), dpi=dpi)
    f.suptitle(title, y=0.9)
    index = 0
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(true[:, group], alpha=0.75, label='True')
        if type(pred) is list:
            for index, p in enumerate(pred):
                plt.plot(list(range(split_step, true.shape[0])), p[:, group], alpha=0.5, label=f'Pred_{index}')
        else:
            plt.plot(list(range(split_step, true.shape[0])), pred[:, group], alpha=0.75, label='Pred')
        #plt.title(f'S{i}', y=1, loc='right')
        if col_names is None or C-index > len(col_names):
            plt.title(f"Series (-{C-index})", y=1, loc='right')
        else:
            plt.title(f"{col_names[-(C-index)]} (-{C-index})", y=1, loc='right')
        index += 1
        plt.legend(loc='lower left')
        plt.axvline(x=split_step, linewidth=1, color='Purple')
        i += 1
    return f

def mts_visualize_horizontal(pred, true, split_step=720, title='Long-term Time Series Forecasting', dpi=72, width=10, col_names=None):
    groups = range(true.shape[-1])
    C = true.shape[-1]
    i = 1
    # plot each column
    f = plt.figure(figsize=(width, 2.1*len(groups)), dpi=dpi)
    f.suptitle(title, y=0.9)
    index = 0
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(true[:, group], alpha=0.75, label='True')
        if type(pred) is list:
            for index, p in enumerate(pred):
                plt.plot(list(range(split_step, true.shape[0])), p[:, group], alpha=0.75, label=f'Pred_{index}', linestyle=':')
        else:
            plt.plot(list(range(split_step, true.shape[0])), pred[:, group], alpha=0.75, label='Pred')
        #plt.title(f'S{i}', y=1, loc='right')
        if col_names is None or C-index > len(col_names):
            plt.title(f"Series (-{C-index})", y=1, loc='right')
        else:
            plt.title(f"{col_names[-(C-index)]} (-{C-index})", y=1, loc='right')
        index += 1
        plt.legend(ncol=1000, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.axvline(x=split_step, linewidth=1, color='Purple')
        i += 1
    return f