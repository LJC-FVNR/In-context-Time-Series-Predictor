import numpy as np
from IPython.display import display, Markdown, Latex, HTML
import matplotlib.pyplot as plt
import torch


def vis_channel_forecasting(L_I, X, X_adj, X_true, masks, col_names=None):
    """
    Modified function to plot the time series data for visualization with mask highlighting
    that fills the entire vertical range between vmin and vmax.

    Parameters:
    - L_I (int): Length of the input segment.
    - X (torch.Tensor): Tensor containing the preprocessed input and output of the time series model.
                        Shape: (L_I + L_P, C)
    - X_adj (torch.Tensor): Tensor containing the adjusted predictions for a subset of the series.
                            Shape: (L_P, C_s)
    - X_true (torch.Tensor): Tensor containing the ground truth data.
                             Shape: (L_I + L_P, C_s)
    - masks (torch.Tensor): Tensor containing masking information. Shape: (L_I, C)
    """
    L_P = X.shape[0] - L_I  # Predicted length
    C = X.shape[1]  # Number of channels (time series)
    C_s = X_adj.shape[1]  # Number of adjusted series

    fig, axs = plt.subplots(C, 1, figsize=(10, 2 * C), sharex=True, dpi=72)
    
    X_adj = torch.cat([X[[L_I-1], -C_s:], X_adj], dim=0)
    
    X = X.detach().cpu().numpy()
    X_adj = X_adj.detach().cpu().numpy()
    X_true = X_true.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    
    # Check if there is only one time series (C == 1)
    if C == 1:
        axs = [axs]

    for i in range(C):
        # Plot input and output for each time series
        axs[i].plot(range(L_I), X[:L_I, i], label='Input', color='blue', alpha=0.75)
        axs[i].plot(range(L_I-1, L_I + L_P), X[(L_I-1):, i], label='Output', color='orange', alpha=0.75)

        # If this series has adjusted predictions and ground truth
        if i >= C - C_s:
            axs[i].plot(range(L_I-1, L_I + L_P), X_adj[:, i - (C - C_s)], label='Forecasting', color='green', alpha=0.75)
            axs[i].plot(range(0, L_I + L_P), X_true[:, i - (C - C_s)], label='True', color='red', alpha=0.75)

        # Draw a vertical line to show the separation between input and output
        axs[i].axvline(x=L_I-1, color='gray', linestyle='--')

        # Find the masking point and fill the area
        t_i = np.where(masks[:, i] == 1)[0][0] if 1 in masks[:, i] else L_I
        ymin, ymax = axs[i].get_ylim()
        axs[i].fill_between(range(t_i, L_I), ymin, ymax, color='green', alpha=0.1)

        # Adding legend and labels
        axs[i].legend()
        if col_names is None or C-i > len(col_names):
            axs[i].set_ylabel(f"Series (-{C-i})")
        else:
            axs[i].set_ylabel(f"{col_names[-(C-i)]} (-{C-i})")

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    return fig


'''
input instance:
exp = {'Group.1': {'Model.1': [0.2, 0.3, 0.4, 0.5, 0.6], 'Model.2': [0.1, 0.4, 0.5, 0.6, 0.8], 'Repeat': [0.2, 0.4, 0.6, 0.8, 1]}, 
       'Group.2': {'Model.1': [0.2, 0.3, 0.4, 0.5, 0.6], 'Model.2': [0.1, 0.4, 0.5, 0.6, 0.8], 'Repeat': [0.2, 0.4, 0.6, 0.8, 1]}}
metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'SMAPE']
'''

def render_table(exp, metrics, title="Title", subtitle='Subtitle', 
                                      header_format_func=lambda Model, Group: f'{Model} ({Group})',
                                      result_comparing=True,
                                      result_formatting='{:.2%} ({:.3f})',
                                      result_comparing_pivot_index=-1):
    def header(metric='Metric', models=['TARIMAX', 'Repeat'], groups=['Train', 'Test'], format_func=lambda Model, Group: f'{Model} ({Group})', td_class='tg-54swd'):
        model_col = f'<td class="{td_class}" style="border-bottom:solid;border-width:1px;"><span style="font-weight:700;font-style:normal;color:#000;background-color:#FFF">{{}}</span></td>'
        group_cols = f'''
                     <td class="{td_class}"><span style="font-weight:700;font-style:normal;color:#000;background-color:#FFF">&nbsp;&nbsp;</span></td>
                     {{}}
                     '''
        content = '\n'.join([group_cols.format('\n'.join([model_col.format(format_func(m, g)) for m in models])) for g in groups])
        return f'''
               <tr style="border-top:none !important;">
                <td class="{td_class}" style="border-bottom:solid;border-width:1px;"><span style="font-weight:700;font-style:normal;color:#000;background-color:#FFF">{metric}</span></td>
                {content}
               </tr>
               '''
    def result(metric='SMAPE', result=[[0.2, 0.4], [0.5, 0.6]], footer=True, td_class='tg-i45s', 
               comparing=True, formatting='{:.2%} ({:.3f})', pivot_index=-1):
        result = np.array(result)
        result_min = np.min(result, axis=1, keepdims=True)
        bold_flag = result == result_min
        result_comparing = result/result[:, [pivot_index]]
        result_rendered = [[formatting.format(rc, r) for rc, r in zip(rcs, rs)] for rcs, rs in zip(result_comparing, result)] if comparing else [[formatting.format(r) for r in rs] for rs in result]
        bold = 'font-weight:700;'
        ft = ' style="border-bottom:solid;border-width:1px;"' if footer else ''
        result_col = f'<td class="{td_class}"><span style="font-weight:400;font-style:normal;color:#000;background-color:#FFF;{{}}">{{}}</span></td>'
        group_cols = f'''
                    <td class="{td_class}"><span style="font-weight:400;font-style:normal;color:#000;background-color:#FFF">&nbsp;&nbsp;</span></td>
                    {{}}
                    '''
        content = '\n'.join([group_cols.format('\n'.join([result_col.format(bold if b else '', r) for r, b in zip(res, b_flag)])) for res, b_flag in zip(result_rendered, bold_flag)])
        return f'''
              <tr {ft}>
                <td class="{td_class}"><span style="font-weight:400;font-style:normal;color:#000;background-color:#FFF">{metric}</span></td>
                {content}
              </tr>
              '''
    group_names = list(exp.keys())
    model_names = list(exp[group_names[0]].keys())
    table_content = header(metric='Metric', models=model_names, groups=group_names, format_func=header_format_func)
    for index, m in enumerate(metrics):
        result_list = [[exp[g][mn][index] for mn in model_names] for g in group_names]
        table_content += '\n' + result(metric=m, result=result_list, footer=True if index==len(metrics)-1 else False, 
                                       comparing=result_comparing, formatting=result_formatting, pivot_index=result_comparing_pivot_index)
    span_col = 1 + len(group_names) + len(group_names)*len(model_names)
    table = f"""
    <style type="text/css">
    .tg  {{border-collapse:collapse;border-spacing:0;border-top:none;border-bottom:none}}
    .tg td{{border-color:black;border-style:solid;border-width:1px;font-family:font-family:"Times New Roman", Times, serif !important;font-size:14px;
      overflow:hidden;padding:10px 5px;word-break:normal;;border-top:none;border-bottom:none}}
    .tg th{{border-color:black;border-style:solid;border-width:1px;font-family:font-family:"Times New Roman", Times, serif !important;font-size:14px;
      font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;;border-top:none;border-bottom:none}}
    .tg .tg-54swu{{background-color:#FFF;border-color:inherit;font-weight:bold;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-bottom:none}}
    .tg .tg-54swd{{background-color:#FFF;border-color:inherit;font-weight:bold;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-top:none}}
    .tg .tg-c3ow{{background-color:#FFF;border-color:inherit;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-top:none;border-bottom:none}}
    .tg .tg-i45s{{background-color:#FFF;border-color:inherit;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-top:none;border-bottom:none}}
    .tg .tg-rcip{{background-color:#FFF;border-color:inherit;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-top:none;border-bottom:none}}
    .tg .tg-i45b{{background-color:#FFF;border-color:inherit;text-align:center !important;vertical-align:middle !important;border-left:none;border-right:none;border-top:none;border-bottom:none}}
    .table {{font-family:"Times New Roman", Times, serif !important}}
    </style>
    <table class="tg" style='font-family:"Times New Roman", Times, serif !important'>
    <thead style="border-bottom:none !important">
      <tr>
        <th class="tg-54swu" colspan="{span_col}" style="font-weight:700;font-style:normal;border-top:solid;border-width:1px;border-bottom:none; !important;color:#000;background-color:#FFF">{title}</th>
      </tr>
      <tr>
        <td class="tg-54swu" colspan="{span_col}" style="font-weight:700;font-style:normal;border-top:solid;border-width:0px;border-bottom:none !important;border-top:none !important; padding:0px;color:#000;background-color:#FFF">{subtitle}</th>
      </tr>
    </thead>
    <tbody>
      {table_content}
    </tbody>
    </table>
    """
    return table

def generate_output(*args, **kwargs):
    return HTML(render_table(*args, **kwargs))

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns

'''
def plot_aligned_heatmap(x):
    #x = np.random.randn(20, 20)
    
    fig = plt.figure(figsize=(10, 10), dpi=120)

    p1 = plt.subplot2grid((10,11),(0,0), rowspan=8, colspan=2)
    p2 = plt.subplot2grid((10,11),(0,2), rowspan=8, colspan=8)
    p3 = plt.subplot2grid((10,11),(8,2), rowspan=2, colspan=8)
    p4 = plt.subplot2grid((10,11),(0,10), rowspan=8, colspan=1)

    
    im = p2.imshow(x, cmap=sns.cubehelix_palette(as_cmap=True), aspect="auto")
    fig.colorbar(im, cax=p4)
    plt.subplots_adjust(wspace=2.5, hspace=2.5)
    p1.plot(x.sum(axis=1), range(x.shape[0]))
    p3.plot(range(x.shape[1]), x.sum(axis=0))

    p1.yaxis.set_major_locator(MaxNLocator(integer=True))
    p2.xaxis.set_major_locator(MaxNLocator(integer=True))
    p2.yaxis.set_major_locator(MaxNLocator(integer=True))
    p3.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig
'''
def plot_aligned_heatmap(x):
    #x = np.random.randn(20, 20)
    
    fig = plt.figure(figsize=(12, 12), dpi=120)

    p1 = plt.subplot2grid((20,21),(0,0), rowspan=16, colspan=3)
    p2 = plt.subplot2grid((20,21),(0,3), rowspan=16, colspan=16)
    p3 = plt.subplot2grid((20,21),(16,3), rowspan=3, colspan=16)
    p4 = plt.subplot2grid((20,21),(0,19), rowspan=16, colspan=2)

    
    im = p2.imshow(x, cmap=sns.cubehelix_palette(as_cmap=True, reverse=True), aspect="auto")
    fig.colorbar(im, cax=p4)
    plt.subplots_adjust(wspace=100, hspace=100)
    p1.plot(x.mean(axis=1), range(x.shape[0]), color="#4B4453")
    p1.set_ylim(0, x.shape[0]-1)
    p1.invert_yaxis()
    p3.plot(range(x.shape[1]), x.mean(axis=0), color="#4B4453")
    p3.set_xlim(0, x.shape[1]-1)

    p1.yaxis.set_major_locator(MaxNLocator(integer=True))
    p2.xaxis.set_major_locator(MaxNLocator(integer=True))
    p2.yaxis.set_major_locator(MaxNLocator(integer=True))
    p3.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig

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