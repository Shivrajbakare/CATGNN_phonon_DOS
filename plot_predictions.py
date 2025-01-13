import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl
import matplotlib.pyplot as plt
from ase import Atoms

# utilities
from tqdm import tqdm

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)


# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "0123456789")
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize


# colors for datasets
palette = ['#285fb2', '#f3b557', '#67c791', '#c85c46']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

def plot_predictions(df, idx, title=None):    
    # get quartiles
    i_mse = np.argsort(df.iloc[idx]['mse'])
    ds = df.iloc[idx].iloc[i_mse][['formula', 'prop', 'pred', 'mse']].reset_index(drop=True)
    quartiles = np.quantile(ds['mse'].values, (0.25, 0.5, 0.75, 1.))
    iq = [0] + [np.argmin(np.abs(ds['mse'].values - k)) for k in quartiles]
    
    n = 7
    s = np.concatenate([np.sort(np.random.choice(np.arange(iq[k-1], iq[k], 1), size=n, replace=False)) for k in range(1,5)])
    x = df.iloc[0]['freq']

    fig, axs = plt.subplots(4,n+1, figsize=(13,3.5), gridspec_kw={'width_ratios': [0.7] + [1]*n})
    gs = axs[0,0].get_gridspec()
    
    # remove the underlying axes
    for ax in axs[:,0]:
        ax.remove()

    # add long axis
    axl = fig.add_subplot(gs[:,0])

    # plot quartile distribution
    y_min, y_max = ds['mse'].min(), ds['mse'].max()
    y = np.linspace(y_min, y_max, 500)
    kde = gaussian_kde(ds['mse'])
    p = kde.pdf(y)
    axl.plot(p, y, color='black')
    cols = [palette[k] for k in [2,0,1,3]][::-1]
    qs =  list(quartiles)[::-1] + [0]
    for i in range(len(qs)-1):
        axl.fill_between([p.min(), p.max()], y1=[qs[i], qs[i]], y2=[qs[i+1], qs[i+1]], color=cols[i], lw=0, alpha=0.5)
    axl.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axl.invert_yaxis()
    axl.set_xticks([])
    axl.set_ylabel('MSE')

    fontsize = 12
    cols = np.repeat(cols[::-1], n)
    axs = axs[:,1:].ravel()
    for k in range(4*n):
        ax = axs[k]
        i = s[k]
        ax.plot(x, ds.iloc[i]['prop'], color='black')
        ax.plot(x, ds.iloc[i]['pred'], color=cols[k])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ds.iloc[i]['formula'].translate(sub), fontsize=fontsize, y=0.95)
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(title+'_set_predictions.jpg', dpi=700)
    #if title: fig.suptitle(title, ha='center', y=1., fontsize=fontsize + 4)
    #fig.savefig(title+'_set_predictions.jpg', dpi=600)