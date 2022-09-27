import rioxarray
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as riomask
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def add_nice_xaxis(ax, fontsize = None):
    all_ticks = [(ll.get_text(), ll.get_position()) for ll in ax.xaxis.get_majorticklabels() if (ll.get_text()!='')&(ll.get_position()[0]>=0)]
    start = pd.Timestamp(all_ticks[0][0])
    start = pd.Timestamp(year = start.year, month = start.month, day = 1)
    months = pd.date_range(start = start, end = all_ticks[-1][0], freq='MS', normalize = True)
    labels = [dd.month_name()[0:3] if dd.month_name()[0:3] !='Jan' else dd.month_name()[0:3]+f'\n{dd.year}' for dd in months]
    years = ['' if dd.month_name()[0:3] !='Jan' else f'\n{dd.year}' for dd in months]
    
    ax.set_xlim(all_ticks[0][1][0],all_ticks[-1][1][0])
    ax.xaxis.set_visible(False)
    
    dr = pd.date_range(start = all_ticks[0][0], end = all_ticks[-1][0], normalize = True)
    gb = [f'{y}{m:02d}' for m,y in zip(dr.month,dr.year)]
    widths_labels = np.array([len(nn) for kk, nn in dr.groupby(gb).items()])
    widths_labels = widths_labels/widths_labels.sum()
    the_table = ax.table([labels],
                         cellColours=[['w']*len(labels)],
                         colWidths = widths_labels,
                         cellLoc='center',
                         loc='bottom',
                         bbox=[0,-0.15,1,0.15]
                        )
    if fontsize is not None:
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(fontsize)
    return ax

def draw_black_identity_line(ax):
    """
    Function to add a black identity line (y=x) in the background in correlation (scatter) plots to have a reference. 
    """
    xt = ax.get_xticks()
    ll = xt[-1]-xt[0]
    dd = (ll*1.5-ll)//2
    val_kline = np.arange(xt[0]-dd, xt[-1]+dd)
    blk_line = ax.plot(val_kline,val_kline,'k-', zorder = 0)
    ax.set_xlim(xt[0], xt[-1])
    ax.set_ylim(xt[0], xt[-1])
    return ax