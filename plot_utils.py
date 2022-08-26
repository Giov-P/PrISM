import rioxarray
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as riomask
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def add_nice_xaxis(ax):
    all_ticks = [(ll.get_text(), ll.get_position()) for ll in ax.xaxis.get_majorticklabels() if (ll.get_text()!='')&(ll.get_position()[0]>=0)]
    months = pd.date_range(start = all_ticks[0][0], end = all_ticks[-1][0], freq='MS', normalize = True)
    labels = [dd.month_name()[0:3] if dd.month_name()[0:3] !='Jan' else dd.month_name()[0:3]+f'\n{dd.year}' for dd in months]
    years = ['' if dd.month_name()[0:3] !='Jan' else f'\n{dd.year}' for dd in months]
    ax.set_xlim(all_ticks[0][1][0],all_ticks[-1][1][0])
    ax.xaxis.set_visible(False)
    dr = pd.date_range(start = all_ticks[0][0], end = all_ticks[-1][0], normalize = True)
    rr= pd.DataFrame([[f'{d.year}-{d.month:02d}'for d in dr],[1]*len(dr)], index =['A','count'], columns = dr).T
    widths_labels = rr.groupby('A').sum()
    widths_labels = (widths_labels/widths_labels.sum()).iloc[:,0].values
    ax.table([labels],
             cellColours=[['w']*len(labels)],
             colWidths = widths_labels,
              cellLoc='center',
              loc='bottom',
              bbox=[0,-0.15,1,0.15],
             fontsize = 34
            )
    return ax