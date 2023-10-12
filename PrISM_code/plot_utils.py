import rioxarray
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask as riomask
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num


def make_nice_hist(datas: pd.DataFrame, final_res="5min", frequency_irr="hour"):
    """
    Function to resample a pandas (DataFrame or Series) to a 'final_res'
    keeping the same values of the initial resolution specified by 'frequency_irr',
    only for plotting purposes.
    INPUT:
        - datas: pd.DataFrame
        - final_res: str
          It can be any string specifying a delta timeframe
        - frequency_irr: str
          It is the initial frequency of the pandas dataframe.
    """

    to_plot = datas.resample(final_res).ffill()

    if frequency_irr == "hour":
        condition = np.tile(to_plot.index.minute != 0, (len(to_plot.columns), 1)).T
        if len(datas.shape) == 1:
            condition = to_plot.index.minute != 0

    if frequency_irr == "3H":
        con = np.invert((to_plot.index.minute == 0) & (to_plot.index.hour == 3))
        condition = np.tile(con, (len(to_plot.columns), 1)).T
        if len(datas.shape) == 1:
            condition = to_plot.index.minute != 0

    elif (frequency_irr == "day") | (frequency_irr == "daily"):
        condition = np.tile(to_plot.index.hour != 0, (len(to_plot.columns), 1)).T

    elif frequency_irr == "month":
        condition = np.tile(to_plot.index.day != 0, (len(to_plot.columns), 1)).T

    to_plot.where(condition, 0, inplace=True, axis=0)
    return to_plot


def add_nice_xaxis(ax, fontsize=None, remove_year=False):
    all_ticks = [
        (ll.get_text(), ll.get_position())
        for ll in ax.xaxis.get_majorticklabels()
        if (ll.get_text() != "") & (ll.get_position()[0] >= 0)
    ]
    start = pd.Timestamp(all_ticks[0][0])
    start = pd.Timestamp(year=start.year, month=start.month, day=1)
    months = pd.date_range(start=start, end=all_ticks[-1][0], freq="MS", normalize=True)
    labels = [
        dd.month_name()[0:3]
        if (dd.month_name()[0:3] != "Jan") | (remove_year)
        else dd.month_name()[0:3] + f"\n{dd.year}"
        for dd in months
    ]
    years = [
        "" if (dd.month_name()[0:3] != "Jan") | (remove_year) else f"\n{dd.year}"
        for dd in months
    ]

    ax.set_xlim(all_ticks[0][1][0], all_ticks[-1][1][0])
    ax.xaxis.set_visible(False)

    dr = pd.date_range(start=all_ticks[0][0], end=all_ticks[-1][0], normalize=True)
    gb = [f"{y}{m:02d}" for m, y in zip(dr.month, dr.year)]
    if len(labels) > 20:  # if more than 20 months create table by YEAR only
        gb = [f"{y}" for y in dr.year]
        labels = list(set(gb))
        labels.sort()

    widths_labels = np.array([len(nn) for kk, nn in dr.groupby(gb).items()])
    widths_labels = widths_labels / widths_labels.sum()
    #     clean_small_labels
    quant = 0.1 * np.mean(np.array(widths_labels))
    filters = [True if f < quant else False for f in widths_labels]
    labels = ["" if ff else ll for ll, ff in zip(labels, filters)]

    the_table = ax.table(
        [labels],
        cellColours=[["w"] * len(labels)],
        colWidths=widths_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[0, -0.15, 1, 0.15],
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
    ll = xt[-1] - xt[0]
    dd = (ll * 1.5 - ll) // 2
    val_kline = np.arange(xt[0] - dd, xt[-1] + dd)
    blk_line = ax.plot(val_kline, val_kline, "k-", zorder=0)
    ax.set_xlim(xt[0], xt[-1])
    ax.set_ylim(xt[0], xt[-1])
    return ax


###### PLOT animations from xarray ####################################
def animation_xrarray_map_and_timeseries(
    xarray,
    ax_map,
    ax_ts,
    shapes_to_plot=None,
    colors_lines=None,
    highlight_summer_months=False,
    date_format="%B %Y",
    kwargs_2dmap={},
    kwargs_lineplot={},
):
    """
    Create an animation with 2 axis, in the first one a 2D map changes over time,
    in the second one, a line plot of a spatial averaged value is shown with a moving bar.
    INPUTS:
        xarray: xarray.DataArray
                Array with 3 coordinates: x,y,time
        ax_map: matplotlib.axes._subplots.AxesSubplot
                axis where to place the 2D map
        ax_ts:  matplotlib.axes._subplots.AxesSubplot
                axis where to place the timeseries
        shapes_to_plot: geopandas.Geodataframe - default:None
                        shapes to use to crop the map and extract the 2D averaged values for the time-series
        colors_lines: List of str, default=None
                      It decides the color of the different lines extracted from te different shapes. if None, tab colors will be used.
        highlight_summer_months: bool, default=False
                                 highlits in light orange the summer months.
        kwargs_2dmap: dict
                    Additional options for Matplotlib for the map plot, to override the default.
    OUTPUT:
        an_func: function
                animation function to pass to the matplotlib.animation.animation.FuncAnimation
        handles: matplotlib.handles
                 handles to be used in a costumized legend for the line plot if needed.
    """
    kwargs_2dmap_fin = dict(
        add_colorbar=True,
        cmap="RdYlBu",
        vmin=0,
        vmax=0.2,
        cbar_kwargs={"extend": "neither"},
    )

    if kwargs_2dmap:
        kwargs_2dmap_fin.update(kwargs_2dmap)
    kwargs_lineplot_fin = {}

    if kwargs_lineplot:
        kwargs_lineplot_fin.update(kwargs_lineplot)

    if shapes_to_plot is None:
        shapes_crop_ts = [box(*xarray.rio.bounds())]  # shapes is the entire map
        colors_lines = ["tab:blue"]
    elif colors_lines is None:
        colors_lines = cycle(
            ["tab:blue", "tab:red", "tab:green", "tab:orange", "grey", "tab:cyan"]
        )
        colors_lines = [next(colors_lines) for f in range(len(shapes_to_plot))]
        shapes_crop_ts = shapes_to_plot.geometry.values
    else:
        shapes_crop_ts = shapes_to_plot.geometry.values
        assert len(colors_lines) == len(shapes_crop_ts)

    # Initial PLOTS
    if shapes_to_plot is not None:
        shapes_to_plot.boundary.plot(ax=ax_map, color=colors_lines, zorder=100)
    cax = xarray[0, :, :].plot(ax=ax_map, **kwargs_2dmap_fin)
    # lines - 2nd axis
    lines = []
    for shps, col in zip(shapes_crop_ts, colors_lines):
        ln = (
            xarray.rio.clip([shps])
            .mean(dim=["x", "y"])
            .plot(ax=ax_ts, color=col, **kwargs_lineplot_fin)
        )
        lmax = xarray.rio.clip([shps]).max(dim=["x", "y"]).values
        lmin = xarray.rio.clip([shps]).min(dim=["x", "y"]).values
        ax_ts.fill_between(xarray.time.values, y1=lmin, y2=lmax, alpha=0.2, color=col)
        lines.append(ln[0])
    if highlight_summer_months:
        start_month = 5
        end_month = 9
        years_to_add_band = [
            f.year for f in pd.DatetimeIndex(xarray.time) if f.month == start_month
        ]
        for year in years_to_add_band:
            vor = ax_ts.axvspan(
                pd.Timestamp(day=1, month=start_month, year=year),
                pd.Timestamp(day=1, month=end_month, year=year),
                alpha=0.2,
                color="orange",
                zorder=0,
            )
    # moving bar
    vl = ax_ts.axvline(
        pd.Timestamp(xarray.coords["time"].values[0]), -0.01, 1.5, c="k", lw=2
    )
    ax_map.set_aspect("equal", "box")

    # Next we need to create a function that updates the values as well as the title.
    def animate(frame):
        # update 2D map
        cax.set_array(xarray[frame, :, :].values.flatten())
        ax_map.set_title(
            f"date = {pd.Timestamp(xarray.coords['time'].values[frame]).strftime(format = date_format)}"
        )
        ax_ts.set_title(
            f"date = {pd.Timestamp(xarray.coords['time'].values[frame]).strftime(format = date_format)}"
        )
        # update moving bar
        vl.set_xdata(pd.Timestamp(xarray.coords["time"].values[frame]))

    if highlight_summer_months:
        handles = [vor] + lines
    else:
        handles = lines
    return animate, handles


def animation_xarray_multiple_variables(
    xrdataset,
    shapes_to_plot,
    labels_names=None,
    colors_lines=None,
    kwards_animation_function={},
):
    """Description To be written, Function to adapt animation for a dataset with multiple variables."""

    #     if colors_lines is None:
    #         colors_lines = cycle(['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'grey', 'tab:cyan'])
    #         colors_lines = [next(colors_lines) for f in range(len(shapes_to_plot))]
    if labels_names is None:
        labels_names = [
            "summer period",
            "tot_area",
            "sm irrigated area",
            "sm not irrigated area",
        ]
    dv = [f for f in xrdataset.data_vars if f != "spatial_ref"]
    fig, axs = plt.subplots(
        2,
        len(dv),
        figsize=(5 * len(dv), 6),
        gridspec_kw={"height_ratios": [2, 1]},
        dpi=100,
    )
    fig.suptitle("Monthly Soil Moisture from Dispatch SMOS+MODIS 1Km")
    all_anims = []
    kwards_animation_function_final = dict(highlight_summer_months=True)
    if kwards_animation_function:
        kwards_animation_function_final.update(kwards_animation_function)
    for it, var in enumerate(dv):
        tas = xrdataset[var].transpose(
            "time", "y", "x"
        )  # so that they all have the same order
        tas = tas.rio.write_crs("EPSG:4326")
        anm, handles = animation_xrarray_map_and_timeseries(
            xarray=tas,
            ax_map=axs[0, it],
            ax_ts=axs[1, it],
            shapes_to_plot=shapes_to_plot,
            colors_lines=colors_lines,
            **kwards_animation_function_final,
        )
        axs[1, it].legend(handles, labels_names)
        all_anims.append(anm)
    fig.tight_layout()

    def an_tot(frame):
        for anm in all_anims:
            anm(frame)

    # Finally, we use the animation module to create the animation.
    ani = animation.FuncAnimation(
        fig,  # figure
        an_tot,  # name of the function above
        frames=tas.shape[0],  # Could also be iterable or list
        interval=200,  # ms between frames
    )
    return ani
