import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, box
from tqdm.auto import tqdm as ntqdm
import time
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

plt.style.use("bmh")
from pathlib import Path

from PrISM_light import IRRI_PRISM_light


def cdf_matching_fun(model_sm, sat_sm):
    comm_index = model_sm.index.intersection(sat_sm.index)
    # stats
    meansat = sat_sm.loc[comm_index].mean()
    stdsat = sat_sm.loc[comm_index].std()
    meanprm = model_sm.loc[comm_index].mean()
    stdprm = model_sm.loc[comm_index].std()
    # params
    p2 = stdprm / stdsat
    p1 = meanprm - p2 * meansat
    return p1, p2


def modify_winter(df, factor=2):
    return factor * df.copy()


def modify_winter_minmax(df, factor=2):
    dfmin = df.min()
    dfnew = df.copy()
    return (dfnew - dfmin) * factor + dfmin


def modify_winter_percentile(df, factor=2, percentile_v=0.2):
    dfmin = df.quantile(percentile_v)
    dfnew = df.copy()
    return (dfnew - dfmin) * factor + dfmin


def prism_orchestrator(
    precipitation_dry=None,
    satellite_sm_dry=None,
    tau_values_dry=None,
    precipitation_wet=None,
    satellite_sm_wet=None,
    tau_values_wet=None,
    t1_calib="20160101",
    t2_calib="20190101",
    t1="20170315",
    t2="20171101",
    sm_res_v=5,
    sm_sat_v=45,
    d_soil_v=40,
    dt_v=3,
    tau_factor_v=1,
    tau_function=modify_winter,
    p1_v=None,
    p2_v=None,
    len_window=5,
    return_irr_ts=False,
    show_bars=False,
    SPRINKLER_ASSUMPTION=True,
):
    new_tau_dry = tau_function(tau_values_dry, factor=tau_factor_v)
    new_tau_dry = new_tau_dry.where(
        new_tau_dry >= 10, 10
    )  # prevent negative tau values
    params_api_dry = {
        "sm_res": pd.Series([sm_res_v], index=precipitation_dry.columns),
        "sm_sat": pd.Series([sm_sat_v], index=precipitation_dry.columns),
        "tau": new_tau_dry,
        "d_soil": d_soil_v,
    }

    new_tau_wet = tau_function(tau_values_wet, factor=tau_factor_v)
    new_tau_wet = new_tau_wet.where(
        new_tau_wet >= 10, 10
    )  # prevent negative tau values
    params_api_wet = {
        "sm_res": pd.Series(
            [sm_res_v] * satellite_sm_wet.shape[1], index=precipitation_wet.columns
        ),
        "sm_sat": pd.Series(
            [sm_sat_v] * satellite_sm_wet.shape[1], index=precipitation_wet.columns
        ),
        "tau": new_tau_wet,
        "d_soil": d_soil_v,
    }

    sminit = satellite_sm_dry.loc[t1_calib:t2_calib].iloc[5, :]

    IP_dry = IRRI_PRISM_light(
        precipitation=precipitation_dry.loc[t1_calib:t2_calib].loc[sminit.name : :],
        satellite_sm=satellite_sm_dry.loc[t1_calib:t2_calib].loc[sminit.name : :],
        params_api=params_api_dry.copy(),
        verbose=show_bars,
        window=len_window,
    )

    ### CDF matching
    sm_prism_dry = IP_dry.calculate_api(
        prec=precipitation_dry.loc[t1_calib:t2_calib].loc[sminit.name : :],
        initial_sm=sminit,
        dt=dt_v,
        params_api=params_api_dry.copy(),
    )

    p1, p2 = cdf_matching_fun(sm_prism_dry, satellite_sm_dry)
    print(f"P1 = {p1.values[0]:.02f}, P2 = {p2.values[0]:.02f}")

    if (p1_v is None) & (p2_v is None):
        p1_v = p1
        p2_v = p2
    elif p1_v is None:
        p1_v = 0
    elif p2_v is None:
        p2_v = 1

    sm_cdf_matched = p1_v.values[0] + p2_v.values[0] * satellite_sm_wet
    print(sm_cdf_matched)
    #### CALCULATE prism
    IP_wet = IRRI_PRISM_light(
        precipitation=precipitation_wet.loc[t1:t2],
        satellite_sm=sm_cdf_matched.loc[t1:t2],
        params_api=params_api_wet.copy(),
        verbose=show_bars,
        window=len_window,
    )

    all_irr = IP_wet.run(assumption_sprinkler=SPRINKLER_ASSUMPTION)

    mean_precirr_i = (
        IP_wet.est_irr_min.resample(IP_wet.res_sm).sum()
        + IP_wet.est_irr_max.resample(IP_wet.res_sm).sum()
    ) / 2
    delta_precirr_i = (
        IP_wet.est_irr_max.resample(IP_wet.res_sm).sum()
        - IP_wet.est_irr_min.resample(IP_wet.res_sm).sum()
    )
    mean_precirr = mean_precirr_i.where(
        IP_wet.prec.resample(IP_wet.res_sm).sum() == 0, 0
    )
    delta_precirr = delta_precirr_i.where(
        IP_wet.prec.resample(IP_wet.res_sm).sum() == 0, 0
    )
    sum_model = mean_precirr.iloc[:, 0].sum()

    if return_irr_ts:
        return sum_model, mean_precirr, new_tau_dry, sm_prism_dry, delta_precirr, IP_wet
    else:
        return sum_model
