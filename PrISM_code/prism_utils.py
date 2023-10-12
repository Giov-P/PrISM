import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm as ntqdm
from pathlib import Path
import warnings
import pickle
import time
import tqdm
import itertools
import gc

warnings.filterwarnings("ignore")
from itertools import product


def antecedent_precipitation_index(
    sm_t0: float,  # soil moisture (m3/m3)
    sm_res: float,  # residual soil moisture (m3/m3)
    dt: float,  # time step (h)
    tau: float,  # soil moisture drying-out velocity (h)
    p_t1: float,  # precipitation (mm)
    d_soil: float,  # soil thickness (mm)
    sm_sat: float = 0.45,  # saturated soil moisture (m3/m3) according to Pellarin
):
    """
    Formula to calculate the antecedent_precipitation_index (API) as described in Pellarin et al., 2020
    https://www.mdpi.com/2072-4292/12/3/481/htm
    INPUTS:
        sm_t0: float
            Soil moisture (m3/m3) at time t0
        sm_res: float
            Residual soil moisture (m3/m3)
        dt: float
            Time step (h) used for the simulation (difference between t1 and t0)
        tau: float
            Soil moisture drying-out velocity (h)
        p_t1: float
            Precipitation (mm) cumulative during the period dt
        d_soil: float
            Soil thickness (mm)
        sm_sat: float
            Saturated soil moisture (m3/m3) default is 0.45 (according to Pellarin et al., 2020 for Africa).
    OUTPUT:
        sm_t1: float
            Modelled soil moisture (m3/m3) at time t1

    """
    sm_t1 = (
        (sm_t0 - sm_res) * np.exp(-dt / tau)
        + (sm_sat - (sm_t0 - sm_res)) * (1 - np.exp(-p_t1 / d_soil))
        + sm_res
    )

    #     sm_t1 = (sm_t0 - sm_res)*np.exp(-dt/tau) + tau/dt*p_t1/d_soil*(1-np.exp(-dt/tau)) + sm_res
    return sm_t1


def inverse_antecedent_precipitation_index(
    sm_t0: float,  # soil moisture at t0 (m3/m3)
    sm_res: float,  # residual soil moisture (m3/m3)
    dt: float,  # time step (h)
    tau: float,  # soil moisture drying-out velocity (h)
    sm_t1: float,  # soil moisture at t1 (m3/m3)
    d_soil: float,  # soil thickness (mm)
    sm_sat: float = 0.45,  # saturated soil moisture (m3/m3) according to Pellarin
):
    """
    INVERSE Formula to calculate the Precipitation from the soil moisture inverting the API
    from the formula as described in Pellarin et al., 2020
    https://www.mdpi.com/2072-4292/12/3/481/htm
    INPUTS:
        sm_t0: float
            Soil moisture (m3/m3) at time t0
        sm_res: float
            Residual soil moisture (m3/m3)
        dt: float
            Time step (h) used for the simulation (difference between t1 and t0)
        tau: float
            Soil moisture drying-out velocity (h)
        sm_t1: float
            Soil moisture (m3/m3) at time t1
        d_soil: float
            Soil thickness (mm)
        sm_sat: float
            Saturated soil moisture (m3/m3) default is 0.45 (according to Pellarin et al., 2020 for Africa).
    OUTPUT:
        sm_t1: float
            Modelled soil moisture (m3/m3) at time t1
    """
    delta_sm0 = sm_t0 - sm_res
    pr_t1 = -d_soil * np.log(
        1 - ((sm_t1 - (delta_sm0 * np.exp(-dt / tau) + sm_res)) / (sm_sat - delta_sm0))
    )
    return pr_t1


def calculate_dt_series(series):
    result = series.copy()
    result.loc[series.dropna().index] = (
        pd.Series(series.dropna().index).diff(1) / pd.Timedelta("1H")
    ).values
    return result


def calcualte_precedent_smvalues(series):
    result = series.copy()
    result.loc[series.dropna().index] = series.dropna().shift(1).values
    return result


def move_values_deltas(series_in, deltas):
    series_out = series_in.dropna().copy()
    dts = deltas.loc[:, series_in.name].dropna().copy()
    series_out.index = [
        f - pd.Timedelta(f"{dtirr}H") for f, dtirr in zip(dts.index, dts.values)
    ]
    series_out = series_out.reindex(series_in.index)
    return series_out


def select_initial_sm(series):
    return series.dropna().iloc[0] if len(series.dropna()) > 0 else np.nan


# according to Pellarin et al., 2020
def calculate_tau_P2020(t_air):
    """
    Calculate tau (soil moisture drying-out velocity) based on air temperature according to Pellarin et al., 2020.
    https://www.mdpi.com/2072-4292/12/3/481/htm#B50-remotesensing-12-00481
    """
    tau = 400 - (350 / (1 + np.exp(-0.1 * (t_air - 7.5))))
    return tau
