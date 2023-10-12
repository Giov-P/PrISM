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
import xarray as xr
import sys

sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / "PrISM_code"))

from prism_orchestrator import prism_orchestrator, modify_winter_percentile
from prism_utils import calculate_tau_P2020
from data_utils import (
    check_input_zarr_variable,
    from_xarray_to_pandas,
    from_pandas_to_xarray,
)


def parallel_prism_orchestrator(inps):
    sum_model, mean_precirr, new_tau, sm_prism_dry, delta_precirr = prism_orchestrator(
        **inps
    )
    return mean_precirr, delta_precirr / 2


if __name__ == "__main__":
    # parameters
    cxd = 0.975  # belianes dry pixel
    cy = 41.595  # belianes dry pixel
    task_division = 32
    n_cores = task_division // 2

    # files
    main_folder = Path().cwd().parent.parent.parent / "DATA"
    shpfileirr = (
        main_folder / "INSITU" / "Urgell" / "SHAPEFILE" / "Urgell_irrigated.geojson"
    )
    shpfilenotirr = main_folder / "INSITU" / "Urgell" / "SHAPEFILE" / "belianes.geojson"

    temperature_file = (
        main_folder
        / "INSITU"
        / "Lleida"
        / "RAW_DATA"
        / "Lleida_Precipitation_agrimeteo"
        / "TEMPERATURE"
        / "MovWinMeanNearest_temp1km_Urgell_48stations_from01012010to31122023.zarr"
    )
    precipitation_file = (
        main_folder
        / "INSITU"
        / "Lleida"
        / "RAW_DATA"
        / "Lleida_Precipitation_agrimeteo"
        / "MovWinMeanNearest_Prec1km_Urgell_48stations_from01012010to31122023.zarr"
    )
    soil_moisture_file = (
        main_folder
        / "SATELLITES"
        / "Lleida"
        / "Lleida_Dispatch_SMAP"
        / "ALLvariables_p21"
        / "Dispatch_Lleida_SMAP_p21_1km_20152023.zarr"
    )

    t_start = f"20160101"  # min date
    t_end = f"20231231"  # max date

    print("reading files")
    # shapefiles
    shpirr = gpd.read_file(shpfileirr).to_crs(4326)
    shpnotirr = gpd.read_file(shpfilenotirr).to_crs(4326)
    # xarrays
    temp_f = xr.open_dataset(temperature_file).sel(time=slice(t_start, t_end))
    prec_f = xr.open_dataset(precipitation_file).sel(time=slice(t_start, t_end))
    sm_tot = (
        xr.open_dataset(soil_moisture_file).sel(time=slice(t_start, t_end)).dispatch_SM
        * 100
    )

    # clean empty SM
    timereal = sm_tot.time[
        sm_tot.isnull().sum(["x", "y"]) != sm_tot.x.shape[0] * sm_tot.y.shape[0]
    ]
    sm_tot = sm_tot.sel(time=timereal)
    sm_tot.name = "SoilMoisture"

    # crop irrigated and dryland
    satellite_sm_dry = sm_tot.sel(x=cxd, y=cy).to_dataframe()[["SoilMoisture"]].dropna()
    satellite_sm_dry = satellite_sm_dry.dropna(how="all", axis=0)
    satellite_sm_dry.columns = ["p1"]
    precipitation_dry = (
        prec_f.sel(x=cxd, y=cy)
        .to_dataframe()[["Precipitation"]]
        .dropna(how="all", axis=1)
    )
    precipitation_dry.columns = ["p1"]
    print("cropping data")
    sm_tot = sm_tot.rio.write_crs(4326).rio.clip(
        [shpirr.geometry[0]] + [shpnotirr.geometry[0]]
    )
    prec_f = prec_f.rio.write_crs(4326).rio.clip(
        [shpirr.geometry[0]] + [shpnotirr.geometry[0]]
    )
    temp_f = temp_f.rio.write_crs(4326).rio.clip(
        [shpirr.geometry[0]] + [shpnotirr.geometry[0]]
    )
    print("converting to pandas")
    satellite_sm = from_xarray_to_pandas(sm_tot, column="SoilMoisture").dropna(
        axis=0, how="all"
    )
    precipitation = (
        from_xarray_to_pandas(prec_f, column=check_input_zarr_variable(prec_f))
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )
    temperature = from_xarray_to_pandas(
        temp_f, column=check_input_zarr_variable(temp_f)
    ).dropna(axis=0, how="all")

    tau_p = calculate_tau_P2020(temperature)
    tau_pr = (
        tau_p.rolling(28, min_periods=1, center=True)
        .mean()
        .resample("1D")
        .mean()
        .reindex(precipitation.index)
        .interpolate()
    )
    tau_d = tau_pr[[f"{cxd}_{cy}"]].rename(columns={f"{cxd}_{cy}": "p1"})

    all_cols = precipitation.dropna(how="all", axis=1).columns

    for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        fileout_irr = Path(
            f"/data/PhDGiovanni/5.IrriEst/DATA/PRISM_outputs/Urgell/IRR_Urgell_fromsensitivitystudy_tau4_{year}.zarr"
        )
        fileout_deltas = Path(
            f"/data/PhDGiovanni/5.IrriEst/DATA/PRISM_outputs/Urgell/DELTA_Urgell_fromsensitivitystudy_tau4_{year}.zarr"
        )

        ccs = 0
        input_list = []
        step = len(all_cols) // task_division
        for it, cc in enumerate(range(step, len(all_cols) + step, step)):
            print(it, len(all_cols[ccs:cc]))
            name_cols = all_cols[ccs:cc]
            input_list.append(
                dict(
                    precipitation_dry=precipitation_dry,
                    satellite_sm_dry=satellite_sm_dry,
                    precipitation_wet=precipitation[name_cols].dropna(
                        how="all", axis=1
                    ),
                    satellite_sm_wet=satellite_sm[name_cols].dropna(how="all", axis=1),
                    t1_calib="20160101",
                    t2_calib="20231031",
                    t1=f"{year}0101",
                    t2=f"{year}1231",
                    sm_res_v=5,
                    sm_sat_v=45,
                    d_soil_v=40,
                    dt_v=3,
                    tau_values_dry=tau_d,
                    tau_values_wet=tau_pr[
                        name_cols
                    ],  # tau_d.rename(columns = {tau_d.columns[0]:"p1"}),
                    tau_factor_v=4,
                    tau_function=modify_winter_percentile,
                    p1_v=None,
                    p2_v=None,
                    return_irr_ts=True,
                    show_bars=True,
                    SPRINKLER_ASSUMPTION=False,
                )
            )
            ccs = cc

        with Pool(n_cores) as p:
            cc = list(
                ntqdm(
                    p.imap_unordered(parallel_prism_orchestrator, input_list),
                    total=len(input_list),
                )
            )

        av_irr_tot = from_pandas_to_xarray(
            pd.concat([f[0] for f in cc], axis=1), name_var="average_irrigation"
        )
        delta_irr_tot = from_pandas_to_xarray(
            pd.concat([f[1] for f in cc], axis=1), name_var="delta_irrigation"
        )
        av_irr_tot.to_zarr(fileout_irr)
        delta_irr_tot.to_zarr(fileout_deltas)
