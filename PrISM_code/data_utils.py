import pandas as pd
import numpy as np
import xarray as xr


def check_input_zarr_variable(datas):
    sel_vars = [var for var in datas.data_vars if len(datas[var].shape) > 0]
    if len(sel_vars) == 1:
        return sel_vars[0]
    elif len(sel_vars) >= 1:
        raise (
            f"PROBLEM with the input data. The zarr file has 2 possible variables: {sel_vars}.\nThere should be only one to use!"
        )
    elif len(sel_vars) == 0:
        raise (
            f"PROBLEM with the input data. NO useful variables found in the .zarr file."
        )


def from_xarray_to_pandas(data, column=None):
    data = data.to_dataframe()
    if column is not None:
        data = data[[column]]
    data = (
        data.reset_index().pivot(index="time", columns=["x", "y"]).droplevel(0, axis=1)
    )
    # reduce columns to one level with coordinates separated by '_'
    data.columns = [f"{x}_{y}" for x, y in data.columns.to_list()]
    return data


def from_pandas_to_xarray(dataframe, name_var="SM"):
    if isinstance(dataframe, list):
        if not isinstance(name_var, list):
            name_var = [f"{name_var}_{it}" for it in range(len(dataframe))]
        dd = []
        for it, df in enumerate(dataframe):
            dd.append(from_pandas_to_xarray(df, name_var=name_var[it]))
        return xr.merge(dd)

    # step 1 - transform columns from single values of 'x_y' to MultiIndex of [x,y]
    idx = pd.MultiIndex.from_tuples([x.split("_") for x in dataframe.columns])
    idx = idx.set_levels([idx.levels[0].astype(float), idx.levels[1].astype(float)])
    dataframe.columns = idx
    # step 2 - invert pivoting - from table to single columns with three MultiIndex ['time','y','x']
    dataframe2 = (
        dataframe.stack()
        .stack()
        .reset_index(name=name_var)
        .rename(columns={"level_0": "time", "level_1": "y", "level_2": "x"})
    )
    # step 3 - from dataframe to xarray.Dataset
    dataset = dataframe2.set_index(["time", "y", "x"]).to_xarray().astype(float)
    return dataset
