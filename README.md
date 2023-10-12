# PrISM4Irrigation
This repository contains the code developed to estimate irrigation in a highly cultivated area in Catalunya. 

Results will be soon presented.

the repository is an implementation of PrISM (Precipitation Inferred from Soil Moisture) model to estimate irrigation amounts. The initial model is retrieved from [Pellarin et al., 2020](https://www.mdpi.com/2072-4292/12/3/481/htm) and [Pellarin et al., 2013](https://www.sciencedirect.com/science/article/abs/pii/S0034425713001387).

The repository is structured as follow:

    └── PrISM/
        └── PrISM_code/
        └── examples/

The folder **PrISM_code** contains the skeleton of the algorithm, While **example** contains jupyter notebooks with some indication on how to run the code using as inputs data from **csv** files or **netcdf**/**zarr** rasters.

The methodology mainly requires 3 time series:
- Precipitation time-series
- Observed Soil Moisture
- Temperature time-series (to extract auxiliary parameters)