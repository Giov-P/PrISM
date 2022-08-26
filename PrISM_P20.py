import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm
import matplotlib.pyplot as plt

def antecedent_precipitation_index(
    sm_t0 : float,      # soil moisture (m3/m3)
    sm_res: float,     # residual soil moisture (m3/m3)
    dt    : float,         # time step (h)      
    tau   : float,        # soil moisture drying-out velocity (h)
    p_t1  : float,        # precipitation (mm)
    d_soil: float ,    # soil thickness (mm)  
    sm_sat: float = 0.45,     # saturated soil moisture (m3/m3) according to Pellarin 
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
    sm_t1 = (sm_t0 - sm_res)*np.exp(-dt/tau) + (sm_sat - (sm_t0-sm_res))*(1-np.exp(-p_t1/d_soil)) + sm_res
    
    return sm_t1
########################################################################################################
# According to Pellarin et al., 2013
def calcualte_sm_sat_P2013(sand_percentage):
    """ 
    Calculate sm saturated based on sand percentage, according to Pellarin et al., 2013. Parameters from the ISBA LSM model.
    https://www.sciencedirect.com/science/article/abs/pii/S0034425713001387
    """
    sm_sat = 0.1*(-108 * sand_percentage + 494.305)  
    return sm_sat/100
def calculate_tau_P2013(clay_fraction):
    """ 
    Calculate tau (soil moisture drying-out velocity) based on clay fraction in the soil (from 0 to 1) according to Pellarin et al., 2013.
    If clay fraction is too low tau becomes lower than 60 h and it is corrected to keep that value.
    https://www.sciencedirect.com/science/article/abs/pii/S0034425713001387
    """
    tau = 32*np.log(clay_fraction)+174 
    return np.where(tau<60,60, tau)

########################################################################################################
#according to Pellarin et al., 2020
def calculate_tau_P2020(t_air):
    """ 
    Calculate tau (soil moisture drying-out velocity) based on air temperature according to Pellarin et al., 2020.
    https://www.mdpi.com/2072-4292/12/3/481/htm#B50-remotesensing-12-00481
    """
    tau = 400 - (350/(1+np.exp(-0.1*(t_air-7.5))))
    return tau
def calcualte_d_soil_P2020(mean_ndvi):
    """ 
    Calculate d_soil (Soil thickness (mm)) based on average NDVI value of the area according to Pellarin et al., 2020.
    https://www.mdpi.com/2072-4292/12/3/481/htm#B50-remotesensing-12-00481
    """
    d_soil = 120 - 80 / (1+(178482301*np.exp(-100*mean_ndvi)))
    return d_soil
def calculate_sm_res_P2020(mean_ndvi, mean_t_air):
    """ 
    Calculate sm_res (residual soil moisture (m3/m3)) based on average NDVI and air temperature value of the area according to Pellarin et al., 2020.
    https://www.mdpi.com/2072-4292/12/3/481/htm#B50-remotesensing-12-00481
    """
    sm_res = 0.04676 + 0.05936 *mean_ndvi-0.00136 * mean_t_air
    return sm_res
########################################################################################################
def unit_test_api():
    """
    UNIT TEST FOR THE FUNCTION 'antecedent_precipitation_index' 
    """
    p_t1=0
    for sm_t0 in np.arange(0,1,0.01):
        sm_res = sm_t0
        for sm_sat in np.arange(0,1,0.2):
            for dt in np.arange(0,10,2):
                for tau in np.arange(1,10,2):
                    for d_soil in np.arange(1,20,2):
                        sm_t1 = antecedent_precipitation_index(sm_t0=sm_t0, sm_res=sm_res, sm_sat=sm_sat, dt=dt, tau=tau, p_t1=p_t1, d_soil=d_soil )
                        assert sm_t1 == sm_t0
                        
                        
def calculate_api_ts(prec:pd.DataFrame, initial_sm:float = 3.7, params_api:dict = {}):
    """
    1st Wrapper: Function to performs API time-series computation for PrISM (Pellarin, 2020).
    INPUTS:
        prec: pandas.DataFrame
            Dataframe/Series that has as index the datetime and as values all the precipitation observation in mm between the dataframe included.
        params_api: dict {str:float}
            Additional params to be included for the calculation of the API [sm_res, dt, tau, p_t1, d_soil, sm_sat] otherwise default values are adopted (from Pellarin et al., 2020, Niger site, Figure 2).
            
    """    
    #check parameters api
    dt = (prec.index[1] - prec.index[0]).components.hours
    params_api_final = {'sm_res': 3, 'sm_sat': 45, 'dt': dt,'tau': 200, 'd_soil': 100}
    params_api_final.update(params_api)
    
    #calculate api time-series
    sm_pred = pd.DataFrame(index = prec.index, columns = prec.columns)
    for it, inn in enumerate(prec.index):
        if it==0:
            sm_pred.loc[inn,:] = initial_sm
        else:
            sm_pred.loc[inn,:] = antecedent_precipitation_index(sm_t0=sm_pred.iloc[it-1,:],
                                                                    p_t1=prec.iloc[it,:],**params_api_final)
    return sm_pred

def perturbate_series(series: pd.Series, n_perturbation: int = 100, min_val:float = 0, max_val:float = 2):
    """
    2nd Wrapper: Function to create a perturbation of a series for the particle filter. 
    Multiply the series with a set of normal distributed numbers,
    with length  = 'n_perturbation', that goes from 'min_val' to 'max_val'.
    """
    # calculate a pandas.Dataframe with index the datetime and columns the perturbation factor
    particle_filter_values = np.random.uniform(min_val,max_val,n_perturbation)
    vv = (np.repeat([series.values],n_perturbation,axis=0).T*particle_filter_values)
    series_perturbed = pd.DataFrame(vv, index = series.index, columns = particle_filter_values)
    
    #sort by increasing values of perturbation (stored in the pandas.Dataframe columns)
    columns_ordered = list(series_perturbed.columns)
    columns_ordered.sort()
    series_perturbed = series_perturbed[columns_ordered]
    
    return series_perturbed

def select_best_trajectories(sm_perturbed:pd.DataFrame, 
                             satellite_sm:pd.Series,
                             n_selection:int = 30):
    """
    3rd Wrapper: Function to compare the perturbed soil moisture trajectories with the satellite observations
    and return the first 'n_selection' number of trajectories with the lowest RMSE. 
    """
    errors =  sm_perturbed.loc[satellite_sm.index] - satellite_sm.values
    rmse = ((errors**2/errors.shape[0]).mean()**0.5).sort_values(ascending = True)
    
    selected_trajectories = list(rmse.iloc[0:n_selection].index)
    
    return selected_trajectories
    
def apply_PrISM(
    prec:pd.DataFrame,
    satellite_sm:pd.DataFrame,
    initial_sm:float = 3.7,
    params_api:dict = {},
               ):
    """
    GENERIC Wrapper: Function to performs all the steps for calculating the corrected precipitation and Soil Moisture from PrISM (Pellarin, 2020).
    INPUTS:
        prec: pandas.DataFrame
            Dataframe/Series that has as index the datetime and as values all the precipitation observation in mm between the dataframe included.
        satellite_sm: pandas.DataFrame
            Dataframe/Series that has as index the datetime and as values all the soil moisture observation.
        params_api: dict {str:float}
            Additional params to be included for the calculation of the API [sm_res, dt, tau, p_t1, d_soil, sm_sat] otherwise default values are adopted (from Pellarin et al., 2020, Niger site, Figure 2).
            
    """
    n_selection = 30
    
    #Calculate first SM guess from incorrect 1st values of precipitation
    sm_guess = calculate_api_ts(prec, initial_sm = initial_sm, params_api = params_api)
    
    # Particle filter applied to precipitation
    prec_perturbed = perturbate_series(prec.iloc[:,0], n_perturbation=100, min_val = 0, max_val = 2)
    sm_perturbed   = calculate_api_ts(prec_perturbed, initial_sm = initial_sm, params_api = params_api)
    
    # Select the Best precipitation and soil moisture trajectory that fits the satellite soil moisture (minimize RMSE)
    selected_trajectories = select_best_trajectories(sm_perturbed, satellite_sm, n_selection)
    selected_prec = prec_perturbed[selected_trajectories].mean(axis=1)                                # best precipitation
    selected_sm   = calculate_api_ts(pd.DataFrame(selected_prec, columns = ['P']), initial_sm = initial_sm, params_api = params_api) # best soil moisture

    return sm_guess, prec_perturbed, sm_perturbed, selected_trajectories, selected_prec, selected_sm
    
# unit_test_api()
# plt.plot(np.arange(0,1,0.01), calcualte_sm_sat(np.arange(0,1,0.01)))