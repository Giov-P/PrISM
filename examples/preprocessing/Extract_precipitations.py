import pandas as pd
import numpy as np
import os
import glob
import tqdm
import geopandas as gpd
import requests
import urllib3
import io
import warnings
warnings.filterwarnings("ignore")
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1'

def weight_av(line, dist):
    """Weighted average function for the time series of the stations"""
    dist=np.array(dist)
    indicator = np.isnan(line)
    if np.sum(dist[~(indicator.values)])!=0:
        value = np.average(line[~(indicator.values)].values, weights=dist[~(indicator.values)])
        return value
    else:
        return 10

def read_time_series_MeteoStation(point, points_stations,variable='Precipitació acumulada diària',csv_files = 'default',max_distance = 30000, method='fast'):
    #PROBLEM: EXCEL is superslow, use csv (one for each location)
    # method can be {"weighted","fast"}
    #           weighted: weighted average of multiple closest stations
    #           fast :    closest station
    
    stations=gpd.read_file(points_stations)
    
    vv = stations.distance(point).sort_values()
    good_index = vv[vv<max_distance].index
    good_weight = vv[vv<=max_distance].values#/np.nansum(vv[vv<max_distance].values)
    
    #read sheets in the excel file
    if csv_files=='default':
        #excel_file=os.path.join(os.path.dirname(points_stations),'DataXEMAStations_catalunya.xlsx')
        csv_files='/home/gio/Desktop/2-IrrigationType/RESULTs/3-Irrigation_data_study/Precipitation/DataXEMA_{}_Station_catalunya.csv'
    
    r = lambda x: (np.where(x=='--',0,x))
    
    if len(good_index)>0:
        if method == 'weighted':
            for ll,ind in tqdm.tqdm(enumerate(good_index),total = len(good_index)):
                sheet_name = stations.loc[ind,'names']
                csv_file = csv_files.format(sheet_name)
                ee = pd.read_csv(csv_file, usecols=[variable],converters={variable:r})
#                 ee = ee[[variable]]
#                 ee[variable]=np.array(ee[variable],dtype=float)
#                 print(csv_file)
                index = pd.DatetimeIndex(pd.read_csv(csv_file, usecols=['Days'],infer_datetime_format=True)['Days'])
                ee.index=index
                ee=ee.rename(columns={'Precipitació acumulada diària':sheet_name})
                if ll==0:
                    #sol = ee#*good_weight[ll]
                    sol= pd.DataFrame(ee,
#                                       index=index,
                                      columns=[sheet_name])
                else:
                    #sol+=ee*good_weight[ll]
#                     sol[ll]=ee
#                     sol2= pd.DataFrame(ee,index=index,columns=[sheet_name])
                    sol=pd.merge(sol,ee,how='outer', left_index=True, right_index=True)
            sol3 = pd.DataFrame(np.array(sol,dtype=float),index=sol.index,columns=sol.columns)
            sol=sol3.apply(weight_av,axis=1,args=(good_weight,))
            #sol.index=index
        if method == 'fast':
            sheet_name = stations.loc[good_index[0],'names']
            print('reading')
            
            csv_file = csv_files.format(sheet_name)
            print(csv_file, sheet_name,variable)
            index = pd.DatetimeIndex(pd.read_csv(csv_file, usecols=['Days'],infer_datetime_format=True)['Days'])
            ee = pd.read_csv(csv_file, usecols=[variable],dtype=float,converters={variable:r})
            ee[variable]=np.array(ee[variable],dtype=float)
            print('finished reading')
            sol = ee[[variable]]
            sol.index=index

    return sol


def read_time_series_MeteoStation_big_shape(shape, points_stations,variable='Precipitació acumulada diària',csv_files = 'default'):
    #PROBLEM: EXCEL is superslow, use csv (one for each location)
    
    # method weighted: weighted average of the closest stations
    #           fast :    closest station
    
    stations = gpd.read_file(points_stations)
    good_index = stations[stations.within(shape)].index
    
    
#     vv = stations.distance(point).sort_values()
#     good_index = vv[vv<max_distance].index
#     good_weight = vv[vv<=max_distance].values#/np.nansum(vv[vv<max_distance].values)
    sol3 = pd.DataFrame([])
    #read sheets in the excel file
    if csv_files=='default':
        #excel_file=os.path.join(os.path.dirname(points_stations),'DataXEMAStations_catalunya.xlsx')
        csv_files='/home/gio/Desktop/2-IrrigationType/RESULTs/3-Irrigation_data_study/Precipitation/DataXEMA_{}_Station_catalunya.csv'
    
    r = lambda x: (np.where(x=='--',0,x))
    
    if len(good_index)>0:
        for ll,ind in tqdm.tqdm(enumerate(good_index),total = len(good_index)):
            sheet_name = stations.loc[ind,'names']
            csv_file = csv_files.format(sheet_name)
            ee = pd.read_csv(csv_file, usecols=[variable],converters={variable:r})

            index = pd.DatetimeIndex(pd.read_csv(csv_file, usecols=['Days'],infer_datetime_format=True)['Days'])
            ee.index=index
            ee=ee.rename(columns={'Precipitació acumulada diària':sheet_name})
            if ll==0:
                sol= pd.DataFrame(ee,
                                  columns=[sheet_name])
            else:
                sol=pd.merge(sol,ee,how='outer', left_index=True, right_index=True)
                
        sol3 = pd.DataFrame(np.array(sol,dtype=float),index=sol.index,columns=sol.columns)
        sol=sol3.mean(axis=1)
        
        
    return sol3

def read_meteo_stations_in_pandas(idStation,date_start="01/01/17",date_end="01/12/2020", variables = None):
    """"
    Download meteo stations in catalunya data from the ruralcat website and return a pandas dataframe.
    INPUTS:
        idStation: int, id of the station
        date_start: str, initial date in the format "DD/MM/YYYY"
        date_end: str, final date in the format "DD/MM/YYYY"
        variables: list of int, list of variables to download
    OUTPUTS:
        df: pandas dataframe, dataframe with the data of the station
    """

    b_ini=[
        "https://ruralcat.gencat.cat/web/guest/agrometeo.estacions?",
        "p_p_id=AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT&",
        "p_p_lifecycle=2&",
        "p_p_state=normal&",
        "p_p_mode=view&",
        "p_p_cacheability=cacheLevelPage&",
        "p_p_col_id=column-1&", #"p_p_col_id=column-2&",
        "p_p_col_pos=2&", #"p_p_col_pos=1&",
        "p_p_col_count=4&",#"p_p_col_count=2&",
        "_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_action=dadesEstacioResource&",
        "_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_action=viewGoEstacion&",
        "_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_dIni=",
              ]                   
    b_ini = ''.join(b_ini)
    initial_date = date_start
    b2 = "&_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_dFi="
    final_date   = date_end
    b3 = "&_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_variables="
    if variables is None:
        variables = [
            35 # Precipitació horaria
        ]
    #     variables = [
#     1000, # Temperatura mitjana diària
#     1001, # Temperatura màxima diària + hora
#     1002, # Temperatura minima diària + hora
#     1601, # Velocitat escalar del vent a 2m
#     1100, # Humitat relativa mitjana diària   
#     1101, # Humitat relativa màxima diària + data   
#     1300, # Precipitació acumulada diària   
#     1400, # Irradiació solar global diària  
#     #1401, #    
#     1700, # Evapotranspiració de referència   
#     ]
    h = ','.join([str(f) for f in variables])
    b4 = "&_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_tipus="
    tipusDades   = "H"#"D"
    b5 ="&_AgrometeoEstacions_WAR_AgrometeoEstacions100SNAPSHOT_idEstaciones="

    url_request = b_ini + initial_date + b2+ final_date + b3 + h + b4 + tipusDades + b5 + str(idStation)
#     print(url_request)
    rr = requests.get(url_request)
    try:
        with io.BytesIO(rr.content) as fh:
            df = pd.io.excel.read_excel(fh)
    except Exception as e:
        print(f'ERROR {e}')
        return rr
    return df