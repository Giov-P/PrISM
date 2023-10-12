from re import S
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

from prism_utils import *


class IRRI_PRISM_light:
    def __init__(
        self,
        precipitation: pd.DataFrame,
        satellite_sm: pd.DataFrame,
        params_api: dict,
        window: int = 5,
        verbose: bool = None,
    ):
        ## check inputs
        self.sat_sm = self.check_dataframe(dataframe=satellite_sm)
        self.prec = self.check_dataframe(dataframe=precipitation)
        self.window = window
        self.res_prec = self.calculate_resolution(self.prec)
        self.res_sm = self.calculate_resolution(self.sat_sm)
        self.params_api = params_api
        self.clean_inputs()
        self.verbose = True if verbose is None else verbose

    ### 1. FUNCTIONS for INITIAL CLEANING
    def check_dataframe(self, dataframe: pd.DataFrame):
        """Function to check the dataframe types."""
        if not isinstance(dataframe, pd.DataFrame):
            if not isinstance(dataframe, pd.Series):
                raise ValueError(
                    f"The input dataframe is of type {type(dataframe)}!\nPlease check that the dataframe is a pandas.dataframe or a pandas.Series."
                )
            else:
                return pd.DataFrame(dataframe)
        else:
            return dataframe.copy()

    def calculate_resolution(self, dataframe: pd.DataFrame):
        """Function to calculate the temporal resolution of a pandas dataframe (or series)"""
        resolution_dataframe = (
            pd.Series(dataframe.dropna(how="all", axis=0).index).diff(1).min()
        )
        return resolution_dataframe

    def clean_inputs(self):
        """Function to check all the inputs and allign them temporally (plus it fill the nan of precipitation with 0)."""
        start_date = max(self.sat_sm.index[0], self.prec.index[0])
        end_date = min(self.sat_sm.index[-1], self.prec.index[-1])
        #         self.prec   = self.prec.loc[start_date:end_date].fillna(0)
        new_index_prec = pd.date_range(
            self.prec.index[0], self.prec.index[-1], freq=self.res_prec
        )
        self.prec = self.prec.reindex(new_index_prec).fillna(0)
        self.sat_sm = self.sat_sm.loc[start_date:end_date]
        # clean params_api
        if "tau" in self.params_api.keys():
            if isinstance(self.params_api["tau"], float) | isinstance(
                self.params_api["tau"], int
            ):
                self.params_api["tau"] = (
                    np.ones(self.prec.shape) * self.params_api["tau"]
                )
                self.params_api["tau"] = pd.DataFrame(
                    self.params_api["tau"],
                    index=self.prec.index,
                    columns=self.prec.columns,
                )
        if isinstance(self.params_api["sm_res"], float) | isinstance(
            self.params_api["sm_res"], int
        ):
            self.params_api["sm_res"] = pd.Series(
                [self.params_api["sm_res"]] * len(self.prec.columns),
                index=self.prec.columns,
            )
        if isinstance(self.params_api["sm_sat"], float) | isinstance(
            self.params_api["sm_sat"], int
        ):
            self.params_api["sm_sat"] = pd.Series(
                [self.params_api["sm_sat"]] * len(self.prec.columns),
                index=self.prec.columns,
            )
        return start_date, end_date

    ### 2. PrISM FUNCTIONS
    def run(
        self,
        n_best_values: int = 1,
        assumption_sprinkler=False,
        filtering_1d_nans=False,
        text_bar="",
    ):
        perfect_prec = False

        # 1st -> guess of irrigation
        if not assumption_sprinkler:
            if self.verbose:
                print("NO SPRINKLER!")

        if (self.res_sm >= pd.Timedelta("1D")) & (assumption_sprinkler):
            self.irr_from_inverse_api = self.calculate_inverse_api(
                sat_sm=self.sat_sm.resample("1H")
                .interpolate()
                .resample("1D")
                .interpolate(),
            )
        else:
            self.irr_from_inverse_api = self.calculate_inverse_api(
                sat_sm=self.sat_sm,
            )

        # 2nd -> merge irrigation with precipitation (at precipitation resolution - 2 scenarios)
        if (self.res_sm >= pd.Timedelta("1D")) & (assumption_sprinkler):
            (
                self.guess_min,
                self.guess_max,
                self.prec,
            ) = self.merge_initial_precipitation_with_irrigation(
                assumption_sprinkler=assumption_sprinkler,
                dt_irr=24,
            )
        else:
            (
                self.guess_min,
                self.guess_max,
                self.prec,
            ) = self.merge_initial_precipitation_with_irrigation(
                assumption_sprinkler=assumption_sprinkler
            )

        # we add precipitation to the irrigation
        self.guess_min = self.guess_min.add(self.prec, axis="index")
        self.guess_max = self.guess_max.add(self.prec, axis="index")

        # 3rd -> create particle filter
        self.pf_values = self.create_factors_particle_filters(
            n_perturbation=150, min_val=0.1, max_val=10, n_zeros=50
        )
        self.pf_factors_zeroirr = self.create_factors_particle_filters(
            n_perturbation=150, min_val=0.9, max_val=1.1, n_zeros=0
        )
        # 4th -> apply particle filter
        (
            self.guess_min_perturbed,
            self.guess_max_perturbed,
        ) = self.create_perturbed_dataset(add_norrigation_datasets=True)

        # 5th run prism
        (
            self.est_irr_min,
            self.est_sm_min,
            self.prism_sm_table_min,
        ) = self.prism_function_v2p1_2D(
            perturbed_irr_dataset=self.guess_min_perturbed,
            satellite_sm=self.sat_sm,
            n_best_values=n_best_values,
            text_bar=text_bar,
            window=self.window,
            filtering_1d_nans=filtering_1d_nans,
        )

        if (
            self.res_sm <= 2 * self.res_prec
        ):  # there is no difference between min and max
            self.est_irr_max = self.est_irr_min
            self.est_sm_max = self.est_sm_min
            self.prism_sm_table_max = self.prism_sm_table_min

        else:
            (
                self.est_irr_max,
                self.est_sm_max,
                self.prism_sm_table_max,
            ) = self.prism_function_v2p1_2D(
                perturbed_irr_dataset=self.guess_max_perturbed,
                satellite_sm=self.sat_sm,  # .resample("1D").mean().copy(),
                text_bar=text_bar,
                n_best_values=n_best_values,
                window=self.window,
                filtering_1d_nans=filtering_1d_nans,
            )

    def prism_function_v2p1_2D(
        self,
        perturbed_irr_dataset: pd.DataFrame = None,
        satellite_sm: pd.DataFrame = None,
        high_res: pd.Timedelta = None,
        steps_index: pd.DatetimeIndex = None,
        window: int = 5,
        filtering_1d_nans=False,
        n_best_values: int = 1,
        text_bar="",
    ):
        if self.verbose:
            print(f"\n\n\n\n WINDOW = {window}!")
        # check inputs
        if perturbed_irr_dataset is None:
            perturbed_irr_dataset = self.guess_min_perturbed
        if satellite_sm is None:
            satellite_sm = self.sat_sm
        if filtering_1d_nans:
            if self.verbose:
                print("\n\nFILTERING GAPS!\n\n")
            # check for too many consecutive NaNs (only works for 1D timeseries of daily Soil Moisture)
            # count the consecutive NaNs in the time series
            col1 = satellite_sm.columns[0]
            satellite_sm["consecutive_NaNs"] = (
                satellite_sm[col1]
                .isnull()
                .astype(int)
                .groupby(satellite_sm[col1].notnull().astype(int).cumsum())
                .cumsum()
            )
            #  remove row if there are more than 4 consecutive NaNs
            satellite_sm = satellite_sm[
                satellite_sm["consecutive_NaNs"] < window - 1
            ].loc[:, [col1]]
        if high_res is None:
            high_res = self.res_prec
        if steps_index is None:
            if self.res_prec < pd.Timedelta("1D"):
                #                 steps_index = satellite_sm.resample('1D').mean().dropna().index
                steps_index = (
                    satellite_sm.groupby(satellite_sm.index.date)
                    .apply(lambda x: x.index[0])
                    .values
                )
            else:
                steps_index = satellite_sm.dropna(how="all", axis=1).index
        self.steps_index = steps_index
        # parameters
        sort_columns_order = perturbed_irr_dataset.loc[
            :, perturbed_irr_dataset.columns.get_level_values(0)[0]
        ].columns
        n_perturbations = len(set(perturbed_irr_dataset.columns.get_level_values(0)))
        params_api = self.params_api.copy()
        params_api["tau"] = pd.concat(
            [params_api["tau"].loc[:, sort_columns_order]] * n_perturbations, axis=1
        )
        params_api["tau"].columns = perturbed_irr_dataset.columns
        params_api["sm_res"] = pd.concat(
            [params_api["sm_res"].loc[sort_columns_order]] * n_perturbations
        )
        params_api["sm_res"].index = perturbed_irr_dataset.columns
        params_api["sm_sat"] = pd.concat(
            [params_api["sm_sat"].loc[sort_columns_order]] * n_perturbations
        )
        params_api["sm_sat"].index = perturbed_irr_dataset.columns

        # outputs
        mc = list(
            product(
                set(perturbed_irr_dataset.columns.get_level_values(1)),
                [f"win_{ind}" for ind in range(len(steps_index[:-1]))],
            )
        )
        tot_table_sm = pd.DataFrame(
            index=perturbed_irr_dataset.index, columns=pd.MultiIndex.from_tuples(mc)
        )

        for it, center in ntqdm(
            enumerate(steps_index[:-1]),
            total=len(steps_index[:-1]),
            desc=f"PrISM 2.1 {text_bar}",
        ):
            index_start = it - window // 2
            if index_start <= 0:
                index_start = 0
            start = steps_index[index_start]

            index_stop = it + window // 2 + 1
            if index_stop >= len(steps_index):
                index_stop = -1
            stop = steps_index[index_stop]

            from_index = center
            until_index = steps_index[it + 1]  # - high_res

            n_elements = len(satellite_sm.loc[start:stop])
            # print(center, n_elements)
            if it == 0:
                initial_sm_value = satellite_sm.apply(select_initial_sm, axis=0)
            else:
                initial_sm_value = tot_table_sm.loc[start, :].groupby(level=0).mean()
            start_soil_moisture = self.create_noise_sm_at_t0(
                initial_sm_value[sort_columns_order].values
            )
            # 1. calculate perturbed soil moisture
            sim_sm = self.calculate_api(
                prec=perturbed_irr_dataset.loc[start:stop],
                initial_sm=start_soil_moisture,
                dt=high_res / pd.Timedelta("1H"),
                params_api=params_api,
                disable_bar=True,
            )

            # 2. calculate best rmse
            for coords, sim_group in sim_sm.groupby(level=1, axis=1):
                indd = satellite_sm.loc[start:stop, coords].index
                errors = (
                    sim_group.loc[indd, :].astype(float).T
                    - satellite_sm.loc[indd, coords].values
                ) ** 2
                rmse = errors.mean(axis=1) ** (0.5)
                # 3. find best irrigation and soil moisture and save them in the table
                sel = rmse.sort_values().iloc[0:n_best_values].index
                p1 = sim_sm.loc[start:stop, sel].mean(axis=1)
                tot_table_sm.loc[start:stop, (coords, f"win_{it}")] = p1
        if self.verbose:
            print(self.guess_min)
        ## irrigation from the retrieved soil moisture
        final_irr = self.calculate_inverse_api(
            sat_sm=tot_table_sm.mean(axis=1, level=0).dropna(axis=0),
            initial_prec=0,
            params_api=self.params_api,
        ).astype(float)
        final_irr = final_irr.where(final_irr > 0, 0)
        ## final soil moisture corrected
        final_sm = self.calculate_api(
            prec=final_irr,
            initial_sm=tot_table_sm.mean(axis=1, level=0).dropna(axis=0).iloc[0],
            params_api=self.params_api,
        )

        return final_irr, final_sm, tot_table_sm  # prism_irr_table, prism_sm_table

    def calculate_api(
        self,
        prec,
        params_api: dict = None,
        initial_sm: float = None,
        dt: int = None,
        disable_bar=None,
    ):
        """
        1st Wrapper: Function to performs API time-series computation for PrISM (Pellarin, 2020).
        INPUTS:
            prec: pandas.DataFrame
                Dataframe/Series that has as index the datetime and as values all the precipitation observation in mm between the dataframe included.
            params_api: dict {str:float}
                Additional params to be included for the calculation of the API [sm_res, dt, tau, p_t1, d_soil, sm_sat] otherwise default values are adopted (from Pellarin et al., 2020, Niger site, Figure 2).

        """
        # check inputs
        if params_api is None:
            params_api = self.params_api.copy()
        if initial_sm is None:
            initial_sm = self.sat_sm.iloc[0, :]
        if disable_bar is None:
            disable_bar = True
        # check parameters api
        if "tau" in params_api.keys():
            tau = params_api["tau"]
            if (isinstance(tau, float)) | (isinstance(tau, int)):
                tau = np.ones(prec.shape) * tau
                tau = pd.DataFrame(tau, index=prec.index, columns=prec.columns)
            tau = tau.reindex(prec.index).interpolate()
            params_api.pop("tau")
        else:
            raise ValueError(
                "TAU is not specified in params_api, specified as a dict items with key = 'tau'"
            )

        # calculate api time-series
        sm_pred = pd.DataFrame(index=prec.index, columns=prec.columns)
        if dt is None:
            dt = prec.apply(calculate_dt_series, axis=0)
        min_resolution = (sm_pred.index[1] - sm_pred.index[0]) / pd.Timedelta("1H")
        for it, inn in ntqdm(
            enumerate(prec.index),
            total=len(prec),
            desc="API",
            disable=disable_bar,
        ):
            if it == 0:
                sm_pred.loc[inn, :] = initial_sm
            else:
                innold = sm_pred.iloc[it - 1, :].name
                params_api_final = {
                    "sm_res": 5,
                    "sm_sat": 45,
                    "tau": tau.loc[innold:inn].mean(axis=0),
                    "d_soil": 40,
                }
                params_api_final.update(params_api)
                if isinstance(dt, pd.DataFrame):
                    params_api_final["dt"] = dt.loc[
                        inn
                    ]  # dt is directly extracted from time-series
                else:
                    params_api_final["dt"] = dt
                sm_pred.loc[inn, :] = antecedent_precipitation_index(
                    sm_t0=sm_pred.iloc[it - 1, :],
                    p_t1=prec.loc[inn, :],
                    **params_api_final,
                )
        return sm_pred

    def calculate_inverse_api(
        self,
        sat_sm=None,
        params_api=None,
        initial_prec=None,
    ):
        """
        1st Wrapper: Function to performs API time-series computation for PrISM (Pellarin, 2020).
        INPUTS:
            sm: pandas.DataFrame
                Dataframe/Series that has as index the datetime and as values Soil Moisture.
            params_api: dict {str:float}
                Additional params to be included for the calculation of the API [sm_res, dt, tau, p_t1, d_soil, sm_sat] otherwise default values are adopted (from Pellarin et al., 2020, Niger site, Figure 2).
        """
        # check inputs
        if params_api is None:
            params_api = self.params_api.copy()
        if initial_prec is None:
            initial_prec = self.prec.iloc[0, 0]
        if sat_sm is None:
            sat_sm = self.sat_sm.dropna(how="all", axis=0)
        # check parameters api
        if "tau" in params_api.keys():
            tau = params_api["tau"]
            if (isinstance(tau, float)) | (isinstance(tau, int)):
                tau = np.ones(sat_sm.shape) * tau
                tau = pd.DataFrame(tau, index=sat_sm.index, columns=sat_sm.columns)
            tau = tau.reindex(sat_sm.index).interpolate()
            params_api.pop("tau")
        else:
            raise ValueError(
                "TAU is not specified in params_api, specified as a dict items with key = 'tau'"
            )

        # calculate inverse api time-series
        prec_pred = pd.DataFrame(index=sat_sm.index, columns=sat_sm.columns)
        all_dts = sat_sm.apply(calculate_dt_series, axis=0)
        prec_sat_sm = sat_sm.apply(calcualte_precedent_smvalues, axis=0)

        for it, inn in ntqdm(
            enumerate(sat_sm.index),
            total=len(sat_sm),
            desc="inverse API",
            disable=(not self.verbose),
        ):
            if it == 0:
                prec_pred.loc[inn, :] = initial_prec
                innold = inn
            else:
                params_api_final = {
                    "sm_res": 5,
                    "sm_sat": 45,
                    "tau": tau.loc[innold:inn].mean(axis=0),
                    "d_soil": 40,
                }
                params_api_final.update(params_api)

                params_api_final["dt"] = all_dts.loc[
                    inn, :
                ]  # dt is directly extracted from time-series
                prec_pred.loc[inn, :] = inverse_antecedent_precipitation_index(
                    sm_t0=prec_sat_sm.loc[inn, :],
                    sm_t1=sat_sm.loc[inn, :],
                    **params_api_final,
                )
                innold = inn

        return prec_pred

    ### 2. FUNCTIONS
    def downsample_first_guess_irr(
        self,
        dt_irr: pd.Timedelta = None,
        assumption_sprinkler=False,
    ):
        """Function to create a first irrigation guess with teh same temporal resolution of precipitation."""
        # 0. check inputs
        precipitation = self.prec
        irr = self.irr_from_inverse_api
        dt_prec = self.res_prec
        dt_prec = pd.Timedelta(f"{dt_prec}H")
        if dt_irr is None:
            dt_irr = self.res_sm
        if isinstance(dt_irr, int):
            dt_irr = pd.Timedelta(f"{dt_irr}H")

        ## make sure precipitation and irrigation have values for each timestep (if not fill with 0)
        new_index_irr = pd.date_range(irr.index[0], irr.index[-1], freq=dt_irr)
        irr = irr.reindex(new_index_irr)  # .fillna(0) ###-### commented on 12062023 GP

        # 1. downsample precipitation and remove precipitation from first guess
        precipitation_as_irr = (
            precipitation.resample(dt_irr, origin=irr.index[0]).sum().shift(1)
        )
        common_ind = pd.DatetimeIndex(
            [f for f in precipitation_as_irr.index if f in irr.index]
        )
        if len(common_ind) == 0:
            raise ValueError(
                f"ERROR in downsampling precipitation into a datetimeindex that is not the same as the low resolution sm. CHECK it."
            )
        precipitation_as_irr = precipitation_as_irr.loc[common_ind]

        # 2. move irrigation events right before (after) the current (previous) observation,
        #    to get the minimum (maximum) possible irrigation event.
        irr = irr.where(
            (irr > 0) & (~irr.isna()), 0
        )  # second, we clean it from observations that gives negative values.

        if dt_irr < pd.Timedelta(f"1D"):
            # Set only 1 irrigation event per day
            # (if the initial guess has a 'higher' temporal resolution than 1 day,
            #  there will be too many irrigations in one day)
            irr2 = irr.copy()
            #  if irr has a higher resoluton than 24H, only 1 event will be taken per day, with value = sum(irr[24H]).
            test_distribution_irr = (
                lambda gg: gg.where(gg == gg.max(), 0)
                if (gg.values > 0).sum() > 1
                else gg
            )
            for cc in irr2.columns:
                irr2[cc] = irr[cc].groupby(irr.index.date).apply(test_distribution_irr)
            rescale_factor = 1  # 20/irr2.replace({0:np.nan}).quantile(0.99)
            irr = pd.DataFrame(rescale_factor * irr2)
        elif dt_irr >= pd.Timedelta(f"1D"):
            if assumption_sprinkler:
                # in case SM temporal resolution lower than 1 day distribute at least 1 irrigation event per day
                # (assumption for sprinkler irrigation)
                if self.verbose:
                    print("assuming irrigation every day!")
                # rescale to 24H and assume irrigation to be every day when present
                ndays = dt_irr / pd.Timedelta("1D")
                irr = irr.resample("1D").bfill() / ndays
                dt_irr = pd.Timedelta("1D")
            irr = irr.where((precipitation_as_irr.loc[irr.index] == 0).values, 0)
        self.precasirr = precipitation_as_irr.copy()
        irr = irr.where(np.invert(self.irr_from_inverse_api.isna()), np.nan)
        return irr

    def upsample_scenarios(self, dt_irr, dt_prec):
        sim_min = self.intermediate_irr.iloc[1::, :].copy()
        sim_max = self.intermediate_irr.iloc[1::, :].copy()

        if dt_irr > 2 * dt_prec:
            dtas = self.irr_from_inverse_api.apply(calculate_dt_series).astype(float)
            if self.verbose:
                print("s", len(dtas.index), len(sim_max.index), sim_max.index)
            sim_max = sim_max.apply(move_values_deltas, args=(dtas,))
            sim_max.index = sim_max.index + dt_prec

        new_index = pd.date_range(
            self.intermediate_irr.index[0],
            self.intermediate_irr.index[-1],
            freq=dt_prec,
        )
        sim_min = sim_min.reindex(new_index).fillna(0)
        sim_max = sim_max.reindex(new_index).fillna(0)
        all_ind = list(
            (
                sim_max.index.append(sim_min.index).append(self.prec.index)
            ).drop_duplicates()
        )
        all_ind.sort()
        all_ind = pd.DatetimeIndex(all_ind)
        sim_min = sim_min.reindex(all_ind).fillna(0).loc[self.prec.index].round(6)
        sim_max = sim_max.reindex(all_ind).fillna(0).loc[self.prec.index].round(6)

        return sim_min, sim_max

    def merge_initial_precipitation_with_irrigation(
        self,
        dt_irr: pd.Timedelta = None,
        assumption_sprinkler=False,
    ):
        """
        Function to merge the guessed irrigation, computed from Soil moisture at low temporal resolution,
        with the initial precipitation available at high temporal resolution.

        TWO scenarios are created to resolve the temporal mismatch between initial precipitation and guessed irrigation:
        1st scenario: min_irrigation -> irrigation events happens right before the sm observation
        2nd scenario: max_irrigation -> irrigation happens right after the previous sm observation

        INPUTS:
            - assumption_sprinkler: bool
                                    If to distribute first irrigation guess to daily frequency when irrigation is present.
            - dt_irr: int or pd.Timedelta
                     temporal frequency of irrigation guess. if 'int' it is expressed in hours.
        """
        # 0. check inputs
        precipitation = self.prec
        irr = self.irr_from_inverse_api
        dt_prec = self.res_prec
        dt_prec = pd.Timedelta(f"{dt_prec}H")
        if dt_irr is None:
            dt_irr = self.res_sm
        if isinstance(dt_irr, int):
            dt_irr = pd.Timedelta(f"{dt_irr}H")
        # 1. upsample first guess irrigation
        self.intermediate_irr = self.downsample_first_guess_irr(
            dt_irr=dt_irr, assumption_sprinkler=assumption_sprinkler
        )

        # 2. upsample to the 2 scenarios
        sim_min, sim_max = self.upsample_scenarios(dt_irr, dt_prec)

        return sim_min, sim_max, precipitation

    def create_factors_particle_filters(
        self,
        min_val: int = 0,
        max_val: int = 2,
        n_perturbation: int = 100,
        n_zeros: int = 10,
        random_state: int = 1234,
    ):
        """
        Creates a random uniform distribution of 'n_perturbation' values that goes from 'min_val' to 'max_val'.
        Optionally it subsitute a portion of these factors with an arbitrary number 'n_zeros' of zeros.
        'random_state' indicates the state chosen for numpy. use always 1234 to get reproductible values.
        """
        if self.verbose:
            print(
                f"{n_perturbation} n_factors between {min_val} and {max_val} ({n_zeros} are 0)."
            )
        r = np.random.RandomState(random_state)
        particle_filter_values = np.append(
            r.uniform(min_val, max_val, n_perturbation - n_zeros), [0] * n_zeros
        )
        particle_filter_values.sort()
        particle_filter_values[len(particle_filter_values) // 2] = 1
        return particle_filter_values

    def perturbed_dataframe_multiplication(
        self, dataset: pd.DataFrame = None, parameters: np.array = None
    ):
        """
        Function to sort out the dimension of the perturbed dataset and build it.
        The dataset will have index = pd.DatetimeIndex and columns as MultiIndex,
        where column_level=0 is the parameters values, column_level=1 are the original columns'names.
        Each dataset in column_level=1 is the original dataframe multiplied by the perturbed value in column_level=0.
        """
        if dataset is None:
            dataset = self.guess_min.copy()
        if parameters is None:
            parameters = self.pf_values
        parameters_names = [f"{it:03d}_{rr}" for it, rr in enumerate(parameters)]
        columns = pd.MultiIndex.from_tuples(
            [f for f in itertools.product(parameters_names, dataset.columns)]
        )
        dataset_new = pd.concat([dataset] * len(parameters), axis=1) * np.repeat(
            parameters, dataset.shape[1]
        )
        dataset_new.columns = pd.MultiIndex.from_tuples(columns)
        return dataset_new

    def create_perturbed_dataset(
        self,
        dataframe: pd.DataFrame = None,
        pf_factors: np.array = None,
        add_norrigation_datasets=True,
    ):
        """
        From a time-series stored in a pd.Series or pd.DataFrame with one column (and pd.Datetimeindex)
        creates a pd.Dataframe with as many columns as the values in 'pf_factors' (which values will be the new column's names).
        Each time-series corresponds to the initial time-series multiplied by 'pf_factors'.
        """
        # check inputs
        if pf_factors is None:
            pf_factors = self.pf_values
        if dataframe is None:
            res = []
            for df in [self.guess_min, self.guess_max]:
                res.append(self.create_perturbed_dataset(dataframe=df))
            return res

        dataframe_perturbed = self.perturbed_dataframe_multiplication(
            dataset=dataframe, parameters=pf_factors
        )

        dataframe_perturbed = dataframe_perturbed.dropna().astype(float)

        ########## ADDITION to support simulations where there is no irrigation
        ########## (allows to delete irrigation guesses if not real).
        if add_norrigation_datasets:
            zeroirr_perturbed = self.create_perturbed_dataset(
                dataframe=self.prec,
                add_norrigation_datasets=False,
                pf_factors=self.pf_factors_zeroirr,
            )
            #             zeroirr_perturbed.columns= [999000+cc for cc in zeroirr_perturbed.columns]
            zeroirr_perturbed.rename(
                columns={
                    cc: f"noirr_{cc}"
                    for cc in zeroirr_perturbed.columns.get_level_values(0)
                },
                level=0,
                inplace=True,
            )
            dataframe_perturbed = pd.merge(
                dataframe_perturbed,
                zeroirr_perturbed,
                right_index=True,
                left_index=True,
            )

        return dataframe_perturbed

    ### 2. FUNCTIONS PRISM
    def find_fitting_irrigation(
        self,
        perturbed_irr=None,
        results=None,
        win: int = 5,
        show_progress_bar: bool = None,
    ):
        # check inputs
        if perturbed_irr is None:
            perturbed_irr = self.guess_min_perturbed
        if results is None:
            results = self.results_min
        if show_progress_bar is None:
            show_progress_bar = self.show_progress_bar
        assert isinstance(win, int)
        # like a rolling window calculating average along the coumns and taking the diagonal as final result
        allres = []
        for nn, ind in ntqdm(
            enumerate(results.index),
            total=len(results),
            desc="particle filter",
            disable=(not show_progress_bar),
        ):
            sti = nn - win // 2
            eni = nn + win // 2 + 1
            if sti < 0:
                sti = 0
            if eni > results.shape[0] - 1:
                eni = results.shape[0] - 1
            stidate, enidate = results.index[sti], results.index[eni]

            allres.append(
                perturbed_irr[results.loc[ind, :]].mean(axis=1).loc[stidate:enidate]
            )

        all_irrigation_averaged2 = pd.DataFrame(allres)
        final_irrigation = all_irrigation_averaged2.mean(axis=0)
        return pd.DataFrame(final_irrigation, columns=["IRR+prec"])

    def create_noise_sm_at_t0(self, start_value):
        r = np.random.RandomState(1234)
        size_mult = self.guess_max_perturbed.loc[
            :, self.guess_max_perturbed.columns[0][0]
        ].shape[1]
        size_factors = len(set(self.guess_max_perturbed.columns.get_level_values(0)))
        noise_sm = 1 - 0.1 * np.abs(r.normal(scale=1, size=size_mult))
        noise_sm[0 : int(0.15 * len(noise_sm))] = 1
        noise_sm[-int(0.15 * len(noise_sm)) : :] = 1

        start_sm = np.tile(start_value, size_factors) * np.repeat(
            noise_sm, size_factors
        )
        res_sm = np.tile(
            self.params_api["sm_res"][self.prec.columns].values, size_factors
        )

        # check - it cannot be lower than residual soil moisture
        start_sm = np.where(start_sm < res_sm, res_sm, start_sm)
        return start_sm

    def prism_function_v2p1_2D(
        self,
        perturbed_irr_dataset: pd.DataFrame = None,
        satellite_sm: pd.DataFrame = None,
        high_res: pd.Timedelta = None,
        steps_index: pd.DatetimeIndex = None,
        window: int = 5,
        filtering_1d_nans=False,
        n_best_values: int = 1,
        text_bar="",
    ):
        if self.verbose:
            print(f"\n\n\n\n WINDOW = {window}!")
        # check inputs
        if perturbed_irr_dataset is None:
            perturbed_irr_dataset = self.guess_min_perturbed
        if satellite_sm is None:
            satellite_sm = self.sat_sm
        if filtering_1d_nans:
            if self.verbose:
                print("\n\nFILTERING GAPS!\n\n")
            # check for too many consecutive NaNs (only works for 1D timeseries of daily Soil Moisture)
            # count the consecutive NaNs in the time series
            col1 = satellite_sm.columns[0]
            satellite_sm["consecutive_NaNs"] = (
                satellite_sm[col1]
                .isnull()
                .astype(int)
                .groupby(satellite_sm[col1].notnull().astype(int).cumsum())
                .cumsum()
            )
            #  remove row if there are more than 4 consecutive NaNs
            satellite_sm = satellite_sm[
                satellite_sm["consecutive_NaNs"] < window - 1
            ].loc[:, [col1]]
        if high_res is None:
            high_res = self.res_prec
        if steps_index is None:
            if self.res_prec < pd.Timedelta("1D"):
                #                 steps_index = satellite_sm.resample('1D').mean().dropna().index
                steps_index = (
                    satellite_sm.groupby(satellite_sm.index.date)
                    .apply(lambda x: x.index[0])
                    .values
                )
            else:
                steps_index = satellite_sm.dropna(how="all", axis=1).index
        self.steps_index = steps_index
        # parameters
        sort_columns_order = perturbed_irr_dataset.loc[
            :, perturbed_irr_dataset.columns.get_level_values(0)[0]
        ].columns
        n_perturbations = len(set(perturbed_irr_dataset.columns.get_level_values(0)))
        params_api = self.params_api.copy()
        params_api["tau"] = pd.concat(
            [params_api["tau"].loc[:, sort_columns_order]] * n_perturbations, axis=1
        )
        params_api["tau"].columns = perturbed_irr_dataset.columns
        params_api["sm_res"] = pd.concat(
            [params_api["sm_res"].loc[sort_columns_order]] * n_perturbations
        )
        params_api["sm_res"].index = perturbed_irr_dataset.columns
        params_api["sm_sat"] = pd.concat(
            [params_api["sm_sat"].loc[sort_columns_order]] * n_perturbations
        )
        params_api["sm_sat"].index = perturbed_irr_dataset.columns

        # outputs
        mc = list(
            product(
                set(perturbed_irr_dataset.columns.get_level_values(1)),
                [f"win_{ind}" for ind in range(len(steps_index[:-1]))],
            )
        )
        tot_table_sm = pd.DataFrame(
            index=perturbed_irr_dataset.index, columns=pd.MultiIndex.from_tuples(mc)
        )

        for it, center in ntqdm(
            enumerate(steps_index[:-1]),
            total=len(steps_index[:-1]),
            desc=f"PrISM 2.1 {text_bar}",
            disable=(not self.verbose),
        ):
            index_start = it - window // 2
            if index_start <= 0:
                index_start = 0
            start = steps_index[index_start]

            index_stop = it + window // 2 + 1
            if index_stop >= len(steps_index):
                index_stop = -1
            stop = steps_index[index_stop]

            from_index = center
            until_index = steps_index[it + 1]  # - high_res

            n_elements = len(satellite_sm.loc[start:stop])
            # print(center, n_elements)
            if it == 0:
                initial_sm_value = satellite_sm.apply(select_initial_sm, axis=0)
            else:
                initial_sm_value = tot_table_sm.loc[start, :].groupby(level=0).mean()

            self.in_val = initial_sm_value[sort_columns_order].values
            start_soil_moisture = self.create_noise_sm_at_t0(
                initial_sm_value[sort_columns_order].values
            )
            #             start_soil_moisture = pd.Series(np.tile(initial_sm_value[sort_columns_order], n_perturbations),
            #                                             index = perturbed_irr_dataset.columns)
            # 1. calculate perturbed soil moisture
            sim_sm = self.calculate_api(
                prec=perturbed_irr_dataset.loc[start:stop],
                initial_sm=start_soil_moisture,
                dt=high_res / pd.Timedelta("1H"),
                params_api=params_api.copy(),
            )

            # 2. calculate best rmse
            for coords, sim_group in sim_sm.groupby(level=1, axis=1):
                indd = satellite_sm.loc[start:stop, coords].index
                errors = (
                    sim_group.loc[indd, :].astype(float).T
                    - satellite_sm.loc[indd, coords].values
                ) ** 2
                rmse = errors.mean(axis=1) ** (0.5)
                # 3. find best irrigation and soil moisture and save them in the table
                sel = rmse.sort_values().iloc[0:n_best_values].index
                p1 = sim_sm.loc[start:stop, sel].mean(axis=1)
                tot_table_sm.loc[start:stop, (coords, f"win_{it}")] = p1
        if self.verbose:
            print("GUESS MIN\n", self.guess_min)
        ## irrigation from the retrieved soil moisture
        final_irr = self.calculate_inverse_api(
            sat_sm=tot_table_sm.mean(axis=1, level=0).dropna(axis=0),
            initial_prec=0,
            params_api=self.params_api.copy(),
        ).astype(float)
        final_irr = final_irr.where(final_irr > 0, 0)
        ## final soil moisture corrected
        final_sm = self.calculate_api(
            prec=final_irr,
            initial_sm=tot_table_sm.mean(axis=1, level=0).dropna(axis=0).iloc[0],
            params_api=self.params_api.copy(),
        )

        return final_irr, final_sm, tot_table_sm  # prism_irr_table, prism_sm_table
