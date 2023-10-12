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

from PrISM_light import IRRI_PRISM_light


def IRRIPRISM_unit_tests(assumption_sprinkler=False):
    """test all the variables of IRRIPRISM
    to see if they are identical whan using
    pandas series and pandas dataframes."""

    print(f"\nstart testing, assumption_sprinkler = {assumption_sprinkler}")
    # inputs
    testSM = pd.Series(
        [20, 16, 26, 25, 30, np.nan],
        index=pd.date_range("20100101", "20100106", freq="1D"),
    )
    testSM2 = pd.Series(
        [20, 16, 26, np.nan, 30, np.nan],
        index=pd.date_range("20100101", "20100106", freq="1D"),
    )
    testprec = pd.Series(
        [0, 0, 0, np.nan, 0, np.nan] * 6,
        index=pd.date_range("20100101T00", "20100105T11", freq="3H"),
    )
    testprec2 = (
        testprec.copy()
    )  # pd.Series([0, 0, 0, np.nan, 0, np.nan]*6, index = pd.date_range("20100101T00","20100105T11", freq="3H"))
    testprec2.iloc[10] = 10

    tot_testSM = pd.concat([testSM2, testSM], axis=1)
    tot_testpp = pd.concat([testprec, testprec2], axis=1)

    test_params_api = {
        "sm_res": pd.Series([5, 5], index=tot_testSM.columns),
        "sm_sat": pd.Series([45, 45], index=tot_testSM.columns),
        "tau": pd.DataFrame(
            [[150] * len(testprec)] * tot_testSM.shape[1],
            index=tot_testSM.columns,
            columns=testprec.index,
        ).T,
        "d_soil": 40,
    }
    # run models
    start1 = time.time()
    IPl0 = IRRI_PRISM_light(
        precipitation=tot_testpp.iloc[:, 0].dropna(),
        satellite_sm=tot_testSM.iloc[:, 0],
        params_api=test_params_api,
        verbose=False,
    )
    IPl0.run(n_best_values=1, assumption_sprinkler=assumption_sprinkler)
    delta1 = time.time() - start1
    start2 = time.time()
    IPl1 = IRRI_PRISM_light(
        precipitation=tot_testpp.iloc[:, 1].dropna(),
        satellite_sm=tot_testSM.iloc[:, 1],
        params_api=test_params_api,
        verbose=False,
    )
    IPl1.run(n_best_values=1, assumption_sprinkler=assumption_sprinkler)
    delta2 = time.time() - start2

    start3 = time.time()
    IPl = IRRI_PRISM_light(
        precipitation=tot_testpp.dropna(),
        satellite_sm=tot_testSM,
        params_api=test_params_api,
        verbose=False,
    )
    IPl.run(n_best_values=1, assumption_sprinkler=assumption_sprinkler)
    delta3 = time.time() - start3

    # TESTS
    pd.testing.assert_series_equal(
        IPl.irr_from_inverse_api.iloc[:, 0], IPl0.irr_from_inverse_api.iloc[:, 0]
    )
    pd.testing.assert_series_equal(
        IPl.irr_from_inverse_api.iloc[:, 1], IPl1.irr_from_inverse_api.iloc[:, 0]
    )
    print("1/4. initial inversions are equal")
    pd.testing.assert_series_equal(
        IPl.intermediate_irr.iloc[:, 0], IPl0.intermediate_irr.iloc[:, 0]
    )
    pd.testing.assert_series_equal(
        IPl.intermediate_irr.iloc[:, 1], IPl1.intermediate_irr.iloc[:, 0]
    )
    pd.testing.assert_series_equal(IPl.guess_max.iloc[:, 0], IPl0.guess_max.iloc[:, 0])
    pd.testing.assert_series_equal(IPl.guess_max.iloc[:, 1], IPl1.guess_max.iloc[:, 0])
    pd.testing.assert_series_equal(IPl.guess_min.iloc[:, 0], IPl0.guess_min.iloc[:, 0])
    pd.testing.assert_series_equal(IPl.guess_min.iloc[:, 1], IPl1.guess_min.iloc[:, 0])
    print("2/4. initial guesses are equal")
    cols_0 = [f for f in IPl.guess_max_perturbed.columns if f[1] == 0]
    cols_1 = [f for f in IPl.guess_max_perturbed.columns if f[1] == 1]
    pd.testing.assert_frame_equal(
        IPl.guess_max_perturbed[cols_0], IPl0.guess_max_perturbed
    )
    pd.testing.assert_frame_equal(
        IPl.guess_max_perturbed[cols_1], IPl1.guess_max_perturbed
    )
    pd.testing.assert_frame_equal(
        IPl.guess_min_perturbed[cols_0], IPl0.guess_min_perturbed
    )
    pd.testing.assert_frame_equal(
        IPl.guess_min_perturbed[cols_1], IPl1.guess_min_perturbed
    )
    print("3/4. perturbed irrigation guesses are equal")
    pd.testing.assert_frame_equal(IPl.est_irr_min.iloc[:, [0]], IPl0.est_irr_min)
    pd.testing.assert_frame_equal(IPl.est_irr_min.iloc[:, [1]], IPl1.est_irr_min)
    pd.testing.assert_frame_equal(IPl.est_irr_max.iloc[:, [0]], IPl0.est_irr_max)
    pd.testing.assert_frame_equal(IPl.est_irr_max.iloc[:, [1]], IPl1.est_irr_max)

    pd.testing.assert_frame_equal(IPl.est_sm_min.iloc[:, [0]], IPl0.est_sm_min)
    pd.testing.assert_frame_equal(IPl.est_sm_min.iloc[:, [1]], IPl1.est_sm_min)
    pd.testing.assert_frame_equal(IPl.est_sm_max.iloc[:, [0]], IPl0.est_sm_max)
    pd.testing.assert_frame_equal(IPl.est_sm_max.iloc[:, [1]], IPl1.est_sm_max)

    cols_0 = [f for f in IPl.prism_sm_table_max.columns if f[0] == 0]
    cols_1 = [f for f in IPl.prism_sm_table_max.columns if f[0] == 1]
    pd.testing.assert_frame_equal(
        IPl.prism_sm_table_min[cols_0], IPl0.prism_sm_table_min
    )
    pd.testing.assert_frame_equal(
        IPl.prism_sm_table_min[cols_1], IPl1.prism_sm_table_min
    )
    pd.testing.assert_frame_equal(
        IPl.prism_sm_table_max[cols_0], IPl0.prism_sm_table_max
    )
    pd.testing.assert_frame_equal(
        IPl.prism_sm_table_max[cols_1], IPl1.prism_sm_table_max
    )
    print("4/4. SOLUTIONS irrigation estimations are equal")
    print(f"it took {delta1:.04f} sec to compute the fist column of the toy dataset")
    print(f"it took {delta2:.04f} sec to compute the fist column of the toy dataset")
    print(
        f"it took {delta3:.04f} sec to compute both columns of the toy dataset together"
    )
    return IPl


if __name__ == "__main__":
    IPl = IRRIPRISM_unit_tests(assumption_sprinkler=True)
    IPlnosp = IRRIPRISM_unit_tests(assumption_sprinkler=False)
