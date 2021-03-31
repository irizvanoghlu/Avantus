"""
This file tests analysis cases that ONLY contain a SINGLE BATTERY. It is
organized by value stream combination and tests a bariety of optimization
horizons, time scale sizes, and other scenario options. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import pytest
from pathlib import Path
import numpy as np
from test.TestingLib import *


DIR = Path("./test/model_params")
JSON = '.json'
CSV = '.csv'


def test_battery_timeseries_constraints():
    test_file = DIR / f'001-DA_FR_SR_NSR_battery_month_ts_constraints{CSV}'
    results = assert_ran(test_file)
    case_results = results.instances[0]
    timeseries = case_results.time_series_data
    discharge_constraint = timeseries['BATTERY: battery User Discharge Max (kW)']
    charge_constraint = timeseries['BATTERY: battery User Charge Max (kW)']
    assert np.all(timeseries['BATTERY: battery Discharge (kW)'] <= discharge_constraint)
    assert np.all(timeseries['BATTERY: battery Charge (kW)'] <= charge_constraint)

