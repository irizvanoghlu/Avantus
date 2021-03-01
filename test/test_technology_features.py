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
from test.TestingLib import *


DIR = Path("./test/model_params")
JSON = '.json'
CSV = '.csv'


def test_battery_timeseries_constraints():
    test_file = DIR / f'001-DA_FR_SR_NSR_battery_month_ts_constraints{CSV}'
    assert_ran_with_services(test_file, ['DA'])
