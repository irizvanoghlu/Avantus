"""
This file tests market participation analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import pytest
from test.TestingLib import assert_ran
from pathlib import Path

DIR = Path("./test/test_storagevet_features/model_params")


@pytest.mark.slow
def test_batt():
    assert_ran(DIR / "026-DA_FR_SR_NSR_battery_month.csv")


def test_pv_ice():
    assert_ran(DIR / "027-DA_FR_SR_NSR_pv_ice_month.csv")


def test_batt_pv_ice():
    assert_ran(DIR / "028-DA_FR_SR_NSR_battery_pv_ice_month.csv")


@pytest.mark.slow
def test_batt_ts_constraints():
    assert_ran(DIR / "029-DA_FR_SR_NSR_battery_month_ts_constraints.csv")


