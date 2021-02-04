"""
This file tests market participation analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import pytest
from Testing.TestingLib import assert_ran


PROGRAMS = ['storagevet', 'dervet']
TEST_PROGRAM = PROGRAMS[1]  # change this str to switch between storagevet
# and dervet


@pytest.mark.slow
def test_batt():
    assert_ran(r".\Testing\Model_params\battery\069-DA_FR_SR_NSR_battery_month.csv", TEST_PROGRAM)


def test_pv_ice():
    assert_ran(r".\Testing\Model_params\002-DA_FR_SR_NSR_pv_ice_month.csv", TEST_PROGRAM)


def test_batt_pv_ice():
    assert_ran(r".\Testing\Model_params\battery_pv_genset\069-DA_FR_SR_NSR_battery_pv_ice_month.csv", TEST_PROGRAM)


@pytest.mark.slow
def test_batt_ts_constraints():
    assert_ran(r".\Testing\Model_params\battery\001-DA_FR_SR_NSR_battery_month_ts_constraints.csv", TEST_PROGRAM)


