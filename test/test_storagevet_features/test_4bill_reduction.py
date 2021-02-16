"""
This file tests bill reduction analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
from test_storagevet_features.TestingLib import assert_ran, run_case
from pathlib import Path


DIR = Path('./test/model_params/')


def test_ice():
    assert_ran(DIR / "100-billreduction_ice_month.json")


def test_controllable_load_month():
    assert_ran(DIR / "152-billreduction_battery_controllableload_month.json")


def test_ice_pv():
    assert_ran(DIR / "101-pv_ice_bill_reduction.json")


def test_ice_ice():
    assert_ran(DIR / "105-ice_ice_bill_reduction.json")


def test_pv_ice_ice():
    assert_ran(DIR / "106-pv_ice_ice_bill_reduction.json")


def test_pv_pv_ice():
    assert_ran(DIR / "104-pv_pv_ice_bill_reduction.json")


def test_battery():
    """tests fixed size with retail ETS and DCM services through 1 battery"""
    assert_ran(DIR / "004-fixed_size_battery_retailets_dcm.json")


def test_pv_curtail():
    assert_ran(DIR / "103-pv_cutail_bill_reduction.json")


def test_pv_no_curtail():
    assert_ran(DIR / "102-pv_bill_reduction.json")
