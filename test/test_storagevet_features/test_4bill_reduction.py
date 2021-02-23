"""
This file tests bill reduction analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
from test.TestingLib import assert_ran, run_case
from pathlib import Path


DIR = Path("./test/test_storagevet_features/model_params")


def test_ice():
    assert_ran(DIR / "030-billreduction_ice_month.csv")


def test_controllable_load_month():
    assert_ran(DIR / "031-billreduction_battery_controllableload_month.csv")


def test_ice_pv():
    assert_ran(DIR / "032-pv_ice_bill_reduction.csv")


def test_ice_ice():
    assert_ran(DIR / "033-ice_ice_bill_reduction.csv")


def test_pv_ice_ice():
    assert_ran(DIR / "034-pv_ice_ice_bill_reduction.csv")


def test_pv_pv_ice():
    assert_ran(DIR / "035-pv_pv_ice_bill_reduction.csv")


def test_battery():
    """tests fixed size with retail ETS and DCM services through 1 battery"""
    assert_ran(DIR / "004-fixed_size_battery_retailets_dcm.csv")


# def test_pv():
#     assert_ran(DIR / "036-pv_bill_reduction.csv")
