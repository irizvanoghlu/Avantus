"""
This file tests bill reduction analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
from Testing.TestingLib import assert_ran, run_case

PROGRAMS = ['storagevet', 'dervet']
TEST_PROGRAM = PROGRAMS[1]  # change this str to switch between storagevet
# and dervet


MP_DIR = r".\Testing\Model_params"
VENTURA_DIR = r"./Testing/Ventura/Model_Parameters"


def test_ice():
    assert_ran(MP_DIR + r"\000-billreduction_ice_month.csv", TEST_PROGRAM)


def test_controllable_load_month():
    assert_ran(MP_DIR +
               r"\152-billreduction_battery_controllableload_month.csv",
               TEST_PROGRAM)


def test_ice_pv():
    assert_ran(MP_DIR + r"\PV_ice_bill_reduction.csv", TEST_PROGRAM)


def test_ice_ice():
    assert_ran(MP_DIR + r"\ice_ice_bill_reduction.csv", TEST_PROGRAM)


def test_pv_ice_ice():
    assert_ran(MP_DIR + r"\PV_ice_ice_bill_reduction.csv", TEST_PROGRAM)


def test_battery():
    """tests fixed size with retail ETS and DCM services through 1 battery"""
    assert_ran(VENTURA_DIR + r"/Model_Parameters_Template_DER_Ventura_04.csv",
               TEST_PROGRAM)


def test_pv_curtail():
    if TEST_PROGRAM == 'dervet':
        assert_ran(MP_DIR + r"\pv_cutail_bill_reduction.csv", TEST_PROGRAM)


def test_pv_no_curtail():
    assert_ran(MP_DIR + r"\pv_bill_reduction.csv", TEST_PROGRAM)
