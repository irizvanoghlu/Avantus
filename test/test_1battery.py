"""
This file tests analysis cases that ONLY contain a SINGLE BATTERY. It is
organized by value stream combination and tests a bariety of optimization
horizons, time scale sizes, and other scenario options. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import sys
import os
import pytest
from pathlib import Path
from test.TestingLib import assert_ran, compare_proforma_results, run_case


DIR = Path('./test/test_random_cases/model_params/')
JSON = '.json'


def test_da_month():
    assert_ran(DIR / f'000-DA_battery_month{JSON}')


# @pytest.mark.slow
# def test_da_month3():
#     assert_ran(DIR / f'018-DA_battery_month_5min{JSON}', TEST_PROGRAM)
#
#
# def test_da_12hr():
#     assert_ran(DIR / f'019a-DA_battery_month_12hropt{JSON}', TEST_PROGRAM)
#
#
# def test_da_month5():
#     assert_ran(DIR / f'023-DA_month_results_dir_label{JSON}', TEST_PROGRAM)
#
#
# def test_da_month_degradation_multi_yr_battery_replaced_during_optimization():
#     assert_ran(r".\Testing\cba_validation\Model_params" +
#                r"\Model_Parameters_Template_ENEA_S1_8_12_UC1_DAETS.csv",
#                TEST_PROGRAM)
#
#
# def test_da_month_degradation_battery_replaced_during_optimization():
#     assert_ran(DIR / f"degradation_test{JSON}", TEST_PROGRAM)
#
#
# @pytest.mark.slow
# def test_da_fr_month():
#     assert_ran(DIR / f'001-DA_FR_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_deferral_month():  # should run for years: 2017 2023 2024
#     assert_ran(DIR / f'003-DA_Deferral_battery_month{JSON}', TEST_PROGRAM)
#
#
# @pytest.mark.slow
# def test_da_nsr_month():
#     assert_ran(DIR / f'005-DA_NSR_battery_month{JSON}', TEST_PROGRAM)
#
#
# @pytest.mark.slow
# def test_da_nsr_month1():
#     assert_ran(DIR / f'007-nsr_battery_multiyr{JSON}', TEST_PROGRAM)
#
#
# @pytest.mark.slow
# def test_da_sr_month():
#     assert_ran(DIR / f'006-DA_SR_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_user_month():
#     assert_ran(DIR / f'011-DA_User_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_ra_month():
#     assert_ran(DIR / f'012-DA_RApeakmonth_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_ra_month1():
#     assert_ran(DIR / f'013-DA_RApeakmonthActive_battery_month{JSON}',
#                TEST_PROGRAM)
#
#
# def test_da_ra_month2():
#     assert_ran(DIR / f'014-DA_RApeakyear_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_dr_month():
#     assert_ran(DIR / f'015-DA_DRdayahead_battery_month{JSON}', TEST_PROGRAM)
#
#
# def test_da_dr_month1():
#     assert_ran(DIR / f'016-DA_DRdayof_battery_month{JSON}', TEST_PROGRAM)
