"""
This file tests the optimization sizing module. All tests should pass.

The tests in this file can be run with DERVET.

"""
from Testing.TestingLib import run_case, check_lcpc, assert_file_exists, compare_proforma_results, compare_size_results
import pytest

"""
ONLY POST FACTO TESTS
"""


def test_da_pf():
    results = run_case('./Testing/Model_params/battery/053-da_Reliability_post_facto.csv', 'dervet')
    assert_file_exists(results, 'load_coverage_prob')


def test_power_sizing_pf():
    """tests power sizing with retail ETS and DCM services through 1 battery  SIZE: 17500 kWh 4375 kW"""
    results = run_case("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_01.csv", 'dervet')
    assert_file_exists(results, 'load_coverage_prob')


def test_energy_sizing_pf():
    """tests energy sizing with retail ETS and DCM services through 1 battery  SIZE: 17500 kWh 4375 kW"""
    results = run_case("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_02.csv", 'dervet')
    assert_file_exists(results, 'load_coverage_prob')


def test_energy_power_sizing_pf():
    """tests sizing with retail ETS and DCM services through 1 battery  SIZE: 17500 kWh 4375 kW"""
    results = run_case("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_03.csv", 'dervet')
    assert_file_exists(results, 'load_coverage_prob')


"""
SIZING MODULE TESTS
"""


@pytest.mark.slow
def test_energy_sizing_da():
    # assert_ran('./Testing/Model_params/battery/051-DA_Reliability_battery_energy_sizing.csv', 'dervet')
    mp_location = './Testing/Model_params/battery/051-DA_Reliability_battery_energy_sizing.csv'
    results = run_case(mp_location, 'dervet')
    check_lcpc(results, mp_location)


def xtest_energy_power_sizing_da():
    # run_case('./Testing/Model_params/battery/056-DA_Reliability_battery_energy_power_sizing.csv', 'dervet')
    mp_location = './Testing/Model_params/battery/056-DA_Reliability_battery_energy_power_sizing.csv'
    results = run_case(mp_location, 'dervet')
    check_lcpc(results, mp_location)


"""
DISPATCH MIX FOR RELIABILITY TESTS
"""


def xtest_retail_fixed_size():
    # TODO skip sizing module get SOE min timeseries before running bill reduction opt
    # assert_ran(r'.\Testing\Ventura\Model_Parameters\Model_Parameters_Template_DER_Ventura_05.csv', 'dervet')
    mp_location = r'.\Testing\Ventura\Model_Parameters\Model_Parameters_Template_DER_Ventura_05.csv'
    results = run_case(mp_location, 'dervet')
    check_lcpc(results, mp_location)
