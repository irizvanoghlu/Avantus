"""
This file tests the optimization sizing module. All tests should pass.

The tests in this file can be run with DERVET.

"""

import pytest
from Testing.TestingLib import assert_ran, run_case
from ErrorHandelling import *


def test_bat_power_sizing():
    """tests power sizing with retail ETS and DCM services through 1 battery"""
    assert_ran("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_01.csv", 'dervet')


def test_bat_energy_sizing():
    """tests energy sizing with retail ETS and DCM services through 1 battery"""
    assert_ran("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_02.csv", 'dervet')


def test_bat_energy_power_sizing():
    """tests sizing with retail ETS and DCM services through 1 battery"""  # fails on cba-bugs (problem is unbounded)
    assert_ran("./Testing/Ventura/Model_Parameters/Model_Parameters_Template_DER_Ventura_03.csv", 'dervet')


def test_bat_da_energy_sizing():
    assert_ran('./Testing/Model_params/battery/050-DA_battery_energy_sizing.csv', 'dervet')


@pytest.mark.slow  # took 2H 50M to finish
def test_bat_retail_energy_power_sizing():
    assert_ran('./Testing/Model_params/battery/054-retail_energy_power_sizing.csv', 'dervet')


def test_bat_retail_power_sizing_binary():
    with pytest.raises(ParameterError):
        run_case('./Testing/Model_params/battery/057-retail_power_sizing_binary.csv', 'dervet')


def test_ice_bat_pv_sizing_bill_reduction():
    run_case(r".\Testing\Model_params\001-bill_reduction_sizing.csv", 'dervet')


def test_pv_ice_sizing_bill_reduction():
    run_case(r".\Testing\Model_params\001-bill_reduction_sizing_pv_ice.csv", 'dervet')


def test_batt_deferral():
    run_case(r".\Testing\Model_params\Model_Parameters_deferral_sizing.csv", 'dervet')


def test_pv_sizing_bill_reduction():
    run_case(r".\Testing\Model_params\001-bill_reduction_sizing_pv.csv", 'dervet')


def test_ice_sizing_bill_reduction():
    run_case(r".\Testing\Model_params\000-billreduction_ice_sizing.csv", 'dervet')


@pytest.mark.slow  # took 40M to finish
def test_bat_power_market_constraints():
    run_case(r".\Testing\Model_params\sizing_bat_power_market_constraints.csv", 'dervet')


def test_ice_power_market_constraints():
    run_case(r".\Testing\Model_params\sizing_ice_power_market_constraints.csv", 'dervet')
