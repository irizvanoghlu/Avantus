"""
This file tests features of the CBA module. All tests should pass.

The tests in this file can be run with DERVET.

"""
import pytest
from Testing.TestingLib import assert_ran, run_case
from ErrorHandelling import *


CBA_DIR = r".\Testing\cba_validation\Model_params"
BAT_DIR = r'./Testing/Model_params/battery'
MP_DIR = r".\Testing\Model_params"
DERVET = 'dervet'


"""
Evaluation column tests
TODO check that post-facto cash flows are ZERO
TODO add test to make sure that evaluation data of different dt is caught
"""


def test_missing_tariff():
    # following should fail
    with pytest.raises(ModelParameterError):
        run_case(rf'{BAT_DIR}/battery/099-missing_tariff.csv',
                 DERVET)


def test_energy_sensitivity():
    assert_ran(rf'{BAT_DIR}/100-bat_energy_sensitivity.csv', DERVET)


def test_coupling():
    assert_ran(rf'{BAT_DIR}/101-bat_timeseries_dt_sensitivity_couples.csv',
               DERVET)


def test_valuation_wo_data():
    assert_ran(rf'{BAT_DIR}/102-cba_valuation.csv', DERVET)


def test_sensitivity_evaluation():
    assert_ran(rf'{BAT_DIR}/103-cba_valuation_sensitivity.csv', DERVET)


def test_coupled_evaluation():
    assert_ran(rf'{BAT_DIR}/104-cba_valuation_coupled_dt.csv', DERVET)


def test_coupled_dt_timseries_error():
    # following should fail
    with pytest.raises(ModelParameterError):
        run_case(rf'{BAT_DIR}/105-coupled_dt_timseries_error.csv', DERVET)


def test_tariff():
    assert_ran(rf'{BAT_DIR}/106-cba_tariff.csv', DERVET)


def test_monthly():
    assert_ran(rf'{BAT_DIR}/107-cba_monthly.csv', DERVET)


def test_catch_wrong_length():
    # following should fail
    with pytest.raises(ModelParameterError):
        assert_ran(rf'{BAT_DIR}/103-catch_wrong_length.csv', DERVET)


"""
Analysis end mode tests
"""


def test_longest_lifetime_no_replacement():
    # 1) decomissioning costs show up on the last year the equipment is expect
    #   to last DONE
    # 2) no replacement costs DONE
    # 3) proforma should be the length of the longest life time + 1 year DONE
    # 4) no costs for a DER after the end of its expected life time DONE
    assert_ran(rf"{MP_DIR}\longest_lifetime.csv", DERVET)


def test_longest_lifetime_replacement():
    # 1) check to make sure there are replacement costs on the years that a new
    #   DER is installed DONE
    # 2) all decomissioning costs should be on the last year DONE
    # 3) proforma should be the length of the longest life time + 1 year DONE
    assert_ran(rf"{MP_DIR}\longest_lifetime_replaceble.csv",
               DERVET)


def test_shortest_lifetime_no_replacement():
    # 1) no replacement costs DONE
    # 2) proforma should be the length of the shortest life time + 1 year DONE
    # 3) all decomissioning costs should be on the last year DONE
    # 4) proforma should be the same as shortest_lifetime_replacement test DONE
    assert_ran(fr"{MP_DIR}\shortest_lifetime.csv", DERVET)


def test_shortest_lifetime_replacement():
    # 1) no replacement costs DONE
    # 2) proforma should be the length of the shortest life time + 1 year DONE
    # 3) all decomissioning costs should be on the last year DONE
    # 4) proforma should be the same as shortest_lifetime test DONE
    assert_ran(fr"{MP_DIR}\shortest_lifetime_replaceble.csv", DERVET)


# mode==2 + a DER is being sized
def test_shortest_lifetime_sizing_error():
    with pytest.raises(Exception):
        run_case(fr"{MP_DIR}\shortest_lifetime_sizing_error.csv", DERVET)


# mode==3 + a DER is being sized
def test_longest_lifetime_sizing_error():
    with pytest.raises(Exception):
        run_case(fr"{MP_DIR}\longest_lifetime_sizing_error.csv", DERVET)


"""
End of life cost tests
"""


def test_linear_salvage_value():
    # check to make sure that salvage value is some nonzero value DONE
    assert_ran(CBA_DIR + r"\110-linear_salvage_value.csv", DERVET)


def test_user_defined_salvage_value():
    # check to make sure that salvage value is some nonzero value DONE
    assert_ran(CBA_DIR + r"\user_salvage_value.csv", DERVET)


def test_shortest_lifetime_linear_salvage():
    assert_ran(CBA_DIR + r"\shortest_lifetime_linear_salvage.csv", DERVET)


def xtest_decomissioning_costs():
    """decomissioning cost column should have a none zero value"""
    assert_ran(r" ", DERVET)    # TODO


"""
Non-initial investment payment options: PPA, ECC
"""


# mode==4 + e==d
def xtest_carrying_cost_d_is_e_error():
    with pytest.raises(Exception): # TODO
        run_case(CBA_DIR + r"\109-carrying_cost_d_is_e_error.csv", DERVET)


def test_ppa():
    """ Test solar's PPA feature"""
    assert_ran(CBA_DIR + r"\ppa_payment.csv", DERVET)


def xtest_carrying_cost_replacable():
    """ 3 DERs all replaceable for the duration of the project
    This test will check for :
        1) proforma is length of lifetime + 1
        2) no replacement costs
        3) decomissioning cost should be on the last year
        4) capital cost column should be replaced with economic carrying
            capacity
        5) in proforma, economic carrying capacity column should have a value
            for every row except CAPEX Year
        6) taxes dont appear in proforma
        7) ecc breakdown CSV in results
    """
    assert_ran(r" ", DERVET)    # TODO


def xtest_carrying_cost_not_replacable():
    """ 3 DERs all not replaceable for the duration of the project
    This test will check for :
        1) proforma is length of lifetime + 1
        2) replacement costs
        3) decomissioning cost should be on the last year
        4) capital cost column should be replaced with economic carrying
            capacity
        6) taxes dont appear in proforma
        7) ecc breakdown CSV in results
    """
    assert_ran(r" ", DERVET)    # TODO


def test_carrying_cost_error():
    """ 3 DERs not all are replaceable"""
    # ECC should be run in a Reliability/Deferral case
    with pytest.raises(ModelParameterError):
        run_case(CBA_DIR + r"\108-carrying_cost_eccPerc_error.csv", DERVET)


def xtest_ecc_zero_out():
    """ Test that value from services are 0-ed"""
    # TODO
    assert_ran(r" ", DERVET)


def xtest_ecc_shorter_actual_lifetime():
    """ Test ECC calculations when batteries degradation module results in a 
        shorter lifetime than user given"""    # TODO
    assert_ran(r" ", DERVET)


def xtest_ecc_long_actual_lifetime():
    """ Test ECC calculations when batteries degradation module results in a 
        longer lifetime than user given"""    # TODO
    assert_ran(r" ", DERVET)


"""
All other tests for the cost benefit analysis and financials class
"""


def test_da_month_degradation_predict_when_battery_will_be_replaced():
    assert_ran(CBA_DIR +
               r"\Model_Parameters_Template_ENEA_S1_8_12_UC1_DAETS_" +
               r"doesnt_reach_eol_during_opt.csv", DERVET)


def xtest_esclation_btw_analysis_years():
    """ Test rate of esclation btw analysis years if applied correctly.
    This test will also check for:
        - inflation rate applies to services
        - TER apples to equipement costs
        - service value on the year that analysis occured for are not written
            over
    """
    assert_ran(r" ", DERVET)    # TODO
