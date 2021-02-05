"""
This file tests End to end tests passed on the validation report on September
1st. All tests should pass.

The tests in this file can be run with DERVET.

"""

from Testing.TestingLib import run_case, check_lcpc, assert_file_exists, \
    compare_proforma_results, compare_size_results

ERROR_BOUND = 3
usecase1_url = r".\testing\validation_report_Sept1\Model_params\Usecase1"


def test_uc1_1():
    """ 1ESS sizing - BTM with PF reliability calculations"""
    mp_location = r"\Model_Parameters_Template_Usecase1_UnPlanned_ES.csv"
    results = run_case(usecase1_url+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase1\es"
    compare_proforma_results(results, validated_folder+r"\pro_formauc3.csv",
                             ERROR_BOUND)
    compare_size_results(results, validated_folder+r"\sizeuc3.csv",
                         ERROR_BOUND)


def test_uc1_2():
    """ 1ESS sizing, 1PV fixed - BTM with PF reliability calculations"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase1\Model_Parameters_Template_Usecase1_UnPlanned_ES+PV.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase1\ES+PV"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv",
                             ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv",
                         ERROR_BOUND)


def test_uc1_3a():
    """ 1ESS sized, 1PV fixed, 1ICE fixed - user constraints with PF
    reliability calculations"""
    pass


def test_uc1_3b():
    """ 1ESS fixed, 1PV fixed, 1ICE fixed - user constraints with PF
    reliability calculations"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase1\Model_Parameters_Template_Usecase1_UnPlanned_ES+PV+DG_step2.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase1\es+pv+dg_step2"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc2_1a():
    """ 1ESS - size for just reliability"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_Planned_ES.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es\Step1"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es_step1.csv", ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3_es_step1.csv", ERROR_BOUND)


def test_uc2_1b():
    """ 1 ESS- given size , bill reduction and user constraint with PF reliability"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es\Step2"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es_step2.csv", 10) #This is an exception
    compare_size_results(results, validated_folder + r"\sizeuc3_es_step2.csv", ERROR_BOUND)


def test_uc2_2a():
    """ BAT sized for reliability with fixed size PV """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')

    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es+PV\Step1"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step1.csv", ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step1.csv", ERROR_BOUND)
    check_lcpc(results, mp_location)


def test_uc2_2b():
    """ 1ESS, 1PV - fixed size, BTM with user constraint and PF reliabilty calculations"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es+pv\Step2"
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step2.csv", ERROR_BOUND)
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step2.csv", ERROR_BOUND)


def test_uc2_3a():
    """ BAT, PV (fixed size), ICE fixed sized - sized for reliability"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step1.csv"
    results = run_case(mp_location, 'dervet')  # TODO fails
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es+pv+dg\step1"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv+dg_step1.csv", ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv+dg_step1.csv", ERROR_BOUND)


def test_uc2_3b():
    """ fixed size BAT + PV, DCM and retailTimeShift with User constraints and PF reliability calculations"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase2\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase2\es+pv+dg\step2"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv+dg_step2.csv", ERROR_BOUND)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv+dg_step2.csv", ERROR_BOUND)


def test_uc3_p1a():
    """ BAT sizing for a planned outage on one day"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc3_p1b():
    """ Sized BAT, DA + FR + User constraints """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc3_p2a():
    """ BAT sizing for planned outage with fixed PV """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es+pv"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc3_p2b():
    """User constraints + FR + DA with fixed size PV and Battery"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es+pv"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc3_p3a():
    """ BAT sizing + fixed PV + fixed ICE  for reliability"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV+DG.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es+pv+dg"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


def test_uc3_p3b():
    """ fixed sized BAT, PV, ICE for FR + DA"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV+DG_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es+pv+dg"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


# def test_uc3_up1a():      SAME AS UC2-1A
#     """ BAT sized for reliability """
#     mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Unplanned\Model_Parameters_Template_Usecase3_Planned_ES.csv"
#     results = run_case(mp_location, 'dervet')
#     assert_file_exists(results, 'load_coverage_prob')
#     check_lcpc(results, mp_location)
#     validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\es"
#     compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", ERROR_BOUND)
#     compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


# def test_uc3_up2a():      SAME AS UC2-2A
#     """ BAT sized for reliability with fixed size PV """
#     mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Unplanned\Model_Parameters_Template_Usecase3_Planned_ES+PV.csv"
#     results = run_case(mp_location, 'dervet')
#     assert_file_exists(results, 'load_coverage_prob')
#     check_lcpc(results, mp_location)
#     validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\es+pv"
#     compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", ERROR_BOUND)
#     compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


# def test_uc3_up3a():      SAME AS UC2-3A
#     """BAT sizing, fixed PV + fixed ICE for reliability"""
#     mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Unplanned\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG.csv"
#     results = run_case(mp_location, 'dervet')
#     assert_file_exists(results, 'load_coverage_prob')
#     check_lcpc(results, mp_location)
#     validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\es+pv+dg"
#     compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", ERROR_BOUND)
#     compare_size_results(results, validated_folder + r"\sizeuc3.csv", ERROR_BOUND)


usecase3_unplanned_step2 = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Unplanned\Step2_Wholesale"


def test_uc3_up1b():
    """ FR + DA + UserConstraints, BAT fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, usecase3_unplanned_step2+mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es_step2.csv", ERROR_BOUND)


def test_uc3_up2b():
    """ FR + DA + UserConstraints, BAT + PV fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, usecase3_unplanned_step2+mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es+pv1"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step2.csv", ERROR_BOUND)


def test_uc3_up3b():
    """ FR + DA + UserConstraints, BAT + PV fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es+pv+dg"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step2.csv", ERROR_BOUND)
