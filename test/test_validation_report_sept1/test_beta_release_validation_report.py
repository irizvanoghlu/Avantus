"""
This file tests End to end tests passed on the validation report on September
1st. All tests should pass.

The tests in this file can be run with DERVET.

"""
from pathlib import Path
from test.TestingLib import *

MAX_PERCENT_ERROR = 3
TEST_DIR = Path("./test/test_validation_report_sept1")

USECASE1 = Path("./Model_params/Usecase1")


class TestUseCase1A:
    """ 1ESS sizing - BTM with PF reliability calculations"""
    def setup_class(self):
        mp_name = "Model_Parameters_Template_Usecase1_UnPlanned_ES.csv"
        self.results = run_case(TEST_DIR / USECASE1 / mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR+2)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR-1)


class TestUseCase1B:
    """ 1ESS sizing, 1PV fixed - BTM with PF reliability calculations"""
    def setup_class(self):
        mp_name = "Model_Parameters_Template_Usecase1_UnPlanned_ES+PV.csv"
        self.results = run_case(TEST_DIR / USECASE1 / mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/ES+PV")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR+1)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR-1)


""" Skip Usecase 1C part 1, as it is a duplicate -- 1ESS sized, 1PV fixed, 1ICE fixed - user 
constraints with PF reliability calculations"""


class TestUseCase1C:
    """ Part 2 of Usecase 1C - 1ESS fixed, 1PV fixed, 1ICE fixed - user constraints with PF
    reliability calculations"""
    def setup_class(self):
        mp_name = "Model_Parameters_Template_Usecase1_UnPlanned_ES+PV+DG_step2.csv"
        self.results = run_case(TEST_DIR / USECASE1 / mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es+pv+dg_step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


USECASE2 = Path("./Model_params/Usecase2")


class TestUseCase2A1:
    """ Part 1 of Usecase 2A - 1ESS - size for just reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/Step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step1.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step1.csv", .1)


class TestUseCase2A2:
    """ Part 2 of Usecase 2A - 1 ESS - given size , bill reduction and user constraint with PF
    reliability"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/Step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 11)  # This is an exception

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             MAX_PERCENT_ERROR-1)


class TestUsecase2B1:
    """ Part 1 of Usecase 2B - BAT sized for reliability with fixed size PV"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+PV/Step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv_step1.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step1.csv",
                             MAX_PERCENT_ERROR)


class TestUsecase2B2:
    """ Part 2 of Usecase 2B - 1ESS, 1PV - fixed size, BTM with user constraint and PF
    reliabilty calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv/Step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv_step2.csv",
                                 MAX_PERCENT_ERROR+2)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             MAX_PERCENT_ERROR)


class TestUsecase2C1:
    """ Part 1 of Usecase 2C - BAT, PV (fixed size), ICE fixed sized - sized for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv+dg/step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv+dg_step1.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv+dg_step1.csv",
                             MAX_PERCENT_ERROR)


class TestUsecase2C2:
    """ Part 2 of Usecase 2C - fixed size BAT + PV, DCM and retailTimeShift with User constraints
    and PF reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv+dg/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv+dg_step2.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv+dg_step2.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3A1:
    """ Part 1 of Usecase 3A - BAT sizing for a planned outage on one day"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/Step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 11)  # This is an exception

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             MAX_PERCENT_ERROR - 1)


def xtest_uc3_p1a():
    """ """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


def xtest_uc3_p1b():
    """ Sized BAT, DA + FR + User constraints """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


def xtest_uc3_p2a():
    """ BAT sizing for planned outage with fixed PV """
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es+pv"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


def xtest_uc3_p2b():
    """User constraints + FR + DA with fixed size PV and Battery"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es+pv"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


def xtest_uc3_p3a():
    """ BAT sizing + fixed PV + fixed ICE  for reliability"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV+DG.csv"
    results = run_case(mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\es+pv+dg"
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


def xtest_uc3_p3b():
    """ fixed sized BAT, PV, ICE for FR + DA"""
    mp_location = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Planned\Model_Parameters_Template_Usecase3_Planned_ES+PV+DG_Step2.csv"
    results = run_case(mp_location, 'dervet')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Planned\step2\es+pv+dg"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3.csv", MAX_PERCENT_ERROR)


""" Skip Usecase 3 unplanned case A part 1, as it is a duplicate SAME AS UC2-A1 
-- BAT sized for reliability"""

""" Skip Usecase 3 unplanned case B part 1, as it is a duplicate SAME AS UC2-B1 
-- BAT sized for reliability with fixed size PV"""

""" Skip Usecase 3 unplanned case C part 1, as it is a duplicate SAME AS UC2-C1 
-- BAT sizing, fixed PV + fixed ICE for reliability"""

usecase3_unplanned_step2 = r".\Testing\validation_report_Sept1\Model_params\Usecase3\Unplanned\Step2_Wholesale"


def xtest_uc3_up1b():
    """ FR + DA + UserConstraints, BAT fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, usecase3_unplanned_step2+mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es_step2.csv", MAX_PERCENT_ERROR)


def xtest_uc3_up2b():
    """ FR + DA + UserConstraints, BAT + PV fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    check_lcpc(results, usecase3_unplanned_step2+mp_location)
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es+pv1"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step2.csv", MAX_PERCENT_ERROR)


def xtest_uc3_up3b():
    """ FR + DA + UserConstraints, BAT + PV fixed size with PF reliability"""
    mp_location = r"\Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
    results = run_case(usecase3_unplanned_step2+mp_location, 'dervet')
    assert_file_exists(results, 'load_coverage_prob')
    validated_folder = r".\Testing\validation_report_Sept1\Results\Usecase3\Unplanned\step2_ws\es+pv+dg"
    compare_proforma_results(results, validated_folder + r"\pro_formauc3_es+pv_step2.csv", 10)
    compare_size_results(results, validated_folder + r"\sizeuc3_es+pv_step2.csv", MAX_PERCENT_ERROR)
