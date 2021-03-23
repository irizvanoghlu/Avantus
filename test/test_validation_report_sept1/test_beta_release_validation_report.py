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


class TestUseCase1EssSizing4Btm:
    """ 1ESS sizing - BTM with PF reliability calculations"""
    def setup_class(self):
        self.mp_name = "Model_Parameters_Template_Usecase1_UnPlanned_ES.csv"
        self.results = run_case(TEST_DIR / USECASE1 / self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def xtest_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR+2)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR-1)


class TestUseCase1EssSizingPv4Btm:
    """ 1ESS sizing, 1PV fixed - BTM with PF reliability calculations"""
    def setup_class(self):
        self.mp_name = "Model_Parameters_Template_Usecase1_UnPlanned_ES+PV.csv"
        self.results = run_case(TEST_DIR / USECASE1 / self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase1/es+pv")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def xtest_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR+1)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR-1)


class TestUseCase1EssPvIce4UserConstraints:
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


class TestUseCase2EssSizing4Reliability:
    """ Part 1 of Usecase 2A - 1ESS - size for just reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / "Model_Parameters_Template_Usecase3_Planned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/step1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step1.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step1.csv", .1)


class TestUseCase2Ess4BtmUserConstraints:
    """ Part 2 of Usecase 2A - 1 ESS - given size , bill reduction and user constraint with PF
    reliability"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             MAX_PERCENT_ERROR-1)


class TestUsecase2EssSizingPv4Reliability:
    """ Part 1 of Usecase 2B - BAT sized for reliability with fixed size PV"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv/step1")

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


class TestUsecase2EssPv4BtmUserConstraints:
    """ Part 2 of Usecase 2B - 1ESS, 1PV - fixed size, BTM with user constraint and PF
    reliabilty calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv_step2.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             MAX_PERCENT_ERROR)


class TestUsecase2EssSizingPvIce4Reliability:
    """ Part 1 of Usecase 2C - BAT, PV (fixed size), ICE fixed sized - sized for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step1.csv"
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


class TestUsecase2EssPvIce4BtmUserConstraints:
    """ Part 2 of Usecase 2C - fixed size BAT + PV, DCM and retailTimeShift with User constraints
    and PF reliability calculations"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase2/es+pv+dg/step2")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3_es+pv+dg_step2.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv+dg_step2.csv",
                             MAX_PERCENT_ERROR)


USECASE3PLANNED = Path("./Model_params/Usecase3/Planned")


class TestUseCase3EssSizing4PlannedOutage:
    """ Part 1 of Usecase 3A - BAT sizing for a planned outage on one day"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Planned/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3Ess4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3A - Sized BAT, DA + FR + User constraints"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Planned/step2/es")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3EssSizingPv4PlannedOutage:
    """ Part 1 of Usecase 3A - BAT sizing for planned outage with fixed PV"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Planned/es+pv")

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3EssPv4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3A - User constraints + FR + DA with fixed size PV and Battery"""

    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Planned/step2/es+pv")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3EssSizingPvIce4PlannedOutage:
    """ Part 1 of Usecase 3 Planned C - BAT sizing + fixed PV + fixed ICE  for reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV+DG.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/planned/es+pv+dg")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


class TestUseCase3EssPvIce4DaFrUserConstraintsPlannedOutage:
    """ Part 2 of Usecase 3 Planned C - fixed sized BAT, PV, ICE for FR + DA"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3PLANNED / \
                       "Model_Parameters_Template_Usecase3_Planned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Planned/step2/es+pv+dg")

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results,
                                 self.validated_folder / "pro_formauc3.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3.csv",
                             MAX_PERCENT_ERROR)


USECASE3UNPLANNED_STEP2 = Path("./Model_params/Usecase3/Step2_Wholesale")


class TestUseCase3Ess4DaFrUserConstraintsUnplannedOutage:
    """ Part 2 of Usecase 3 Unplanned A - FR + DA + UserConstraints, BAT fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED_STEP2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Unplanned/step2_ws/es")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder / "pro_formauc3_es_step2.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es_step2.csv",
                             MAX_PERCENT_ERROR-1)


class TestUseCase3EssPv4DaFrUserConstraintsUnplannedOutage:
    """ Part 2 of Usecase 3 Unplanned B - FR + DA + UserConstraints, BAT + PV fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED_STEP2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Unplanned/step2_ws/es+pv1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder /
                                 "pro_formauc3_es+pv_step2.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             MAX_PERCENT_ERROR-1)


class TestUseCase3EssPvIce4DaFrUserConstraintsUnplannedOutage:
    """ Part 2 of Usecase 3 Unplanned C - FR + DA + UserConstraints, BAT + PV fixed size with PF
    reliability"""
    def setup_class(self):
        self.mp_name = TEST_DIR / USECASE3UNPLANNED_STEP2 / \
                       "Model_Parameters_Template_Usecase3_UnPlanned_ES+PV+DG_Step2.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = TEST_DIR / Path("./Results/Usecase3/Unplanned/step2_ws/es+pv+dg")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_lcpc_meets_target(self):
        check_lcpc(self.results, self.mp_name)

    def test_proforma_results_are_expected(self):
        opt_years = self.results.instances[0].opt_years
        compare_proforma_results(self.results, self.validated_folder /
                                 "pro_formauc3_es+pv_step2.csv",
                                 MAX_PERCENT_ERROR, opt_years)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "sizeuc3_es+pv_step2.csv",
                             MAX_PERCENT_ERROR-1)
