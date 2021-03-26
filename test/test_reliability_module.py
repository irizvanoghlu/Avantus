"""
This file tests analysis cases have to do with the reliability module.

NOTE there are reliability module tests in test_beta_release_validation_report. This file tests
features that were not showcased through the validation report

"""
from pathlib import Path
from test.TestingLib import *

RESULTS = Path("./test/test_load_shedding/results")
SIZING_RESULTS = Path("./test/test_load_shedding/results/Sizing")
MP = Path("./test/test_load_shedding/mp")
MP_SIZING = Path("./test/test_load_shedding/mp/sizing")
JSON = '.json'
CSV = '.csv'


"""
Load shedding TESTS
"""


MAX_PERCENT_ERROR = 3


class TestLoadShedding:
    def setup_class(self):
        self.mp_name = MP / "Model_Parameters_Template_DER_w_ls1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = RESULTS / Path("./reliability_load_shed1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_forma_2mw_5hr.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "size_2mw_5hr.csv",
                             MAX_PERCENT_ERROR)


class TestWoLoadShedding:

    def setup_class(self):
        self.mp_name = MP / "Model_Parameters_Template_DER_wo_ls1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = RESULTS / Path("./reliability_load_shed_wo_ls1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_forma_2mw_5hr.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "size_2mw_5hr.csv",
                             MAX_PERCENT_ERROR)


class TestSizingLoadShedding:
    def setup_class(self):
        self.mp_name = MP_SIZING / "Model_Parameters_Template_DER_w_ls1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = SIZING_RESULTS / Path("./w_ls1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_forma_2mw_5hr.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "size_2mw_5hr.csv",
                             MAX_PERCENT_ERROR)


class TestSizingWoLoadShedding:
    def setup_class(self):
        self.mp_name = MP_SIZING / "Model_Parameters_Template_DER_wo_ls1.csv"
        self.results = run_case(self.mp_name)
        self.validated_folder = SIZING_RESULTS / Path("./wo_ls1")

    def test_lcpc_exists(self):
        assert_file_exists(self.results, 'load_coverage_prob')

    def test_proforma_results_are_expected(self):
        compare_proforma_results(self.results, self.validated_folder / "pro_forma_2mw_5hr.csv",
                                 MAX_PERCENT_ERROR)

    def test_size_results_are_expected(self):
        compare_size_results(self.results, self.validated_folder / "size_2mw_5hr.csv",
                             MAX_PERCENT_ERROR)
