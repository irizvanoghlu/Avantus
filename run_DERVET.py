"""
runDERVET.py

This Python script serves as the initial launch point executing the Python-based version of DERVET
(AKA StorageVET 2.0 or SVETpy).
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Evan Giarta', 'Miles Evans']
__email__ = ['egiarta@epri.com', 'mevans@epri.com']

import sys
from pathlib import Path
import os.path
import logging
import time
from datetime import datetime
import argparse
import pandas as pd

# ADD STORAGEVET TO PYTHONPATH BEFORE IMPORTING ANY LIBRARIES OTHERWISE IMPORTERROR

# dervet's directory path is the first in sys.path
# determine storagevet path (absolute path)
storagevet_path = os.path.join(sys.path[0], 'storagevet')

# add storagevet (source root) to PYTHONPATH
sys.path.insert(0, storagevet_path)

from ScenarioSizing import ScenarioSizing
from ParamsDER import ParamsDER as Params
from cbaDER import CostBenefitAnalysis
from ResultDER import ResultDER as Result

# TODO: make multi-platform by using path combine functions

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')

class DERVET:
    """ DERVET API. This will eventually allow StorageVET to be imported and used like any
    other python library.

    """

    def __init__(self, model_parameters_path, schema_path):
        """
            Constructor to initialize the parameters and data needed to run StorageVET\

            Args:
                model_parameters_path (str): Filename of the model parameters CSV or XML that
                    describes the optimization case to be analysed
                schema_path (str): relative path to the Schema.xml that storagevet uses
        """
        if model_parameters_path.endswith(".csv"):
            opt_model_parameters_path = Params.csv_to_xml(model_parameters_path)
        else:
            opt_model_parameters_path = model_parameters_path

        # Initialize the Params Object from Model Parameters and Simulation Cases
        # should we leave the name as ParamsDER instead of Params for easier identification? - TN
        Params.initialize(opt_model_parameters_path, schema_path)
        u_logger.info('Successfully initialized the Params class with the XML file.')

        # Initialize the CBA module
        # CostBenefitAnalysis.initialize_evaluation()
        u_logger.info('Successfully initialized the CBA class with the XML file.')

        self.model_params = Params

    def solve(self):
        verbose = self.model_params.instances[0].Scenario['verbose']
        if verbose:
            self.model_params.class_summary()
            self.model_params.series_summary()
        self.model_params.validateDER()
        self.run()

    def run(self):
        starts = time.time()

        # should we leave the name as ResultDER instead of Result for easier identification? - TN
        Result.initialize(self.model_params.Results, self.model_params.df_analysis)

        for key, value in self.model_params.instances.items():
            if not value.other_error_checks():
                continue
            value.prepare_scenario()
            value.prepare_technology()
            value.prepare_services()
            value.prepare_finance()

            run = ScenarioSizing(value)
            run.add_technology()
            run.add_services()
            run.add_control_constraints()
            run.optimize_problem_loop()

            Result.add_instance(key, run)

        Result.calculate()
        Result.save_to_disk()
        ends = time.time()
        print("DERVET runtime: ")
        print(ends - starts)


if __name__ == '__main__':
    """
        the Main section for runStorageVET to run by itself without the SVETapp GUI 
    """

    parser = argparse.ArgumentParser(prog='StorageVET.py',
                                     description='The Electric Power Research Institute\'s energy storage system ' +
                                                 'analysis, dispatch, modelling, optimization, and valuation tool' +
                                                 '. Should be used with Python 3.6.x, pandas 0.19+.x, and CVXPY' +
                                                 ' 0.4.x or 1.0.x.',
                                     epilog='Copyright 2018. Electric Power Research Institute (EPRI). ' +
                                            'All Rights Reserved.')
    parser.add_argument('parameters_filename', type=str,
                        help='specify the filename of the CSV file defining the PARAMETERS dataframe')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='specify this flag for verbose output during execution')
    parser.add_argument('--gitlab-ci', action='store_true',
                        help='specify this flag for gitlab-ci testing to skip user input')
    arguments = parser.parse_args()

    script_rel_path = sys.argv[0]
    dir_rel_path = script_rel_path[:-len('run_DERVET.py')]
    schema_rel_path = dir_rel_path + "SchemaDER.xml"

    case = DERVET(arguments.parameters_filename, schema_rel_path)
    case.solve()

    # print("Program is done.")
