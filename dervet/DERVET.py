"""
runDERVET.py

This Python script serves as the initial launch point executing the
Python-based version of DERVET.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). ' \
                'All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta',
               'Halley Nathwani', "Thien Nguyen", 'Kunle Awojinrin']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'

import time
import argparse

from dervet.MicrogridScenario import MicrogridScenario
from dervet.DERVETParams import ParamsDER
from dervet.MicrogridResult import MicrogridResult
from storagevet.ErrorHandelling import *


class DERVET:
    """ DERVET API. This will eventually allow StorageVET to be imported and
    used like any other python library.

    """

    def __init__(self, model_parameters_path, verbose=False, **kwargs):
        """
            Constructor to initialize the parameters and data needed to run

            Args:
                model_parameters_path (str): Filename of the model parameters
                    CSV or XML that describes the optimization case to be
                    analysed

            Notes: kwargs is in place for testing purposes
        """
        self.verbose = verbose

        # Initialize Params Object from Model Parameters and Simulation Cases
        self.cases = ParamsDER.initialize(model_parameters_path, self.verbose)
        self.results = MicrogridResult.initialize(ParamsDER.results_inputs,
                                                  ParamsDER.case_definitions)

        if self.verbose:
            from storagevet.Visualization import Visualization
            Visualization(ParamsDER).class_summary()

    def solve(self):
        starts = time.time()

        for key, value in self.cases.items():
            run = MicrogridScenario(value)
            run.set_up_poi_and_service_aggregator()
            run.initialize_cba()
            run.fill_and_drop_extra_data()
            run.sizing_module()
            run.optimize_problem_loop()

            MicrogridResult.add_instance(key, run)

        MicrogridResult.sensitivity_summary()

        ends = time.time()
        TellUser.info(f"DERVET runtime: {ends - starts}")

        return MicrogridResult
