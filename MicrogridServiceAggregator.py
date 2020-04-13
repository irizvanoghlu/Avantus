__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Thien Nguyen', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = "x.x.x"

from storagevet.Controller import Controller
from ValueStreamsDER.Reliability import Reliability


class MicrogridController(Controller):
    """ ***NAME IS NOT FINAL*** The entity that tracks the value streams and bids the Microgrid's capabilities
    into energy markets

    """

    def __init__(self, reliability, deferral, dr, ra, backup, volt, user, daets, fr, lf, sr, nsr, dcm, retailets):
        """Initialized the value streams that the DERs will be evaluated under"""
        super().__init__(deferral, dr, ra, backup, volt, user, daets, fr, lf, sr, nsr, dcm, retailets)
        self.inputs_map['Reliability'] = reliability

    def initialize_valuestreams(self, opt_years, frequency):
        if self.inputs_map['Reliability']:
            inputs = self.inputs_map['Reliability']
            new_service = Reliability(inputs)
            new_service.estimate_year_data(opt_years, frequency)
            self.value_streams['Reliability'] = new_service
            self.inputs_map.pop('Reliability')

        super().initialize_valuestreams(opt_years, frequency)

