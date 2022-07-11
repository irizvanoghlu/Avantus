# Changelog
Notable changes to DER-VET are documented in this CHANGELOG.md.

Questions and feedback can be submitted to the Electric Power Research Institute (EPRI) from the Survey Monkey feedback linked on the StorageVET website (https://www.storagevet.com/).

The format is based on [Keep a Changelog] (https://keepachangelog.com/en/1.0.0/).

## [1.2.2] - 2022-05-05 to 2022-07-07
### Added
- pytests for small tweaks to the default model parameters CSV
  - create instances of infeasibility with bad data in the input time series
  - setting ts-limit booleans ON should not fail
### Changed
- improved handling of input time series data
  - new method: get_single_series()
  - introduce more specific error classes: TimeseriesDataError and TimeseriesMissingError
  - better error and warning messaging and readability
- sets all min and max battery constraint values to empty in the default input time series
- adds column: LF Price ($/kW) to default time series
- improvements to the migration script
  - when the input is already v1.2, exit with a clear message
  - have better error messages for poorly formatted json inputs
### Fixed
- the broken reliability service, when post_facto_only is OFF, has been fixed
  - Adds back the SysEneMinReq constraint
  - NOTE: this was mistakenly removed in a commit from August 2021
  - adds a pytest to ensure that when an optimization loop is run, with Reliability active,
    the first X hours in the load_coverage_prob.csv file are 1 meaning 100 percent covered

## [1.2.1] - 2022-03-31 to 2022-05-04
### Added
- adds warning message to ignore the duration_max parameter when not sizing a battery

## [1.2.0] - 2021-09-10 to 2022-03-30
### Added
- added a migrations/migrate_project_DERVET_GUI.py script
  - this will transform a project exported from the GUI v1.1.2 into GUI v1.2.0
- pytests to ensure that the default model parameter CSV file runs when input into run_DERVET.py
- pytests to ensure that with each technology active along with a battery, the default model parameter CSV runs
- adds required rows for all technologies in the default model parameter csv
- adds a new scenario tag input to allow/disallow an electric load_dump
- a copy of the model parameters input CSV is now copied to the Results folder for each run
- adds thermal technologies: CHP, Boiler, and Chiller
  - a Chiller can serve a cooling load only, and can be powered by electricity, natural gas, or heat
  - a Boiler can serve a heating load (hot water and/or steam), and can be powered by electricity or natural gas
  - a CHP can serve a heating load (hot water and/or steam), and an electrical load
  - an active thermal technology requires the appropriate thermal input time series data

### Changed
- upgrade supported/recommended Python version to 3.8.13
  - Python package requirements have been updated
  - Update installation instructions: Python environment creation, conda-route, pip-route
- re-structures how fuel costs are handled (see storagevet CHANGELOG)
- force use of the GLPK_MI solver when a project has an active thermal technology
- limit MACRS term to no greater than 20

### Removed
- remove incl_thermal_load boolean from model parameter inputs

### Fixed
- disallow sizing of CAES since it has not been validated

## [1.1.2] - 2021-08-04 to 2021-09-09
### Changed
- Changed the expected type to float for yearly_degrade battery input

### Fixed
- Degradation Fix: more descriptive column header names on Results files
- Simplifies system_requirements infeasibility checks
- Fix to allow minimum battery sizing user constraints to work

## [1.1.1] - 2021-07-09 to 2021-08-03
### Fixed
- Removed comma from soc_target description in the Model Parameters CSV

## [1.1.0] - 2021-04-14 to 2021-07-09
### Added
- this CHANGELOG.md file
- useful error messaging and warning for extreme soc_target values with reliability
- all growth rates have a minimum value of -100 percent
- Fleet EV will output the Baseline Load time series

### Changed
- description of battery soc_target updated for reliability based ES sizing
- modified the README.md with better and more thorough instructions for Installation
- increased the max limit (hours) on optimization window to be 8784

### Fixed
- corrected the logic and docstrings in ParamsDER class bad_active_combo method
- load_technology bug regarding names_list was fixed
