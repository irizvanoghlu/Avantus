# DER-VET™

DER-VET™ provides a free, publicly accessible, open-source platform for calculating, understanding, and optimizing the value of distributed 
energy resources (DER) based on their technical merits and constraints. An extension of EPRI's [StorageVET®](./storagevet) tool, DER-VET supports 
site-specific assessments of energy storage and additional DER technologies—including solar, wind, demand response, electric vehicle charging, 
internal combustion engines, and combined heat and power—in different configurations, such as microgrids. It uses load and other data to determine 
optimal size, duration, and other characteristics for maximizing benefits based on site conditions and the value that can be extracted from targeted 
use cases. Customers, developers, utilities, and regulators across the industry can apply this tool to inform project-level decisions based on sound 
technical understanding and unbiased cost-performance data.

DER-VET was developed with funding from the California Energy Commission. EPRI plans to support continuing updates and enhancements.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for 
notes on how to deploy the project on a live system.

### Prerequisites & Installing

#### 1. Install [Anaconda](https://www.anaconda.com/download/) for python 3.**

#### 2. Open Anaconda Prompt

#### 3. Activate Python 3.6 environment

On Linux/Mac   
Note that pip should be associated to a python 3.6 installation  
```
pip install virtualenv
virtualenv dervet-venv
source dervet-venv/bin/activate
```
On Windows  
Note that pip should be associated to a python 3.6 installation    
```
pip install virtualenv
virtualenv dervet-venv
"./dervet-venv/Scripts/activate"
```
With Conda
Note that the python version is specified, meaning conda does not have to be associated with a python 3.6
```
conda create -n dervet-venv python=3.6
conda activate dervet-venv
```

#### 3. Install project dependencies
 
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e ./storagevet
```

## Running the tests

To run tests, activate Python environment. Then enter the following into your terminal:
```
python -m pytest test
```

## Deployment

To use this project as a dependency in your own, clone this repo directly into the root of your project.
Open terminal or command prompt from your project root, and input the following command:
```
pip install -e ./dervet
```

## Versioning

We use [Gitlab](https://gitlab.epri.com/storagevet/storagevet) for versioning. For the versions available, 
see the [tags on this repository](https://gitlab.epri.com/storagetvet/storagevet/tags). 

## Authors

* **Miles Evans**
* **Ramakrishnan Ravikumar**
* **Suma Jothibasu**
* **Andres Cortes**
* **Halley Nathwani**
* **Andrew Etringer**
* **Evan Giarta**
* **Thien Nguyen**
* **Micah Botkin-Levy**
* **Yekta Yazar**
* **Kunle Awojinrin**
* **Arindam Maitra**
* **Giovanni Damato**


## License

This project is licensed under the BSD (3-clause) License - see the [LICENSE.txt](./LICENSE.txt) file for details

