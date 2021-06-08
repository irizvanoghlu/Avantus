# DER-VET™

[DER-VET™](https://der-vet.com) provides a free, publicly accessible, open-source platform for calculating, understanding, and optimizing the value 
of distributed 
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

### New Installation

#### 1. Clone/download this repository onto your local computer.

#### 2. Clone/download the sub-repository, [StorageVET](https://github.com/epri-dev/StorageVET), onto your local computer in the root of the dervet folder (created in the previous step).

#### 3. Install [Anaconda](https://www.anaconda.com/products/individual) for python 3.**
Please note that it is recommended for Windows users to install and use Anaconda

#### 4. Install system requirements
On Windows
Install the Build Tools for [Visual Studio](https://visualstudio.microsoft.com/downloads/). When prompted by the installer, select C++ build tools and the appropriate Windows SDK specified below and install.

     | Windows OS   | SDK        |
     |--------------|------------|
     | Windows 7    | Windows 8.1|
     | Windows 8.1  | Windows 8.1|
     | Windows 10   | Windows 10 |

On Mac
Install [Xcode](https://developer.apple.com/xcode/) and [GLPK](https://formulae.brew.sh/formula/glpk)

#### 5. Open Anaconda Prompt (Windows) or a corresponding shell/terminal/console/prompt in **administrator mode**

#### 6. Navigate to your "dervet" folder
This is the location of the repository on your computer.

#### 7. Create Python 3.6 environment
We give the user 2 paths to create a python environment. Each path results in a siloed python environment, but with different properties.
Choose path A or B and stick to it--commands are not interchangeable. 
You will need to activate the environment to run the model, always. This is the next step. 
**Please remember which environment is created in order to activate it again later.**

##### Path A
On Linux/Mac/Windows  
Note that pip should be associated to a python 3.6 installation  
```
pip install virtualenv
virtualenv dervet-venv
```
#### Path B
With Conda
Note that the python version is specified, meaning conda does not have to be associated with a python 3.6
```
conda create -n dervet-venv python=3.6
```

#### 8. Activate Python 3.6 environment
##### Path A
On Linux/Mac   
Note that pip should be associated to a python 3.6 installation  
```
source dervet-venv/bin/activate
```
On Windows  
Note that pip should be associated to a python 3.6 installation    
```
"./dervet-venv/Scripts/activate"
```
#### Path B
With Conda
Note that the python version is specified, meaning conda does not have to be associated with a python 3.6
```
conda activate dervet-venv
```

#### 9. Install project dependencies
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e ./storagevet
```

### Update Old Installation (Replace Beta Release Happy Path) For Windows

#### 1. Delete all folders in "C:\DERVET" expect "C:\DERVET\DervetBackEnd"

#### 2. Place the new "dervet" folder in "C:\DERVET\DervetBackEnd", placing the existing one

#### 3. Open Anaconda Prompt

#### 4. Activate Python 3.6 environment

```
conda activate "C:\DERVET\DervetBackEnd\"
cd C:\DERVET\DervetBackEnd\dervet
```

#### 5. Install project dependencies
```
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e ./storagevet
```

### Running Your First Case

#### 1. Activate Python environment. 
Skip this step if your python environment is already active. Refer to installation or update installation steps for activation instructions.

#### 2. Enter the following into your terminal from inside the root "dervet" folder:

```
python run_DERVET.py Model_Parameters_Template_DER.csv
```

## Running the tests

#### 1. Activate Python environment. 
Skip this step if your python environment is already active. Refer to installation or update installation steps for activation instructions.

#### 2. Enter the following into your terminal from inside the root "dervet" folder:
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

We use [Gitlab](https://gitlab.epri.com/storagevet/dervet) for versioning. For the versions available, 
see the [tags on this repository](https://gitlab.epri.com/storagetvet/dervet/tags). 

## Authors

* **Halley Nathwani**
* **Miles Evans**
* **Suma Jothibasu**
* **Ramakrishnan Ravikumar**
* **Andrew Etringer**
* **Andres Cortes**
* **Evan Giarta**
* **Thien Nguyen**
* **Micah Botkin-Levy**
* **Yekta Yazar**
* **Kunle Awojinrin**
* **Arindam Maitra**
* **Giovanni Damato**

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the BSD (3-clause) License - see [LICENSE.txt](./LICENSE.txt).

DER-VET v1.0.0

Copyright © 2021 Electric Power Research Institute, Inc. All Rights Reserved.

Permission to use, copy, modify, and distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright
notice and this permission notice appear in all copies.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL EPRI BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Third-Party Software
EPRI does not own any portion of the software that is attributed
below.

<CVXPY/1.1.11> - &lt;Steven Diamond&gt;, <diamond@cs.stanford.edu>
Copyright © 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CVXPY is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular
purpose are disclaimed.

This software relies on CVXPY to interface with work(s) covered by the
following copyright and permission notice(s): 

GLPK 5.0 - Andrew Makhorin, mao@gnu.org
Copyright © 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
2010, 2011, 2012 Andrew Makhorin, Department for Applied Informatics,
Moscow Aviation Institute, Moscow, Russia. All rights reserved.

Licensed under GNU Public License v3.0; you may not use GLPK except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/gpl-3.0.en.html.

GLPK is a free program and is provided by the copyright holders and
contributors "as is" and any express or implied warranties, including,
but not limited to, the implied warranties of merchantability and fitness
for a particular purpose are disclaimed.