---
title: "StoragetVET Py"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites & Installing

#### 1. Install [Anaconda](https://www.anaconda.com/download/) for python 3.**

#### 2. Install Microsoft Visual C++ 14.0

- Download and install Build Tools 2017 for [Visual Studio 2017](https://visualstudio.microsoft.com/downloads/).

- Open the Visual Studio Installer and select Visual Studio Build Tools to be installed, but do not install yet

- Select the appropriate "Windows SDK" specified below:

| Windows OS   | SDK        |
|--------------|------------|
| Windows 7    | Windows 8.1|
| Windows 8.1  | Windows 8.1|
| Windows 10   | Windows 10 |


- Install Visual Studio Build Tools

#### 3. Open Anaconda Prompt and install the following dependencies

- Install ecos 
    
```
conda install ecos
pip install ecos
```
When installing with conda, you maybe be prompted by the command prompt. Do not worry, it is just asking if you want to install the library. To continue, type "y" and enter in the command prompt.

- Install cvxopt 
    
```
conda install cvxopt
pip install cvxopt
```
You will be prompted again. Just keep going.

- Install cvxpy 
    
```
pip install cvxpy
```

## Running the tests

To run tests, simply enter the following into your terminal:
```
Python/Testing/runSvetTests.bat
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Gitlab](http://gitlab.com/) for versioning. For the versions available, see the [tags on this repository](https://gitlab.epri.com/storagetvet/SVETpy/tags). 

## Authors

* **Miles Evans**
* **Andres Cortes**
* **Evan Giarta**
* **Halley Nathwani**
* **Micah Botkin-Levy**

## License

This project is licensed under the BSD (3-clause) License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
