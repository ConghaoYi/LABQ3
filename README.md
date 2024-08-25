# LABQ3
**Description:** **L**inear **A**ttenuation **B**ayesian **Q**uantitative **3**D-mapper (**LABQ3**) is a method for quantifying the chemical composition of XCT data. It takes the measured attenuation coefficient ($\mu$) from XCT scans as the input. Using the data from mineral standards as references, LABQ3 quantifies the input data via Bayesian decision theory along a continuous compositional spectrum. 

# Dependencies
The script was built on Python 3.10.9.

The script was built on the following Python packages:
* NumPy 1.24.2
* SciPy 1.10.1
* Matplotlib 3.7.1
* scikit-image 0.21.0
* Pillow 9.5.0
* xraylib 4.1.5
* pytest 7.1.2

# Usage
## Binary case:
### 1. **Input parameters**
* std1_name: This is the name of one of the standard minerals that should be defined by the user. 
* std2_name: This is the name of one of the standard minerals that should be defined by the user. 
* dir_scan: The complete path to the scan that needs to be quantified. This directory is expected to be the folder with every slice of the scan in txt or npy format.
* dir_out:  The complete path for the output results.
* dir_std1: The complete path for the folder that contains the text images of only standard mineral 1.
* dir_std2: The complete path for the folder that contains the text images of only standard mineral 2.
* step: The chemical composition resolution in percentage 1 = 1%
* scale: The scale for downsampling the input scan using skimage.measure.block_reduce.
* erosion_filter: Select if erosion filter should be applied to the scan. True for applying and False for not applying.
* median_filter: Select if median filter should be applied to the scan. True for applying and False for not applying.
* erosion_filter: The scale for erosion filter.
* erosion_filter: The scale for median filter.
* use_pre_existing: Use previously compute Bayesian classifier and distor if available for these configuration parameters
* unique_run_key: Should be a string that uniquely identifies the Bayesian classifier to use

### 2. **Execution**
See LABQ3_run for example use and a config

## Ternary case:
### 1. **Input parameters**
* std1_name: This is the name of one of the standard minerals that should be defined by the user. 
* std2_name: This is the name of one of the standard minerals that should be defined by the user.
* std3_name: This is the name of one of the standard minerals that should be defined by the user.  
* dir_scan_8kev: The complete path to the scan at 8keV that needs to be quantified. This directory is expected to be the folder with every slice of the scan in txt or npy format.
* dir_scan_8kev: The complete path to the scan at 10keV that needs to be quantified. This directory is expected to be the folder with every slice of the scan in txt or npy format.
* dir_out:  The complete path for the output results.
* dir_std1_8kev: The complete path for the folder that contains the text images of only standard mineral 1 at 8keV.
* dir_std2_8kev: The complete path for the folder that contains the text images of only standard mineral 2 at 8keV.
* dir_std3_8kev: The complete path for the folder that contains the text images of only standard mineral 3 at 8keV.
* dir_std1_10kev: The complete path for the folder that contains the text images of only standard mineral 1 at 10keV.
* dir_std2_10kev: The complete path for the folder that contains the text images of only standard mineral 2 at 10keV.
* dir_std3_10kev: The complete path for the folder that contains the text images of only standard mineral 3 at 10keV.
* step: The chemical composition resolution in percentage 1= 1%.
* scale: The scale for downsampling the input scan using skimage.measure.block_reduce.
* erosion_filter: Select if erosion filter should be applied to the scan. True for applying and False for not applying.
* median_filter: Select if median filter should be applied to the scan. True for applying and False for not applying.
* erosion_filter: The scale for erosion filter.
* erosion_filter: The scale for median filter.
* use_pre_existing: Use previously compute Bayesian classifier and distor if available for these configuration parameters
* unique_run_key: Should be a string that uniquely identifies the Bayesian classifier to use


### 2. **Example usage**
See LABQ3_run for example use and the config within the script.

# Output
The output is the quantification result for each slice in .npy format. The result is the volume fractions of each standard mineral. The volume fraction of a standard mineral is stored in a folder corresponding to the standard mineral (e.g., the volume fraction of standard mineral 1 is stored in the folder "std1").

Additionally, a "ratio.txt" text file is generated. This text file contains the details of the quantification. 

# Test
## 1. Pytest
***NOTE: DO NOT CHANGE THE test_LABQ3.py SCRIPT. THIS SCRIPT IS MEANT TO TEST WHETHER LABQ3 CAN RUN AS EXPECTED. THE RELEVANT VARIABLES AND VALUES WERE HARD-CODED INTO THE SCRIPT TO TEST IF LABQ3 IS WORKING PROPERLY. ALL TESTS SHOULD PASS.***

Enter the command "pytest" in the terminal at the source directory to run the test_LABQ3.py script. This script is used to test if LABQ3 is working properly. A total of 5 tests will be performed. The tests are reproducing the errors of LABQ3, the metal contents at the rim and core, and the results in Table 1, which are all reported in the manuscript. (It takes time to run the script when running it for the first time as it needs to generate data.)

## 2. Example outputs
Run the LABQ3_run.py script to generate example outputs.