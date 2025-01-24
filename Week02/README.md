# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course provides a comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs).


## Week02: Developing statistical models from EEG data
This class we spent time analyzing EEG signals to detect seizures. The analysis includes statistical tests, visualization, and linear regression modeling. The project is divided into several scripts and a Jupyter Notebook, each performing specific tasks.

## Files and Their Functions

### 1. AnalyzeSignalSofT.R

This R script performs the following tasks:
- Reads the EEG signal data from a CSV file.
- Creates box plots to visualize the signal distribution for different classes (Normal and Seizure).
- Performs t-tests and Wilcoxon rank-sum tests to compare the means of the two classes.
- Plots the density of the signals for both classes.
- Visualizes in-control vs. out-of-control signals using 2 SD and 1 SD rules.
- Saves the modified data with control rules to a new CSV file.

### 2. linearRegressionFit.R

This R script fits a linear regression model to predict the normalized value of a signal using the un-normalized value. It performs the following tasks:
- Reads the modified EEG signal data with control rules from a CSV file.
- Fits a linear regression model using the `lm` function.
- Outputs the estimated coefficients of the model.

### 3. OLSsolution.m

This MATLAB script demonstrates the ordinary least squares (OLS) method to fit a linear model. It performs the following tasks:
- Generates synthetic data points.
- Defines parameters for the linear model.
- Calculates and plots the y values.
- Fits the linear model to the data and estimates the parameters using OLS.

### 4. ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb

This Jupyter Notebook performs the following tasks:
- Reads EEG signal data from a .mat file.
- Plots the raw signal.
- Computes and plots the spectrogram of the signal.

## How to Run the Scripts

### Prerequisites

- R and RStudio
- MATLAB
- Python with Jupyter Notebook
- Required R libraries: `readr`, `dplyr`
- Required Python libraries: `scipy`, `numpy`, `matplotlib`, `prettytable`

### Steps

1. **AnalyzeSignalSofT.R**
   - Open the script in RStudio.
   - Ensure the CSV file `s_of_t_subset_CLEAN.csv` is in the working directory.
   - Run the script.

2. **linearRegressionFit.R**
   - Open the script in RStudio.
   - Ensure the CSV file `s_of_t_subset_CLEAN_with2SDrule.csv` is in the working directory.
   - Run the script.

3. **OLSsolution.m**
   - Open the script in MATLAB.
   - Run the script.

4. **ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb**
   - Open the notebook in Jupyter.
   - Ensure the .mat file `session4_train_2018.mat` is in the specified path.
   - Run the notebook cells.

s