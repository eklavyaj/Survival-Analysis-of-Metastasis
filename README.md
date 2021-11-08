# Survival Analysis of Distant Metastasis in Breast Cancer Patients

## Contents
1. Objective
2. Dataset
3. Survival Function
4. Implementation of Predictive Models
5. COBRA Implementation and Results
7. Directory Structure
8. References

## Objective

We use the Transbig Dataset to predict the survival function for distant metastasis in breast cancer patients using survival analysis and combined regression strategies. 


## Dataset

* The data from `TRANSBIG` validation study of 198 patients is used to perform the analysis.
* In TRANSBIG, the datasets `Patient Characteristic and Diagnostic Details`, and `Gene Features` are joined.
* Patient Characteristic and Diagnostic Details contains the clinical information of the patients.
* The observations in this dataset are censored, in the sense that for some units the event of interest has not occured at the time the data was analyzed or collected.
* Gene Features dataset contains the information of 22283 genes features of the patients.

 
## What is Survival Function?

By definition survival function is a function that gives the probability that a patient will survive beyond any specified time. 
Mathematically, if T is a continuous random variable with pdf f(t) and cdf F(t). Then the probability that the patient suffered distant metastasis by time duration t is nothing but the survival function. 

![image](https://user-images.githubusercontent.com/50804314/140692638-e80749d1-3662-4d80-9b99-a638ab61483b.png)

## Implementation of Predictive Models
Multiple ML Models are applied to predict the survival function:
<ul>
 <li> Support Vector Machine (svm.SVC)
 <li> KNeighborsClassifier
 <li> DecisionTreeClassifier
 <li> Gaussian Naive Bayes
 <li>  LinearDiscriminantAnalysis
</ul>

In addition to this, we also implemented Random Survival Forest to predict the survival function in two ways. 
* Using the python library `Random Survival Forest`
* Native Implementation by defining Indicator variables for various values of `t`, and using Regression models to predict these, and calculating survival function as in above formula. 

## COBRA Implementation and Results

Cobra implementation is done from scratch by defining a class in python and using various models to combine them through voting. 

Applying Random Survival Forest yielded an `R^2` value of 0.65. Later, applying Cobra increased the `R^2` value to 0.77. 

In addition to `R^2` analysis, the survival function is plotted against the time in days, and it agrees with the mathematical behavior and python implementation of Random Survival Forest.

COBRA:

![image](https://user-images.githubusercontent.com/50804314/140694209-2c29fe58-440c-408d-aa9f-b3fdf2fd473f.png)

Random Survival Forest:

![image](https://user-images.githubusercontent.com/50804314/140694262-edf24fc5-a91d-4037-852a-2d2ce671a8df.png)


## Directory Tree:

```
MA691-COBRA-3
|
└───README.md
|   
│
└───Documentation  // TRANSBIG Dataset README and other documentations
|
│
└───Data
│   │   GSE7390_family.soft.gz    // TRANSBIG Dataset
│   │   cleaned_data.csv          // Cleaned Data with all patient characterstics and gene features 
│   │   gene.csv                  // List of 76 most important genes to analyze time to distant metastasis
|   |   selected.csv              // Subset of cleaned data containing only 76 important gene features and patient characteristics
│      
└───Literature   // Various research papers that we used and implemented in our work
│   
│   
└───Notebooks
|   │   cobra.ipynb            // COBRA Implementation 
|   │   data_cleaning.ipynb    // Cleaning and extracting relevant data from TRANSBIG zip
|   │   indicators.ipynb       // Native Implementation to predict Survival Function
|   │   regression.ipynb       // Application of various regression models to predict time to distant metastasis
|   │   survival.ipynb         // Application of Random Survival Forest of Scikit-learn
|
|
└───Scripts
|   │   cobra_estimator.py     // Definition of Cobra Class
|   │   data_clean.py          // Script to extract, clean and save relevant data in a new csv file
|   │   main.py                // Main script which calls the COBRA model
|   │   random_survival.py     // Script that runs in-built Random Survival Forest 
|
```

