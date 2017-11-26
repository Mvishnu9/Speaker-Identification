# Speaker-Identification
A comparitive study on different Classifiers for the problem of Speaker Identification.
In-depth information for each of the classifier and Feature Extraction has been provided in the Report

## The Following classifiers have been used 
* Gaussian Mixture Model
* Support Vector Machine
* K Nearest Neighbours
* Radius Nearest Neighbours
* Neural Network

The results of each of these are provided in raw format in Results.xlsx

## Running the code
1. Download the TIMIT dataset and reorganise it so that same speakers are present in both train and test. Split in 7:3 ratio between train and test.
2. Convert files into .wav format.
3. Run `python Preprocess.py` (TRAIN and TEST folders must be in same directory)
4. Run `python KNN.py` `python SVM.py` etc.

### This is part of our Statistical Methods in A.I. course project with the following team
1. Vishnu M.
2. Navya Khare
3. Tanmay Joshi
4. Shubhangi Gautam
