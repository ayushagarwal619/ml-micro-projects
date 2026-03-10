# ModelForge – CPU Usage Prediction
This project implements a machine learning solution for the **ModelForge competition**, where the objective is to predict CPU user time (`usr`) using system activity metrics.

## Problem
Predict CPU user time (`usr`) based on system-level performance data such as I/O activity, memory usage, system calls, and process behavior.

## Models Used
* XGBoost
* LightGBM
* CatBoost
An ensemble of these gradient boosting models is used to improve prediction performance.

## Techniques
* Feature Engineering
* Log Transformations
* K-Fold Cross Validation
* Model Ensembling

## Output
The final predictions are saved in `submission2.csv` following the competition submission format.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost
