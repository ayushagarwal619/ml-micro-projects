# ModelForge – CPU Usage Prediction
This repository contains my solution for the **ModelForge: The ML Showdown** competition on Kaggle, organized as part of **Dakshh**, the annual technical fest of Heritage Institute of Technology, Kolkata.
The objective of the competition is to build a machine learning model that predicts **CPU user time (`usr`)** based on system activity metrics such as I/O operations, memory usage, and system calls.
🔗 Competition Link: https://www.kaggle.com/competitions/model-forge-dakshh

## Problem Statement
Predict the CPU user time (`usr`) of a system using performance indicators including I/O activity, memory operations, system calls, and process behavior.

## Models Used
* XGBoost
* LightGBM
* CatBoost
An ensemble of gradient boosting models is used to improve prediction accuracy.

## Techniques
* Feature Engineering
* Log Transformations
* K-Fold Cross Validation
* Model Ensembling

## Output
The final predictions are saved in **`submission2.csv`**, following the required Kaggle submission format.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost
