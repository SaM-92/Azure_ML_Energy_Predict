# README for Energy Efficiency ML Project


## Overview
This repository contains a machine learning project focused on predicting the thermal load of buildings. It leverages the Energy Efficiency Dataset to develop a model that accurately estimates a building's energy performance, particularly its thermal load. The project follows a structured workflow, including data handling, model training, and evaluation, primarily executed in Azure and leveraging MLFlow for metric tracking.

## Dataset
We use the Energy Efficiency Dataset, originally curated by Angeliki Xifara and Athanasios Tsanas at the University of Oxford, UK. It consists of 768 samples and 8 features, simulating different building shapes to predict thermal load. The dataset is in Excel format and is incorporated into our workspace for machine learning purposes.

## Attribute Information:
X1: Relative Compactness
X2: Surface Area
X3: Wall Area
X4: Roof Area
X5: Overall Height
X6: Orientation
X7: Glazing Area
X8: Glazing Area Distribution
y1: Heating Load
y2: Cooling Load

## Workflow
Data Reading: A Jupyter notebook reads data from a local machine.
Data Upload and Registration: The dataset is uploaded and registered into Azure as AssetTypes.URI_FILE.
Cluster Creation: The workflow includes steps to create ML clusters in Azure.
Model Training Script: A Python script (train.py) is provided for model training. Key functions in the script include:
get_data(): Reads the data.
split_data(): Splits the data into training and testing sets.
train_model(): Trains the model using a RandomForestRegressor.
eval_model(): Evaluates the model using metrics like RMSE and R-squared.
MLFlow Integration: We use MLFlow for logging metrics during model training and evaluation.

##Execution
To run the project:

- Ensure Azure ML environment is set up with required clusters.
- Execute the provided notebook to handle data upload and registration.
- Run the train.py script for training and evaluating the model.
- Monitor the performance metrics through MLFlow.
  
## Objective
The primary aim is to predict the thermal load of buildings using machine learning techniques. This project, while focusing on thermal load prediction, provides a framework that can be adapted for broader energy efficiency studies.

## Dependencies
Azure ML SDK
MLFlow
Scikit-Learn
Pandas
Numpy

# Note
This project serves as a template for machine learning applications in energy efficiency and can be extended or modified for related tasks in the domain.
