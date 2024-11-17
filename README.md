# Death-Predictor-using-hospital-data

## Overview
This project predicts the age of death using demographic data (education, sex, race, marital status) and cause of death information (ICD codes). By leveraging machine learning models, the project identifies key factors influencing age predictions and visualizes their importance.

## Features
- **Data Preprocessing**: Cleans and preprocesses demographic and ICD code data.
- **Feature Engineering**: Combines `Marital_Status` and `Sex`, maps ICD codes to broad causes of death, and applies one-hot encoding.
- **Machine Learning Models**:
  - Linear Regression
  - Ridge Regression
  - Gradient Boosting Regression
- **Evaluation**: Evaluates models using Mean Squared Error (MSE), Root Mean Squared Error(RMSE), and \( R^2 \) Score.
- **Visualization**:
  - Predicted vs. Actual Age scatter plots for all models.
  - Feature importance for Gradient Boosting Regression.
- **Output**: Saves the cleaned dataset and analysis results.

## Dataset
The input data is read from a fixed-width text file (`data.txt`) with the following columns:
- **Education**: Number of years of formal education.
- **Sex**: Binary (1: Male, 2: Female).
- **Race**: Encoded racial categories.
- **Age**: Age at the time of death.
- **Marital_Status**: Encoded marital status.
- **ICD_Code**: Cause of death based on the International Classification of Diseases (ICD).
