# Medical Dataset Analysis 🏥

<img src="Medical Dataset/Heart_disease_code/Cardiologist-pana.png" alt="Roller Coaster" width="300" height="300">

## Overview

This project focuses on the analysis of a synthetic medical dataset designed to simulate patient data for heart disease research. The dataset includes various health metrics, demographic information, and treatment details, providing a comprehensive framework for machine learning model training and evaluation.

## Dataset Description 🗂️

The synthetic dataset consists of **10,000 samples** with the following features:

| Column Name      | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `blood_pressure` | Blood pressure in mmHg (range: 120-140)                      |
| `cholesterol`    | Cholesterol level in mg/dL (range: 200-250)                  |
| `bmi`            | Body Mass Index (BMI) (range: 18.5-30)                       |
| `glucose_level`  | Glucose level in mg/dL (range: 90-140)                       |
| `heart_disease`  | Target variable indicating heart disease presence (0 or 1)   |
| `gender`         | Gender of the patient (Male/Female)                          |
| `age`            | Age of the patient (range: 18-80)                            |
| `state`          | US state of residence (randomly selected from 50 states)     |
| `country`        | Country of residence (always 'USA')                          |
| `first_name`     | Randomly assigned first name of the patient                  |
| `last_name`      | Randomly assigned last name of the patient                   |
| `hospital`       | Hospital name where the patient received treatment           |
| `treatment`      | Type of treatment received (e.g., Physiotherapy, Medication) |
| `treatment_date` | Date of treatment (randomly generated over the past 5 years) |

## Data Analysis Steps 🔍

### 1. Importing Data 📥

The dataset is created and loaded into a pandas DataFrame for analysis. Relevant libraries such as `numpy`, `pandas`, and `sklearn` are imported for data manipulation and analysis.

### 2. Exploratory Data Analysis (EDA) 📈

Initial exploration of the dataset includes:

- Understanding the distribution of each feature.
- Identifying relationships between features and the target variable.
- Visualizing distributions and correlations using plots (e.g., histograms, scatter plots).

### 3. Outlier Detection and Handling 🚨

Outliers in the dataset are identified using statistical methods (e.g., `z-scores` or `IQR`). They are then visualized and, if necessary, removed to enhance model performance.

### 4. Data Cleaning 🧹

Data cleaning involves:

- Handling missing values (if any).
- Ensuring data types are correct for each feature.
- Encoding categorical variables (e.g., gender and treatment type) using one-hot encoding.

### 5. Class Balancing ⚖️

The dataset is checked for class imbalances in the target variable (`heart_disease`). If imbalances are detected, techniques such as:

- **Oversampling** the minority class.
- **Undersampling** the majority class.
- Using **SMOTE** (Synthetic Minority Over-sampling Technique).

### 6. Feature Engineering 🛠️

New features are created to enhance the dataset:

- Extracting additional date-related features from `treatment_date` (e.g., month, year, day of the week).
- Creating interaction terms or aggregating existing features where applicable.
- Extracting dummy variables from Gender column.

### 7. Training and Testing with Stratified K-Fold Cross-Validation 📊

- The dataset is split into training and testing sets using `Stratified K-Fold` cross-validation to ensure each fold has the same proportion of classes.
- A `Pipeline` is created for model training, ensuring consistent preprocessing across folds.

### 8. Model Training and Evaluation 🧠

Various machine learning models are trained using the processed data. Metrics such as accuracy and F1 score are used to evaluate model performance across folds.

### 9. Feature Importance 🔑

- The importance of each feature is calculated using model-based techniques (e.g., decision trees).
- Feature importance scores are plotted to identify which features contribute most to the prediction of heart disease.

### 10. Visualization 📊

Visualizations are created to showcase:

- The distribution of features.
- Outliers plot
- Model performance metrics.
- Feature importance rankings.
- Top 10 Hospitals.
- Top 10 Treatment by Hospital.

## Usage 🚀

This dataset and analysis can be used for various purposes including:

- **Machine Learning Models**: Building predictive models for heart disease.
- **Statistical Analysis**: Analyzing correlations between features.
- **Data Visualization**: Creating visual insights into health data.

## Installation 🛠️

Ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
