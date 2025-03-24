# Flight Delay Prediction using Machine Learning

## Project Overview
The goal of this project is to predict flight delays based on historical data using machine learning techniques. We aim to build a model that can forecast whether a flight will be delayed or not.

## Objective
- Predict flight delays with a machine learning model.
- Focus on understanding how external factors (e.g., weather, airline, and airport data) influence flight delays.

## Technologies
- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - Numpy
  - Scikit-learn
  - XGBoost
- **Model**: XGBoost Classifier
- **Data Source**: Flight delay dataset (available on [Kaggle](https://www.kaggle.com))

## Installation
To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flight-delay-prediction.git
   cd flight-delay-prediction

## Model Performance
- **Accuracy**: 0.91
- **Confusion Matrix**: [Confusion Matrix results burada yer alabilir]
- **Classification Report**: [Classification report burada yer alabilir]

## Data Preprocessing
- Handle missing values with median imputation for numerical columns.
- Categorical columns encoded using LabelEncoder.
- Feature engineering: Removed columns like `PERSON_AGE` for improved model performance.

## Future Improvements
- Explore different machine learning models like Random Forest or SVM.
- Implement hyperparameter tuning using GridSearchCV.
- Incorporate additional features like historical weather data.
