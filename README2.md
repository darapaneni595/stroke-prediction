# Stroke Prediction

This project aims to predict the likelihood of a stroke based on various health and lifestyle features. It uses Exploratory Data Analysis (EDA) to analyze the dataset and applies six different machine learning models to predict stroke outcomes.

## Objective
To predict whether a person will have a stroke based on their health and lifestyle factors. The dataset includes attributes such as age, hypertension, heart disease, marital status, and more.

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting
- Decision Trees
- K-Nearest Neighbors (KNN)

## Key Steps
1. **Data Preprocessing**: Handle missing values, feature encoding for categorical variables, and feature scaling if necessary.

2. **Exploratory Data Analysis (EDA)**: Visualize distributions and relationships between features. Identify patterns and correlations.

3. **Model Building**: Split the data into training and test sets, train models using the training data, and evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

4. **Model Evaluation**: Compare the performance of different models and select the best model based on evaluation metrics.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`

## Installation
To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction

2. Install the required dependencies:

bash
Copy code
pip install -r requirements.txt

3.Run the Jupyter Notebook to view the full analysis:

bash
Copy code
jupyter notebook stroke-eda-prediction.ipynb
