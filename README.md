## Overview

This project analyzes pet adoption data to uncover factors that influence adoption rates and build a classification model to predict adoption status.

### [Presentation Link](https://gamma.app/docs/Pet-Adoption-y53svrpmz3k75mv)


## Questions Addressed

1. What is the overall adoption rate of pets in the dataset?
2. Do adoption rates vary by pet type?
3. What are the key factors that influence whether a pet gets adopted?
4. Does the pet's age affect its likelihood of adoption?
5. Is there a correlation between the pet's breed and its adoption status?
6. Do specific attributes like color, breed, or health status impact adoption rates?
7. What is the average time it takes for pets to get adopted?
8. How do different factors (age, breed, health status) affect the time to adoption?
9. How do special conditions (e.g., disabilities) impact the adoption chances of pets?
10. How does the vaccination status of pets affect their adoption rates?

## Steps

1. **Libraries Used**:
    ```python
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from sklearn.decomposition import PCA
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.metrics import balanced_accuracy_score, classification_report, recall_score
    ```

2. **Dataset**:
    
    ### [Predict Pet Adoption Status Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)
    

3. **Exploratory Data Analysis (EDA)**:
    - The notebook performs EDA to answer the questions listed above, using various visualization techniques.
    ```python
    df.head()
    df.describe()
    df.info()
    ```

4. **Splitting Data**:
    - The data is split into training and testing sets.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

5. **Standardization**:
    - Numerical features are standardized to have a mean of 0 and a standard deviation of 1.
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

6. **Encoding**:
    - Categorical features are encoded using one-hot encoding or ordinal encoding.
    ```python
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    ```

7. **Synthetic Sampling**:
    - Synthetic sampling techniques like RandomOverSampler are used to handle class imbalance.
    ```python
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    ```
8. **Model Selection**:
    - Various machine learning models are trained and evaluated.
    ```python
    models = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': tree.DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }
    ```

9. **Optimization**:
    - Hyperparameter tuning is performed using GridSearchCV
    ```python
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    ```

## Conclusion

This notebook provides a comprehensive analysis of pet adoption data and builds a classification model to predict adoption outcomes based on various features. The steps include data preprocessing, exploratory data analysis, model training, and optimization.

## Requirements

- pandas
- numpy
- plotly
- matplotlib
- scikit-learn
- imbalanced-learn

## How to Use

1. Clone the repository.
2. Install the required packages.
3. Run the Jupyter notebook to explore the analysis and results.

## Contributors
Ramona Ciobanu, Emmanuel Charles, Enock Mudzamiri, Ezra Timmons, Jaidev Kler and Oliver Tabibzadeh