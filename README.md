
![ML Steps](Images/cover.png)<br />
# Pet Adoption

## Overview

The objective of this project is to develop a machine learning classification model that assists animal rescue organizations in identifying animals that are less likely to be adopted promptly. This will enable these organizations to transfer these animals to no-kill shelters or shelters that have a longer waiting period, thereby improving their chances of survival. The machine learning model will follow the following steps:<br />
<br />

![ML Steps](Images/ml_steps.png#gh-light-mode-only)<br />
![ML Steps](Images/ml_steps_dark.png#gh-dark-mode-only)<br />

The project also seeks to offer valuable insights, through EDA, into the characteristics that people prioritize when adopting a pet. <br />

## Presentation

### [Gamma.app Presentation Link](https://gamma.app/docs/Pet-Adoption-y53svrpmz3k75mv)<br />

## Questions Addressed during EDA

1. **What is the overall adoption rate of pets in the dataset? According to our data, the overall adoption rate for pets is around 33%. This means that for every 30 pets that enter the shelter system, 10 are successfully adopted by a loving family. This rate reflects the demand for pet companions across the country and the commitment of animal welfare organizations to finding homes for animals in need. Unfortunately, this also means that the remaining 20 pets may face humane euthanasia, as some shelters adhere to a 72-hour waiting period before making this decision.**<br />

![Pet Adoption Rates](Images/pet_adoption_rates.png#gh-light-mode-only)<br />
![Pet Adoption Rates](Images/pet_adoption_rates_dark.png#gh-dark-mode-only)<br />

2. **Do adoption rates vary by pet type? The data indicates that Dogs have the highest adoption rate at 46%, with Birds following at 30%, Cats at 29%, and Rabbits at 25%. This suggests that certain pet types may be more appealing to prospective adopters.**<br />

![Pet Type Adoption Rates](Images/pet_type_adoption_rate.png#gh-light-mode-only)<br />
![Pet Type Adoption Rates](Images/pet_type_adoption_rate_dark.png#gh-dark-mode-only)<br />

3. **What are the key factors that influence whether a pet gets adopted? We used the Chi-Square test to calculate the p-values for each feature in order to deteremine their importance. The six most important features were:**<br />

![Key Factors](Images/key_factors.png#gh-light-mode-only)<br />
![Key Factors](Images/key_factors_dark.png#gh-dark-mode-only)<br />

4. **Does the pet's age affect its likelihood of adoption? Younger pets are more energetic, trainable, and adaptable, making them a more desirable adoption choice with 2.5 times higher adoption rates. Older pets often face challenges in finding homes due to concerns about potential medical issues and lower activity levels. We used bins to divide the pets in age groups.**<br />

![Age Percentage](Images/age_adoption_perc.png#gh-light-mode-only)<br />
![Age Percentage](Images/age_adoption_perc_dark.png#gh-dark-mode-only)<br />

5. **Is there a correlation between the pet's breed and its adoption status? Labradors are one of the most popular dog breeds and consistently have high adoption rates of 72% as compared to 29% for Golden Retrievers and other breeds. Their friendly, intelligent, and adaptable nature make them a highly sought-after companion for many families. Some breeds may be unfairly stereotyped, causing potential adopters to overlook them despite their loving and loyal personalities. Persian cats have an adoption rate of 26% as they are stereotyped as not children friendly.**<br />

![Breed Adoption](Images/breed_adoption_perc.png#gh-light-mode-only)<br />
![Breed Adoption](Images/breed_adoption_perc_dark.png#gh-dark-mode-only)<br />

6. **Do specific attributes like color or size impact adoption rates? We found that color has little to no relation to the pet adoption rate. The data shows that medium-sized pets have the highest adoption rate at 62%, likely because they fit well into a variety of living spaces and lifestyles. Small pets come in second at 18%, potentially appealing to adopters with limited space or who prefer a more compact companion. Larger pets have the lowest adoption rate at 15%, possibly due to concerns over accommodating their size and energy requirements in certain homes.**<br />

![Size Percentage](Images/size_adoption_rate.png#gh-light-mode-only)<br />
![Size Percentage](Images/size_adoption_rate_dark.png#gh-dark-mode-only)<br />

7. **What is the average time it takes for pets to get adopted? The data shows that it takes 44 days for a pet to get adopted.**<br />

8. **How do different factors (age, breed, health status) affect the time to adoption? The time it takes for pets to find their forever homes does not vary significantly. On average, it takes pets 39-51 days depending on the different factors (pet type, breed, age,  color, size, vaccinated, health condition and previous owner).**<br />

9. **How do health conditions (e.g., disabilities) impact the adoption chances of pets? Healthy pets are also more likely to be adopted at 39%, versus 10% for pets with medical conditions. These statistics highlight the importance of proper medical care in preparing pets for adoption.**<br />

![Health Condition](Images/health_condition_adoption_rate.png#gh-light-mode-only)<br />
![Size Percentage](Images/size_adoption_rate_dark.png#gh-dark-mode-only)<br />

10. **How does the vaccination status of pets affect their adoption rates? Vaccinated pets have a 42% adoption rate, compared to 11% for unvaccinated pets.**<br />

![Size Percentage](Images/vaccination_adoption_rate.png#gh-light-mode-only)<br />
![Size Percentage](Images/vaccination_adoption_rate_dark.png#gh-dark-mode-only)<br />

## Machine Learning Model Steps

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44)
    ```

5. **Standardization**:
    - Numerical features are standardized to have a mean of 0 and a standard deviation of 1.
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

6. **Encoding**:
    - Categorical features are encoded using one-hot encoding and ordinal encoding (size).
    ```python
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    encoder = OrdinalEncoder(categories=[['Small','Medium','Large']], encoded_missing_value=-1)
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    ```

7. **Synthetic Sampling**:
    - Synthetic sampling techniques like RandomOverSampler are used to handle class imbalance.
    ```python
    ros = RandomOverSampler(random_state=44)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    ```
8. **Model Selection**:
    - Various machine learning models are trained and evaluated:
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
    
    - Results of the models:<br />
    <br /> 
![Model Results](Images/model_results.png#gh-light-mode-only)<br /> 
![Model Results](Images/model_results_dark.png#gh-dark-mode-only)<br /> 


9. **Optimization**:
    - Best Depth and Hyperparameter tuning is performed using GridSearchCV
    ```python
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    ```

    - Results after optimization:<br />
    <br /> 
![Model Results](Images/model_results_optimization.png#gh-light-mode-only)<br />
![Model Results](Images/model_results_optimization_dark.png#gh-dark-mode-only)<br /> 

10. **Important Features**
    - We used the important features method with the Ada Boost and Random Forest classifier that had the best results.
    - ADA Model:<br /> 
![Model Results](Images/ada_important_features.png#gh-light-mode-only)<br /> 
![Model Results](Images/ada_important_features_dark.png#gh-dark-mode-only)<br /> 
    - Random Forest Model:<br /> 
![Model Results](Images/rfc_important_features.png#gh-light-mode-only)<br /> 
![Model Results](Images/rfc_important_features_dark.png#gh-dark-mode-only)<br /> 
    - The important features of the two models and the ones derived using the Chi-Square test yielded different results.

## Conclusion

Based on our comprehensive analysis of pet adoption data, we've identified key insights and strategies to help increase adoption rates nationwide. By focusing on areas like pet attributes we can create a more positive and sustainable future for shelter animals.

We were successful in developing a machine learning model that achieves a precision score exceeding 95%. This model can assist rescue workers in transferring animals to shelters with no-kill policies or longer waiting periods, thereby enhancing their prospects for survival.

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
