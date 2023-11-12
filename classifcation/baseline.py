import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv('./Dataset/normalised_data.csv')

predictors = ['Neuroticism (N-Score)', 'Extroversion (E-Score)', 'Openness (O-Score)', 'Agreeableness (A-Score)',
              'Conscientiousness (C-Score)', 'Impulsiveness (BIS-11)', 'Sensation (SS)',
              'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54', 'Age_55-64', 'Age_65+',
              'Gender_F', 'Gender_M', 'Country_Australia', 'Country_Canada', 'Country_Ireland', 'Country_New Zealand',
              'Country_Other', 'Country_UK', 'Country_USA', 'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Other',
              'Ethnicity_White', 'Ethnicity_White-Asian', 'Ethnicity_White-Black',
              'Education_A', 'Education_B', 'Education_C', 'Education_D', 'Education_E', 'Education_F',
              'Education_G', 'Education_H', 'Education_I']
predicates = ['Nicotine']
total_set = predictors + predicates
df = dataset[total_set]

# Splitting data into training and test splits
predictor_total_set = df[predictors]
predicates_total_set = df[predicates]
X_train, X_test, y_train, y_test = \
    train_test_split(predictor_total_set, predicates_total_set, test_size=0.33, random_state=42)

# Create a baseline random classifier
# “stratified”: This strategy randomly selects class labels based on the class distribution in the training set.
# It aims to maintain the same class distribution as the training data, making it useful for imbalanced classes.
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)

# Fit the baseline classifier on the training data
dummy_clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dummy_clf.predict(X_test)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Baseline Classifier Accuracy:", accuracy)
print("Classification Report:")
print(report)
