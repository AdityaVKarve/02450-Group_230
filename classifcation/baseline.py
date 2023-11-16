import nicotine_df
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report


def baseline():
    df = nicotine_df.df

    # Splitting data into training and test splits
    predictor_total_set = df[nicotine_df.predictors]
    predicates_total_set = df[nicotine_df.predicates]
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
    return dummy_clf.predict(nicotine_df.mcnemra_train_set)
