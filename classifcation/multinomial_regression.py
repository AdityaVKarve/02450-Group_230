import nicotine_df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = nicotine_df.df

def multinomial():
    # split data into training and test sets
    X = df[nicotine_df.predictors]  # personal data
    y = df[nicotine_df.predicates]  # nicotine

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Define a range of lambda values to explore
    C_values = np.logspace(-3, 3, 7)  # This creates a range from 0.001 to 1000

    best_C = None
    best_accuracy = 0
    best_report = None

    best_mm_model = None
    for C in C_values:
        # create and train the multinomial logistic regression model
        multinomial_model = LogisticRegression(multi_class='multinomial', max_iter=5000, random_state=42, C=C)
        multinomial_model.fit(X_train, y_train)

        y_pred = multinomial_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Check if this model has better accuracy than previous ones
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_report = classification_report(y_test, y_pred)
            best_C = C
            best_mm_model = multinomial_model

    # Print the best C value and corresponding accuracy
    print("Best C:", best_C)
    print("Best Accuracy:", best_accuracy)
    print("Classification Report:\n", best_report)
    return best_mm_model.predict(nicotine_df.mcnemra_train_set)