import nicotine_df
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

df = nicotine_df.df

def naive_bayes():
    # split data into training and test sets
    X = df[nicotine_df.predictors]  # personal data
    y = df[nicotine_df.predicates]  # nicotine

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # create and train the Naive Bayes classifier for alpha cross-validation
    naive_bayes_model = MultinomialNB()

    # define the parameter grid to search
    param_grid = {'alpha': np.logspace(-3, 3, 7)}  # This creates a range from 0.001 to 1000

    # Choose a scoring metric (e.g., accuracy) for evaluation
    scorer = make_scorer(accuracy_score)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(naive_bayes_model, param_grid, scoring=scorer, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameter value
    best_alpha = grid_search.best_params_['alpha']
    print("Best alpha:", best_alpha)

    # create and train the model with the best alpha on the entire training set
    best_naive_bayes_model = MultinomialNB(alpha=best_alpha)
    best_naive_bayes_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = best_naive_bayes_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_output = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report_output)


    return best_naive_bayes_model.predict(nicotine_df.mcnemra_train_set)

naive_bayes()